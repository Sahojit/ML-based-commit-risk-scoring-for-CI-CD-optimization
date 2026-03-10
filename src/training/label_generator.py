
import logging
import pandas as pd
import re
from typing import List, Set
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LabelGenerator:
    
    def __init__(self, bug_keywords: List[str] = None):
        self.bug_keywords = bug_keywords or [
            'fix', 'bug', 'error', 'issue', 'patch', 
            'hotfix', 'bugfix', 'defect', 'correct'
        ]
        logger.info(f"LabelGenerator initialized with {len(self.bug_keywords)} keywords")
    
    def generate_labels(self, commits_df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Generating labels for {len(commits_df)} commits")
        
        labels_df = commits_df[['commit_hash', 'message']].copy()
        
        labels_df['is_buggy'] = labels_df['message'].apply(
            lambda msg: self._is_bug_fix(msg)
        )
        
        labels_df['labeled_by'] = 'keyword_heuristic'
        labels_df['confidence'] = labels_df['is_buggy'].apply(
            lambda x: 0.8 if x == 1 else 0.6
        )
        
        buggy_count = labels_df['is_buggy'].sum()
        clean_count = len(labels_df) - buggy_count
        
        logger.info(f"Labeling complete:")
        logger.info(f"  Buggy commits: {buggy_count} ({buggy_count/len(labels_df)*100:.1f}%)")
        logger.info(f"  Clean commits: {clean_count} ({clean_count/len(labels_df)*100:.1f}%)")
        
        return labels_df[['commit_hash', 'is_buggy', 'labeled_by', 'confidence']]
    
    def _is_bug_fix(self, message: str) -> int:
        if pd.isna(message):
            return 0
        
        message_lower = message.lower()
        
        for keyword in self.bug_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, message_lower):
                return 1
        
        return 0
    
    def get_bug_keywords_used(self, commits_df: pd.DataFrame) -> pd.DataFrame:
        keyword_counts = {}
        
        for keyword in self.bug_keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            count = commits_df['message'].str.lower().str.contains(
                pattern, na=False, regex=True
            ).sum()
            keyword_counts[keyword] = count
        
        stats_df = pd.DataFrame({
            'keyword': keyword_counts.keys(),
            'count': keyword_counts.values()
        }).sort_values('count', ascending=False)
        
        return stats_df
    
    def save_labels(self, labels_df: pd.DataFrame, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        labels_df.to_csv(output_path, index=False)
        logger.info(f"Labels saved to {output_path}")
        logger.info(f"Total records: {len(labels_df)}")
    
    def validate_labels(self, labels_df: pd.DataFrame) -> dict:
        validation = {
            'total_commits': len(labels_df),
            'buggy_commits': int(labels_df['is_buggy'].sum()),
            'clean_commits': int((labels_df['is_buggy'] == 0).sum()),
            'bug_ratio': float(labels_df['is_buggy'].mean()),
            'missing_labels': int(labels_df['is_buggy'].isna().sum()),
            'avg_confidence': float(labels_df['confidence'].mean())
        }
        
        if validation['bug_ratio'] < 0.05:
            logger.warning(f"Severe class imbalance: only {validation['bug_ratio']:.1%} buggy commits")
        elif validation['bug_ratio'] < 0.2:
            logger.info(f"Moderate class imbalance: {validation['bug_ratio']:.1%} buggy commits")
        
        return validation

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("TESTING LABEL GENERATOR")
    logger.info("=" * 70)
    
    sample_commits = pd.DataFrame({
        'commit_hash': ['abc123', 'def456', 'ghi789', 'jkl012', 'mno345'],
        'message': [
            'Fix login bug',
            'Add new feature',
            'Bugfix: correct validation',
            'Update documentation',
            'Error handling in payment module'
        ]
    })
    
    print("\nSample commits:")
    print(sample_commits)
    
    generator = LabelGenerator()
    
    labels = generator.generate_labels(sample_commits)
    
    print("\nGenerated labels:")
    print(labels)
    
    keyword_stats = generator.get_bug_keywords_used(sample_commits)
    print("\nKeyword statistics:")
    print(keyword_stats)
    
    validation = generator.validate_labels(labels)
    print("\nValidation metrics:")
    for key, value in validation.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Label generator test complete!")