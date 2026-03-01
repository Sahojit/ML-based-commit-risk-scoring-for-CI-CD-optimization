"""
Developer-Level Feature Engineering
Extracts features about commit authors
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeveloperFeatureExtractor:
    """
    Extracts developer-level features
    """
    
    def __init__(self, recent_window_days: int = 30):
        """
        Initialize DeveloperFeatureExtractor
        
        Args:
            recent_window_days: Number of days to consider for "recent" activity
        """
        self.recent_window_days = recent_window_days
        logger.info(f"DeveloperFeatureExtractor initialized (recent window: {recent_window_days} days)")
    
    def extract_features(
        self,
        commits_df: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Extract developer-level features
        
        Args:
            commits_df: DataFrame with commit data
            labels_df: DataFrame with labels
        
        Returns:
            DataFrame with developer features for each commit
        """
        logger.info(f"Extracting developer features for {commits_df['author'].nunique()} developers")
        
        # Merge commits with labels
        data = commits_df.merge(labels_df, on='commit_hash', how='left')
        
        # Convert timestamp to datetime
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Calculate developer-level aggregates
        dev_features = self._calculate_developer_stats(data)
        
        # Merge back to commits
        result = commits_df[['commit_hash', 'author']].merge(
            dev_features,
            on='author',
            how='left'
        )
        
        logger.info("Developer features extracted successfully")
        
        return result
    
    def _calculate_developer_stats(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistics for each developer
        
        Args:
            data: DataFrame with commits and labels
        
        Returns:
            DataFrame with developer statistics
        """
        # Group by author
        dev_stats = []
        
        for author in data['author'].unique():
            author_commits = data[data['author'] == author]
            
            # Total commits
            total_commits = len(author_commits)
            
            # Bug rate (proportion of buggy commits)
            buggy_commits = author_commits['is_buggy'].sum()
            bug_rate = buggy_commits / total_commits if total_commits > 0 else 0
            
            # Recent activity (commits in last N days)
            if 'timestamp' in author_commits.columns:
                latest_date = author_commits['timestamp'].max()
                cutoff_date = latest_date - timedelta(days=self.recent_window_days)
                recent_commits = author_commits[author_commits['timestamp'] >= cutoff_date]
                recent_frequency = len(recent_commits)
            else:
                recent_frequency = 0
            
            # Average commit size
            avg_lines_added = author_commits['lines_added'].mean() if 'lines_added' in author_commits.columns else 0
            avg_lines_deleted = author_commits['lines_deleted'].mean() if 'lines_deleted' in author_commits.columns else 0
            avg_files_changed = author_commits['files_changed'].mean() if 'files_changed' in author_commits.columns else 0
            
            dev_stats.append({
                'author': author,
                'total_commits': total_commits,
                'buggy_commits': int(buggy_commits),
                'bug_rate': bug_rate,
                'recent_frequency': recent_frequency,
                'avg_lines_added': avg_lines_added,
                'avg_lines_deleted': avg_lines_deleted,
                'avg_files_changed': avg_files_changed
            })
        
        return pd.DataFrame(dev_stats)
    
    def get_feature_statistics(self, features_df: pd.DataFrame) -> dict:
        """
        Get statistics about developer features
        
        Args:
            features_df: DataFrame with developer features
        
        Returns:
            Dictionary with statistics
        """
        # Get unique developers
        unique_devs = features_df.drop_duplicates(subset=['author'])
        
        stats = {
            'total_developers': len(unique_devs),
            'avg_bug_rate': float(unique_devs['bug_rate'].mean()),
            'max_bug_rate': float(unique_devs['bug_rate'].max()),
            'min_bug_rate': float(unique_devs['bug_rate'].min()),
            'avg_commits_per_dev': float(unique_devs['total_commits'].mean()),
            'most_active_developer': {
                'author': unique_devs.loc[unique_devs['total_commits'].idxmax(), 'author'],
                'commits': int(unique_devs['total_commits'].max())
            }
        }
        
        return stats


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("TESTING DEVELOPER FEATURE EXTRACTOR")
    logger.info("=" * 70)
    
    # Create sample commits
    sample_commits = pd.DataFrame({
        'commit_hash': ['abc123', 'def456', 'ghi789', 'jkl012', 'mno345'],
        'author': ['alice@example.com', 'bob@example.com', 'alice@example.com', 
                   'alice@example.com', 'bob@example.com'],
        'timestamp': pd.to_datetime([
            '2024-01-01', '2024-01-02', '2024-01-03', 
            '2024-01-04', '2024-01-05'
        ]),
        'lines_added': [100, 50, 200, 150, 80],
        'lines_deleted': [30, 20, 10, 40, 25],
        'files_changed': [5, 2, 8, 6, 3]
    })
    
    # Create sample labels
    sample_labels = pd.DataFrame({
        'commit_hash': ['abc123', 'def456', 'ghi789', 'jkl012', 'mno345'],
        'is_buggy': [1, 0, 1, 0, 0]
    })
    
    print("\nInput commits:")
    print(sample_commits[['commit_hash', 'author', 'lines_added']])
    
    print("\nInput labels:")
    print(sample_labels)
    
    # Extract features
    extractor = DeveloperFeatureExtractor(recent_window_days=30)
    features = extractor.extract_features(sample_commits, sample_labels)
    
    print("\nExtracted developer features:")
    print(features)
    
    # Get statistics
    stats = extractor.get_feature_statistics(features)
    print("\nDeveloper statistics:")
    print(f"Total developers: {stats['total_developers']}")
    print(f"Average bug rate: {stats['avg_bug_rate']:.2%}")
    print(f"Most active: {stats['most_active_developer']['author']} "
          f"({stats['most_active_developer']['commits']} commits)")
    
    print("\n✅ Developer feature extraction test complete!")