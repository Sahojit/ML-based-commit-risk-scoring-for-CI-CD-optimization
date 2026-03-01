"""
Commit-Level Feature Engineering
Extracts features from individual commits
"""

import logging
import pandas as pd
import numpy as np
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CommitFeatureExtractor:
    """
    Extracts commit-level features
    """
    
    def __init__(self, core_modules: List[str] = None):
        """
        Initialize CommitFeatureExtractor
        
        Args:
            core_modules: List of core module path patterns
        """
        self.core_modules = core_modules or [
            'src/core/',
            'src/api/',
            'src/models/',
            'lib/',
            'core/'
        ]
        logger.info(f"CommitFeatureExtractor initialized")
    
    def extract_features(self, commits_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all commit-level features
        
        Args:
            commits_df: DataFrame with commit data
        
        Returns:
            DataFrame with commit features
        """
        logger.info(f"Extracting commit features for {len(commits_df)} commits")
        
        # Start with basic features (already in data)
        features_df = commits_df[[
            'commit_hash',
            'lines_added',
            'lines_deleted',
            'files_changed'
        ]].copy()
        
        # Calculate total churn
        features_df['total_churn'] = (
            features_df['lines_added'] + features_df['lines_deleted']
        )
        
        # Calculate churn ratio (deletions / additions)
        features_df['churn_ratio'] = features_df.apply(
            lambda row: row['lines_deleted'] / row['lines_added'] 
            if row['lines_added'] > 0 else 0,
            axis=1
        )
        
        # Note: touches_core and touches_tests require file path data
        # For now, we'll set them to 0 (can be enhanced with Git API)
        features_df['touches_core'] = 0
        features_df['touches_tests'] = 0
        
        # Add commit complexity score
        features_df['complexity_score'] = self._calculate_complexity(features_df)
        
        logger.info("Commit features extracted successfully")
        
        return features_df
    
    def _calculate_complexity(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate a complexity score based on multiple factors
        
        Args:
            df: DataFrame with commit metrics
        
        Returns:
            Series with complexity scores
        """
        # Normalize features to 0-1 range
        max_churn = df['total_churn'].max() if df['total_churn'].max() > 0 else 1
        max_files = df['files_changed'].max() if df['files_changed'].max() > 0 else 1
        
        churn_norm = df['total_churn'] / max_churn
        files_norm = df['files_changed'] / max_files
        
        # Weighted average (you can tune these weights)
        complexity = (0.6 * churn_norm) + (0.4 * files_norm)
        
        return complexity
    
    def get_feature_statistics(self, features_df: pd.DataFrame) -> dict:
        """
        Get statistics about extracted features
        
        Args:
            features_df: DataFrame with features
        
        Returns:
            Dictionary with statistics
        """
        numeric_cols = [
            'lines_added', 'lines_deleted', 'total_churn',
            'files_changed', 'complexity_score'
        ]
        
        stats = {}
        for col in numeric_cols:
            if col in features_df.columns:
                stats[col] = {
                    'mean': float(features_df[col].mean()),
                    'median': float(features_df[col].median()),
                    'std': float(features_df[col].std()),
                    'min': float(features_df[col].min()),
                    'max': float(features_df[col].max())
                }
        
        return stats


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("TESTING COMMIT FEATURE EXTRACTOR")
    logger.info("=" * 70)
    
    # Create sample data
    sample_commits = pd.DataFrame({
        'commit_hash': ['abc123', 'def456', 'ghi789'],
        'lines_added': [120, 50, 200],
        'lines_deleted': [45, 30, 10],
        'files_changed': [5, 2, 8]
    })
    
    print("\nInput commits:")
    print(sample_commits)
    
    # Extract features
    extractor = CommitFeatureExtractor()
    features = extractor.extract_features(sample_commits)
    
    print("\nExtracted features:")
    print(features)
    
    # Get statistics
    stats = extractor.get_feature_statistics(features)
    print("\nFeature statistics:")
    print(f"Total churn - mean: {stats['total_churn']['mean']:.2f}, "
          f"max: {stats['total_churn']['max']:.0f}")
    print(f"Files changed - mean: {stats['files_changed']['mean']:.2f}, "
          f"max: {stats['files_changed']['max']:.0f}")
    
    print("\n✅ Commit feature extraction test complete!")