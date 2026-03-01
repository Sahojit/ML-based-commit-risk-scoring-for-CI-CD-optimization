"""
Feature Engineering Pipeline
Combines all feature extractors into unified pipeline
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import pandas as pd
import numpy as np
from src.features.commit_features import CommitFeatureExtractor
from src.features.developer_features import DeveloperFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Main feature engineering pipeline
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize FeatureEngineer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize feature extractors
        core_modules = self.config.get('core_modules', [])
        self.commit_extractor = CommitFeatureExtractor(core_modules=core_modules)
        self.developer_extractor = DeveloperFeatureExtractor(recent_window_days=30)
        
        logger.info("FeatureEngineer initialized")
    
    def engineer_features(
        self,
        commits_df: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Engineer all features
        
        Args:
            commits_df: DataFrame with commit data
            labels_df: DataFrame with labels
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info(f"Starting feature engineering for {len(commits_df)} commits")
        
        # Step 1: Extract commit-level features
        logger.info("Extracting commit features...")
        commit_features = self.commit_extractor.extract_features(commits_df)
        
        # Step 2: Extract developer-level features
        logger.info("Extracting developer features...")
        developer_features = self.developer_extractor.extract_features(
            commits_df, labels_df
        )
        
        # Step 3: Extract temporal features
        logger.info("Extracting temporal features...")
        temporal_features = self._extract_temporal_features(commits_df)
        
        # Step 4: Merge all features
        logger.info("Merging all features...")
        features = commit_features.merge(
            developer_features,
            on='commit_hash',
            how='left'
        ).merge(
            temporal_features,
            on='commit_hash',
            how='left'
        )
        
        # Step 5: Add labels
        features = features.merge(
            labels_df[['commit_hash', 'is_buggy']],
            on='commit_hash',
            how='left'
        )
        
        logger.info(f"Feature engineering complete. Total features: {len(features.columns)}")
        
        return features
    
    def _extract_temporal_features(self, commits_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from commit timestamps
        
        Args:
            commits_df: DataFrame with commit data
        
        Returns:
            DataFrame with temporal features
        """
        temporal = commits_df[['commit_hash', 'timestamp']].copy()
        
        # Convert to datetime
        temporal['timestamp'] = pd.to_datetime(temporal['timestamp'])
        
        # Extract temporal features
        temporal['hour_of_day'] = temporal['timestamp'].dt.hour
        temporal['day_of_week'] = temporal['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
        temporal['is_weekend'] = (temporal['day_of_week'] >= 5).astype(int)
        temporal['month'] = temporal['timestamp'].dt.month
        
        # Drop timestamp column
        temporal = temporal.drop('timestamp', axis=1)
        
        return temporal
    
    def save_features(self, features_df: pd.DataFrame, output_path: str):
        """
        Save features to CSV
        
        Args:
            features_df: DataFrame with features
            output_path: Path to output file
        """
        # Create directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        features_df.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")
        logger.info(f"Total records: {len(features_df)}")
        logger.info(f"Total features: {len(features_df.columns)}")
    
    def get_feature_summary(self, features_df: pd.DataFrame) -> dict:
        """
        Get summary statistics of all features
        
        Args:
            features_df: DataFrame with features
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_records': len(features_df),
            'total_features': len(features_df.columns),
            'feature_types': {},
            'missing_values': {},
            'label_distribution': {}
        }
        
        # Count feature types
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        categorical_cols = features_df.select_dtypes(include=['object']).columns
        
        summary['feature_types'] = {
            'numeric': len(numeric_cols),
            'categorical': len(categorical_cols)
        }
        
        # Check for missing values
        missing = features_df.isnull().sum()
        if missing.sum() > 0:
            summary['missing_values'] = missing[missing > 0].to_dict()
        else:
            summary['missing_values'] = 'None'
        
        # Label distribution
        if 'is_buggy' in features_df.columns:
            summary['label_distribution'] = {
                'buggy': int(features_df['is_buggy'].sum()),
                'clean': int((features_df['is_buggy'] == 0).sum()),
                'bug_ratio': float(features_df['is_buggy'].mean())
            }
        
        # Numeric feature statistics
        summary['numeric_stats'] = {}
        for col in ['lines_added', 'lines_deleted', 'total_churn', 'files_changed']:
            if col in features_df.columns:
                summary['numeric_stats'][col] = {
                    'mean': float(features_df[col].mean()),
                    'median': float(features_df[col].median()),
                    'std': float(features_df[col].std()),
                    'min': float(features_df[col].min()),
                    'max': float(features_df[col].max())
                }
        
        return summary
    
    def validate_features(self, features_df: pd.DataFrame) -> bool:
        """
        Validate feature quality
        
        Args:
            features_df: DataFrame with features
        
        Returns:
            True if validation passes
        """
        issues = []
        
        # Check for missing values
        missing = features_df.isnull().sum().sum()
        if missing > 0:
            issues.append(f"Found {missing} missing values")
        
        # Check for infinite values
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(features_df[col]).any():
                issues.append(f"Column '{col}' contains infinite values")
        
        # Check for duplicate commit_hashes
        if features_df['commit_hash'].duplicated().any():
            issues.append("Found duplicate commit_hashes")
        
        # Check label exists
        if 'is_buggy' not in features_df.columns:
            issues.append("Label column 'is_buggy' not found")
        
        if issues:
            for issue in issues:
                logger.warning(f"Validation issue: {issue}")
            return False
        else:
            logger.info("✅ Feature validation passed")
            return True


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("TESTING FEATURE ENGINEER")
    logger.info("=" * 70)
    
    # Create sample data
    sample_commits = pd.DataFrame({
        'commit_hash': ['abc123', 'def456', 'ghi789'],
        'author': ['alice@example.com', 'bob@example.com', 'alice@example.com'],
        'timestamp': pd.to_datetime(['2024-01-01 14:30', '2024-01-02 09:15', '2024-01-03 22:45']),
        'lines_added': [120, 50, 200],
        'lines_deleted': [45, 30, 10],
        'files_changed': [5, 2, 8]
    })
    
    sample_labels = pd.DataFrame({
        'commit_hash': ['abc123', 'def456', 'ghi789'],
        'is_buggy': [1, 0, 1]
    })
    
    print("\nInput commits:")
    print(sample_commits)
    
    print("\nInput labels:")
    print(sample_labels)
    
    # Initialize engineer
    engineer = FeatureEngineer()
    
    # Engineer features
    features = engineer.engineer_features(sample_commits, sample_labels)
    
    print("\nEngineered features:")
    print(features)
    
    print(f"\nFeature columns: {list(features.columns)}")
    
    # Get summary
    summary = engineer.get_feature_summary(features)
    print("\nFeature summary:")
    print(f"Total records: {summary['total_records']}")
    print(f"Total features: {summary['total_features']}")
    print(f"Numeric features: {summary['feature_types']['numeric']}")
    print(f"Bug ratio: {summary['label_distribution']['bug_ratio']:.2%}")
    
    # Validate
    is_valid = engineer.validate_features(features)
    print(f"\nValidation result: {'✅ PASS' if is_valid else '❌ FAIL'}")
    
    print("\n✅ Feature engineering test complete!")