"""
Run Feature Engineering Pipeline
Engineers features from raw commits and labels
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd
from src.features.feature_engineer import FeatureEngineer
from src.utils.config_loader import ConfigLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/feature_engineering.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main feature engineering pipeline
    """
    logger.info("=" * 70)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 70)
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_main_config()
        
        # Get paths
        commits_path = config['data_ingestion']['raw_data_path']
        labels_path = "data/processed/labels.csv"
        output_path = "data/features/commit_features.csv"
        
        logger.info(f"Commits input: {commits_path}")
        logger.info(f"Labels input: {labels_path}")
        logger.info(f"Features output: {output_path}")
        
        # Load data
        logger.info("Loading data...")
        commits_df = pd.read_csv(commits_path)
        labels_df = pd.read_csv(labels_path)
        
        logger.info(f"Loaded {len(commits_df)} commits")
        logger.info(f"Loaded {len(labels_df)} labels")
        
        # Initialize feature engineer
        feature_config = {
            'core_modules': config['features']['core_modules']
        }
        engineer = FeatureEngineer(config=feature_config)
        
        # Engineer features
        logger.info("Engineering features...")
        features_df = engineer.engineer_features(commits_df, labels_df)
        
        # Validate features
        logger.info("Validating features...")
        is_valid = engineer.validate_features(features_df)
        
        if not is_valid:
            logger.warning("Feature validation found issues (but continuing)")
        
        # Save features
        logger.info("Saving features...")
        engineer.save_features(features_df, output_path)
        
        # Get summary
        summary = engineer.get_feature_summary(features_df)
        
        # Display statistics
        logger.info("=" * 70)
        logger.info("FEATURE ENGINEERING STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total records: {summary['total_records']}")
        logger.info(f"Total features: {summary['total_features']}")
        logger.info(f"Numeric features: {summary['feature_types']['numeric']}")
        logger.info(f"Categorical features: {summary['feature_types']['categorical']}")
        
        if 'label_distribution' in summary:
            logger.info(f"\nLabel distribution:")
            logger.info(f"  Buggy commits: {summary['label_distribution']['buggy']}")
            logger.info(f"  Clean commits: {summary['label_distribution']['clean']}")
            logger.info(f"  Bug ratio: {summary['label_distribution']['bug_ratio']:.2%}")
        
        logger.info(f"\nKey feature statistics:")
        for feature, stats in list(summary['numeric_stats'].items())[:3]:
            logger.info(f"  {feature}:")
            logger.info(f"    Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}")
            logger.info(f"    Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
        
        # Display sample features
        logger.info("\nSample features (first 3 commits):")
        sample_cols = ['commit_hash', 'lines_added', 'total_churn', 'files_changed', 
                      'bug_rate', 'hour_of_day', 'is_buggy']
        available_cols = [col for col in sample_cols if col in features_df.columns]
        logger.info(f"\n{features_df[available_cols].head(3).to_string()}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ FEATURE ENGINEERING COMPLETE")
        logger.info("=" * 70)
        
    except FileNotFoundError as e:
        logger.error(f"❌ Input file not found: {e}")
        logger.error("Please run previous pipelines first:")
        logger.error("  1. python scripts/run_ingestion.py")
        logger.error("  2. python scripts/run_labeling.py")
        raise
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()