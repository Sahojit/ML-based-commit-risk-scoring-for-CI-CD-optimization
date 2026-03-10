
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd
from src.training.label_generator import LabelGenerator
from src.utils.config_loader import ConfigLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/labeling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 70)
    logger.info("STARTING LABEL GENERATION PIPELINE")
    logger.info("=" * 70)
    
    try:
        config_loader = ConfigLoader()
        config = config_loader.load_main_config()
        
        ingestion_config = config['data_ingestion']
        label_config = config['label_generation']
        
        input_path = ingestion_config['raw_data_path']
        output_path = "data/processed/labels.csv"
        bug_keywords = label_config['bug_keywords']
        
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Bug keywords: {', '.join(bug_keywords)}")
        
        logger.info("Loading commits data...")
        commits_df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(commits_df)} commits")
        
        generator = LabelGenerator(bug_keywords=bug_keywords)
        
        logger.info("Generating labels...")
        labels_df = generator.generate_labels(commits_df)
        
        logger.info("Validating labels...")
        validation = generator.validate_labels(labels_df)
        
        logger.info("Saving labels...")
        generator.save_labels(labels_df, output_path)
        
        logger.info("=" * 70)
        logger.info("LABELING STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total commits: {validation['total_commits']}")
        logger.info(f"Buggy commits: {validation['buggy_commits']} ({validation['bug_ratio']:.1%})")
        logger.info(f"Clean commits: {validation['clean_commits']}")
        logger.info(f"Average confidence: {validation['avg_confidence']:.2f}")
        
        keyword_stats = generator.get_bug_keywords_used(commits_df)
        logger.info("\nKeyword usage:")
        for _, row in keyword_stats.head(5).iterrows():
            if row['count'] > 0:
                logger.info(f"  '{row['keyword']}': {row['count']} occurrences")
        
        review_df = commits_df.merge(labels_df, on='commit_hash', how='left')
        buggy_commits = review_df[review_df['is_buggy'] == 1][
            ['commit_hash', 'author_name', 'message', 'is_buggy']
        ]
        
        if len(buggy_commits) > 0:
            logger.info("\nSample buggy commits:")
            for idx, row in buggy_commits.head(3).iterrows():
                logger.info(f"  - {row['message'][:60]}...")
        
        logger.info("=" * 70)
        logger.info("✅ LABEL GENERATION COMPLETE")
        logger.info("=" * 70)
        
    except FileNotFoundError as e:
        logger.error(f"❌ Input file not found: {e}")
        logger.error("Please run data ingestion first: python scripts/run_ingestion.py")
        raise
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()