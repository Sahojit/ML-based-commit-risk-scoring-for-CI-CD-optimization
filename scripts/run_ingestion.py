"""
Run Data Ingestion Pipeline
Extracts commits from GitHub and saves to CSV (and optionally PostgreSQL)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from src.ingestion.git_extractor import GitExtractor
from src.ingestion.db_loader import DatabaseLoader
from src.utils.config_loader import ConfigLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main ingestion pipeline
    """
    logger.info("=" * 70)
    logger.info("STARTING DATA INGESTION PIPELINE")
    logger.info("=" * 70)
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_main_config()
        
        # Get ingestion settings
        ingestion_config = config['data_ingestion']
        repo_owner = ingestion_config['repo_owner']
        repo_name = ingestion_config['repo_name']
        max_commits = ingestion_config['max_commits']
        output_path = ingestion_config['raw_data_path']
        
        logger.info(f"Repository: {repo_owner}/{repo_name}")
        logger.info(f"Max commits: {max_commits}")
        logger.info(f"Output path: {output_path}")
        
        # Initialize extractor
        extractor = GitExtractor()
        
        # Extract commits
        logger.info("Extracting commits from GitHub...")
        df = extractor.extract_commits(
            repo_owner=repo_owner,
            repo_name=repo_name,
            max_commits=max_commits
        )
        
        # Save to CSV
        logger.info("Saving to CSV...")
        extractor.save_to_csv(df, output_path)
        
        # Print statistics
        stats = extractor.get_statistics(df)
        logger.info("=" * 70)
        logger.info("EXTRACTION STATISTICS")
        logger.info("=" * 70)
        logger.info(f"Total commits: {stats['total_commits']}")
        logger.info(f"Unique authors: {stats['unique_authors']}")
        logger.info(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        logger.info(f"Avg files changed: {stats['avg_files_changed']:.2f}")
        logger.info(f"Avg lines added: {stats['avg_lines_added']:.2f}")
        logger.info(f"Avg lines deleted: {stats['avg_lines_deleted']:.2f}")
        
        # Optional: Load to database (if PostgreSQL is set up)
        try:
            db_url = config_loader.get_database_url()
            logger.info("Attempting to load data into PostgreSQL...")
            
            db_loader = DatabaseLoader(db_url)
            db_loader.connect()
            db_loader.create_table_if_not_exists()
            db_loader.load_dataframe(df, if_exists='append')
            
            count = db_loader.get_record_count()
            logger.info(f"Total records in database: {count}")
            
            db_loader.close()
            
        except Exception as e:
            logger.warning(f"Database loading skipped: {e}")
            logger.info("This is normal if PostgreSQL is not set up yet (Phase 5)")
        
        logger.info("=" * 70)
        logger.info("✅ DATA INGESTION COMPLETE")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()