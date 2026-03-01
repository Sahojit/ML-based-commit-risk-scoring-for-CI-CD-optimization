"""
Database Loader
Loads extracted commit data into PostgreSQL
"""

import logging
import pandas as pd
from sqlalchemy import create_engine, text
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseLoader:
    """
    Loads data into PostgreSQL database
    """
    
    def __init__(self, database_url: str):
        """
        Initialize DatabaseLoader
        
        Args:
            database_url: SQLAlchemy database connection string
        """
        self.database_url = database_url
        self.engine = None
        logger.info("DatabaseLoader initialized")
    
    def connect(self):
        """
        Connect to the database
        """
        try:
            self.engine = create_engine(self.database_url)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def create_table_if_not_exists(self, table_name: str = "raw_commits"):
        """
        Create raw_commits table if it doesn't exist
        
        Args:
            table_name: Name of the table to create
        """
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            commit_hash VARCHAR(40) UNIQUE NOT NULL,
            author VARCHAR(255),
            author_name VARCHAR(255),
            timestamp TIMESTAMP,
            message TEXT,
            files_changed INTEGER,
            lines_added INTEGER,
            lines_deleted INTEGER,
            total_changes INTEGER,
            extracted_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text(create_table_sql))
                conn.commit()
            logger.info(f"Table '{table_name}' ensured to exist")
        except Exception as e:
            logger.error(f"Failed to create table: {e}")
            raise
    
    def load_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str = "raw_commits",
        if_exists: str = "append"
    ):
        """
        Load DataFrame into database
        
        Args:
            df: DataFrame to load
            table_name: Target table name
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
        """
        if self.engine is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        try:
            # Load data
            rows_loaded = df.to_sql(
                table_name,
                self.engine,
                if_exists=if_exists,
                index=False
            )
            
            logger.info(f"Successfully loaded {len(df)} records into '{table_name}'")
            return rows_loaded
        
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
    
    def get_record_count(self, table_name: str = "raw_commits") -> int:
        """
        Get total number of records in table
        
        Args:
            table_name: Table name
        
        Returns:
            Number of records
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                count = result.scalar()
            return count
        except Exception as e:
            logger.error(f"Failed to get record count: {e}")
            return 0
    
    def close(self):
        """
        Close database connection
        """
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.utils.config_loader import ConfigLoader
    
    logger.info("=" * 70)
    logger.info("TESTING DATABASE LOADER")
    logger.info("=" * 70)
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        
        # Get database URL
        try:
            db_url = config_loader.get_database_url()
            logger.info(f"Database URL loaded (host hidden for security)")
        except Exception as e:
            logger.warning(f"Could not load database config: {e}")
            logger.warning("Skipping database test (PostgreSQL not set up yet)")
            logger.info("This is normal if you haven't set up PostgreSQL yet")
            logger.info("We'll set up the database in Phase 5")
            sys.exit(0)
        
        # Initialize loader
        loader = DatabaseLoader(db_url)
        
        # Connect
        loader.connect()
        
        # Create table
        loader.create_table_if_not_exists()
        
        # Create sample data
        sample_data = pd.DataFrame({
            'commit_hash': ['abc123', 'def456'],
            'author': ['john@example.com', 'jane@example.com'],
            'author_name': ['John Doe', 'Jane Smith'],
            'timestamp': pd.to_datetime(['2024-01-01', '2024-01-02']),
            'message': ['Test commit 1', 'Test commit 2'],
            'files_changed': [3, 5],
            'lines_added': [100, 200],
            'lines_deleted': [50, 75],
            'total_changes': [150, 275],
            'extracted_at': pd.Timestamp.now()
        })
        
        # Load data
        loader.load_dataframe(sample_data, if_exists='append')
        
        # Get count
        count = loader.get_record_count()
        logger.info(f"Total records in database: {count}")
        
        # Close
        loader.close()
        
        print("\n✅ Database loader test successful!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        logger.info("This is expected if PostgreSQL is not set up yet")