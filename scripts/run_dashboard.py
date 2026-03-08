"""
Run Monitoring Dashboard
Starts the Streamlit monitoring dashboard
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import subprocess
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Start the Streamlit dashboard
    """
    logger.info("=" * 70)
    logger.info("STARTING MONITORING DASHBOARD")
    logger.info("=" * 70)
    logger.info("Dashboard will open at: http://localhost:8501")
    logger.info("=" * 70)
    
    # Run Streamlit
    subprocess.run([
        "streamlit", "run",
        "src/monitoring/dashboard.py",
        "--server.port", "8501",
        "--server.headless", "true"
    ])


if __name__ == "__main__":
    main()