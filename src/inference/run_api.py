"""
Run FastAPI Inference Server
Starts the commit risk prediction API
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Start the FastAPI server
    """
    logger.info("=" * 70)
    logger.info("STARTING ML COMMIT RISK API SERVER")
    logger.info("=" * 70)
    logger.info("API Endpoints:")
    logger.info("  - Health Check:     http://localhost:8000/")
    logger.info("  - Predict Single:   POST http://localhost:8000/predict")
    logger.info("  - Predict Batch:    POST http://localhost:8000/predict/batch")
    logger.info("  - Model Info:       GET  http://localhost:8000/model/info")
    logger.info("  - API Docs:         http://localhost:8000/docs")
    logger.info("  - Alternative Docs: http://localhost:8000/redoc")
    logger.info("=" * 70)
    
    uvicorn.run(
        "src.inference.api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()