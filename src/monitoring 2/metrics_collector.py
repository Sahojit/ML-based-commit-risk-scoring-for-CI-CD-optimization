"""
Metrics Collector
Collects and stores prediction metrics for monitoring
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import logging
from datetime import datetime
from typing import Dict, Any
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and stores prediction metrics
    """
    
    def __init__(self, log_file: str = "logs/predictions.log"):
        """
        Initialize MetricsCollector
        
        Args:
            log_file: Path to predictions log file
        """
        self.log_file = log_file
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"MetricsCollector initialized with log file: {log_file}")
    
    def log_prediction(
        self,
        commit_hash: str,
        risk_score: float,
        risk_level: str,
        features: Dict[str, Any],
        response_time_ms: float
    ):
        """
        Log a single prediction
        
        Args:
            commit_hash: Commit identifier
            risk_score: Predicted risk score
            risk_level: Risk level (LOW/MEDIUM/HIGH)
            features: Input features used
            response_time_ms: API response time
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "commit_hash": commit_hash,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "features": features,
            "response_time_ms": response_time_ms
        }
        
        # Append to log file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def load_predictions(self, limit: int = None) -> pd.DataFrame:
        """
        Load prediction logs as DataFrame
        
        Args:
            limit: Maximum number of recent predictions to load
        
        Returns:
            DataFrame with predictions
        """
        try:
            predictions = []
            
            with open(self.log_file, 'r') as f:
                for line in f:
                    if line.strip():
                        predictions.append(json.loads(line))
            
            if not predictions:
                return pd.DataFrame()
            
            # Apply limit
            if limit:
                predictions = predictions[-limit:]
            
            df = pd.DataFrame(predictions)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
        
        except FileNotFoundError:
            logger.warning(f"Log file not found: {self.log_file}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading predictions: {e}")
            return pd.DataFrame()
    
    def get_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate summary statistics
        
        Args:
            df: DataFrame with predictions
        
        Returns:
            Dictionary with statistics
        """
        if df.empty:
            return {
                "total_predictions": 0,
                "avg_risk_score": 0,
                "high_risk_count": 0,
                "medium_risk_count": 0,
                "low_risk_count": 0,
                "avg_response_time_ms": 0
            }
        
        stats = {
            "total_predictions": len(df),
            "avg_risk_score": float(df['risk_score'].mean()),
            "high_risk_count": int((df['risk_level'] == 'HIGH').sum()),
            "medium_risk_count": int((df['risk_level'] == 'MEDIUM').sum()),
            "low_risk_count": int((df['risk_level'] == 'LOW').sum()),
            "avg_response_time_ms": float(df['response_time_ms'].mean()) if 'response_time_ms' in df.columns else 0
        }
        
        return stats


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("TESTING METRICS COLLECTOR")
    logger.info("=" * 70)
    
    # Initialize collector
    collector = MetricsCollector()
    
    # Log some sample predictions
    print("\nLogging sample predictions...")
    
    sample_predictions = [
        {
            "commit_hash": "abc123",
            "risk_score": 0.75,
            "risk_level": "HIGH",
            "features": {"lines_added": 150, "bug_rate": 0.25},
            "response_time_ms": 45.2
        },
        {
            "commit_hash": "def456",
            "risk_score": 0.35,
            "risk_level": "LOW",
            "features": {"lines_added": 20, "bug_rate": 0.05},
            "response_time_ms": 38.7
        },
        {
            "commit_hash": "ghi789",
            "risk_score": 0.58,
            "risk_level": "MEDIUM",
            "features": {"lines_added": 80, "bug_rate": 0.15},
            "response_time_ms": 42.1
        }
    ]
    
    for pred in sample_predictions:
        collector.log_prediction(**pred)
    
    print(f"✅ Logged {len(sample_predictions)} predictions")
    
    # Load and display
    print("\nLoading predictions...")
    df = collector.load_predictions()
    
    print(f"\nLoaded {len(df)} predictions:")
    print(df[['timestamp', 'commit_hash', 'risk_score', 'risk_level']].head())
    
    # Get statistics
    print("\nSummary Statistics:")
    stats = collector.get_summary_stats(df)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Metrics collector test complete!")