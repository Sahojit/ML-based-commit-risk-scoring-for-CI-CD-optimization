
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any
from src.inference.model_loader import ModelLoader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CommitPredictor:
    
    def __init__(self, model_path: str = "models/advanced_xgboost.pkl"):
        self.model_loader = ModelLoader(model_path)
        self.model_loader.load_model()
        
        logger.info("CommitPredictor initialized and model loaded")
    
    def predict_commit(self, commit_data: Dict[str, Any]) -> Dict[str, Any]:
        features = self._extract_features(commit_data)
        
        risk_score = self.model_loader.predict_proba(features)
        risk_label = self.model_loader.predict(features)
        
        risk_level = self._get_risk_level(risk_score)
        
        recommendation = self._get_recommendation(risk_level)
        
        result = {
            "commit_hash": commit_data.get("commit_hash", "unknown"),
            "risk_score": float(risk_score),
            "risk_label": int(risk_label),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "prediction_time": datetime.now().isoformat()
        }
        
        logger.info(f"Prediction: {commit_data.get('commit_hash')} -> Risk: {risk_score:.4f} ({risk_level})")
        
        return result
    
    def _extract_features(self, commit_data: Dict[str, Any]) -> pd.DataFrame:
        lines_added = commit_data.get('lines_added', 0)
        lines_deleted = commit_data.get('lines_deleted', 0)
        files_changed = commit_data.get('files_changed', 0)
        total_churn = lines_added + lines_deleted
        churn_ratio = lines_deleted / lines_added if lines_added > 0 else 0
        
        touches_core = commit_data.get('touches_core', 0)
        touches_tests = commit_data.get('touches_tests', 0)
        
        complexity_score = commit_data.get('complexity_score', 
                                          min(total_churn / 500, 1.0))
        
        total_commits = commit_data.get('total_commits', 1)
        buggy_commits = commit_data.get('buggy_commits', 0)
        bug_rate = buggy_commits / total_commits if total_commits > 0 else 0
        recent_frequency = commit_data.get('recent_frequency', 0)
        avg_lines_added = commit_data.get('avg_lines_added', lines_added)
        avg_lines_deleted = commit_data.get('avg_lines_deleted', lines_deleted)
        avg_files_changed = commit_data.get('avg_files_changed', files_changed)
        
        timestamp = commit_data.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        hour_of_day = timestamp.hour if hasattr(timestamp, 'hour') else 12
        day_of_week = timestamp.weekday() if hasattr(timestamp, 'weekday') else 2
        is_weekend = 1 if day_of_week >= 5 else 0
        month = timestamp.month if hasattr(timestamp, 'month') else 1
        
        features = pd.DataFrame({
            'lines_added': [lines_added],
            'lines_deleted': [lines_deleted],
            'files_changed': [files_changed],
            'total_churn': [total_churn],
            'churn_ratio': [churn_ratio],
            'touches_core': [touches_core],
            'touches_tests': [touches_tests],
            'complexity_score': [complexity_score],
            'total_commits': [total_commits],
            'buggy_commits': [buggy_commits],
            'bug_rate': [bug_rate],
            'recent_frequency': [recent_frequency],
            'avg_lines_added': [avg_lines_added],
            'avg_lines_deleted': [avg_lines_deleted],
            'avg_files_changed': [avg_files_changed],
            'hour_of_day': [hour_of_day],
            'day_of_week': [day_of_week],
            'is_weekend': [is_weekend],
            'month': [month]
        })
        
        return features
    
    def _get_risk_level(self, risk_score: float) -> str:
        if risk_score >= 0.7:
            return "HIGH"
        elif risk_score >= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _get_recommendation(self, risk_level: str) -> str:
        recommendations = {
            "HIGH": "Run full test suite (45 min) - High bug risk detected",
            "MEDIUM": "Run extended tests (15 min) - Moderate bug risk",
            "LOW": "Run smoke tests (5 min) - Low bug risk"
        }
        
        return recommendations.get(risk_level, "Run standard tests")
    
    def predict_batch(self, commits: list) -> list:
        results = []
        
        for commit_data in commits:
            try:
                result = self.predict_commit(commit_data)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting commit {commit_data.get('commit_hash')}: {e}")
                results.append({
                    "commit_hash": commit_data.get("commit_hash", "unknown"),
                    "error": str(e)
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        return self.model_loader.get_model_info()

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("TESTING COMMIT PREDICTOR")
    logger.info("=" * 70)
    
    try:
        predictor = CommitPredictor()
        
        print("\n--- Single Commit Prediction ---")
        commit = {
            "commit_hash": "abc123",
            "lines_added": 150,
            "lines_deleted": 50,
            "files_changed": 8,
            "touches_core": 1,
            "touches_tests": 0,
            "total_commits": 100,
            "buggy_commits": 25,
            "recent_frequency": 10,
            "timestamp": "2024-03-01 14:30:00"
        }
        
        result = predictor.predict_commit(commit)
        
        print(f"\nCommit: {result['commit_hash']}")
        print(f"Risk Score: {result['risk_score']:.4f} ({result['risk_score']*100:.2f}%)")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendation: {result['recommendation']}")
        
        print("\n--- Batch Prediction (3 commits) ---")
        commits = [
            {
                "commit_hash": "def456",
                "lines_added": 20,
                "lines_deleted": 5,
                "files_changed": 2,
                "touches_core": 0,
                "total_commits": 200,
                "buggy_commits": 10,
                "timestamp": "2024-03-01 09:00:00"
            },
            {
                "commit_hash": "ghi789",
                "lines_added": 500,
                "lines_deleted": 200,
                "files_changed": 15,
                "touches_core": 1,
                "total_commits": 50,
                "buggy_commits": 30,
                "timestamp": "2024-03-01 23:45:00"
            },
            {
                "commit_hash": "jkl012",
                "lines_added": 5,
                "lines_deleted": 2,
                "files_changed": 1,
                "touches_core": 0,
                "total_commits": 300,
                "buggy_commits": 5,
                "timestamp": "2024-03-02 10:15:00"
            }
        ]
        
        batch_results = predictor.predict_batch(commits)
        
        print("\nBatch Results:")
        for r in batch_results:
            if 'error' not in r:
                print(f"  {r['commit_hash']}: {r['risk_level']} ({r['risk_score']:.2f})")
        
        print("\n--- Model Information ---")
        info = predictor.get_model_info()
        print(f"Model Type: {info['model_type']}")
        print(f"Features: {info['num_features']}")
        
        print("\n✅ Commit predictor test complete!")
        
    except FileNotFoundError:
        print("\n❌ Model file not found!")
        print("Please train a model first: python scripts/run_training.py")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()