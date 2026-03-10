
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import joblib
from typing import Optional
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelLoader:
    
    def __init__(self, model_path: str = "models/advanced_xgboost.pkl"):
        self.model_path = model_path
        self.model = None
        self.feature_names = None
        
        logger.info(f"ModelLoader initialized with path: {model_path}")
    
    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
            else:
                self.feature_names = self._get_default_feature_names()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Model type: {type(self.model).__name__}")
            logger.info(f"Features expected: {len(self.feature_names)}")
            
            return self.model
        
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _get_default_feature_names(self) -> list:
        return [
            'lines_added', 'lines_deleted', 'files_changed', 'total_churn',
            'churn_ratio', 'touches_core', 'touches_tests', 'complexity_score',
            'total_commits', 'buggy_commits', 'bug_rate', 'recent_frequency',
            'avg_lines_added', 'avg_lines_deleted', 'avg_files_changed',
            'hour_of_day', 'day_of_week', 'is_weekend', 'month'
        ]
    
    def predict_proba(self, features: pd.DataFrame) -> float:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if self.feature_names:
            missing_features = set(self.feature_names) - set(features.columns)
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                for feature in missing_features:
                    features[feature] = 0
            
            features = features[self.feature_names]
        
        proba = self.model.predict_proba(features)[:, 1]
        
        return proba[0]
    
    def predict(self, features: pd.DataFrame) -> int:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if self.feature_names:
            features = features[self.feature_names]
        
        prediction = self.model.predict(features)
        
        return int(prediction[0])
    
    def get_model_info(self) -> dict:
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_type": type(self.model).__name__,
            "model_path": self.model_path,
            "num_features": len(self.feature_names) if self.feature_names else None,
            "feature_names": self.feature_names
        }

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("TESTING MODEL LOADER")
    logger.info("=" * 70)
    
    loader = ModelLoader()
    
    try:
        loader.load_model()
        
        info = loader.get_model_info()
        print("\nModel Info:")
        print(f"  Status: {info['status']}")
        print(f"  Type: {info['model_type']}")
        print(f"  Features: {info['num_features']}")
        
        sample_features = pd.DataFrame({
            'lines_added': [120],
            'lines_deleted': [45],
            'files_changed': [5],
            'total_churn': [165],
            'churn_ratio': [0.375],
            'touches_core': [1],
            'touches_tests': [0],
            'complexity_score': [0.5],
            'total_commits': [50],
            'buggy_commits': [10],
            'bug_rate': [0.2],
            'recent_frequency': [5],
            'avg_lines_added': [100],
            'avg_lines_deleted': [40],
            'avg_files_changed': [4],
            'hour_of_day': [14],
            'day_of_week': [2],
            'is_weekend': [0],
            'month': [3]
        })
        
        proba = loader.predict_proba(sample_features)
        prediction = loader.predict(sample_features)
        
        print(f"\nTest Prediction:")
        print(f"  Risk Probability: {proba:.4f} ({proba*100:.2f}%)")
        print(f"  Binary Prediction: {prediction} ({'Buggy' if prediction == 1 else 'Clean'})")
        
        print("\n✅ Model loader test complete!")
        
    except FileNotFoundError:
        print("\n❌ Model file not found!")
        print("Please train a model first: python scripts/run_training.py")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")