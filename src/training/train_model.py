"""
Model Training
Trains baseline and advanced models for bug prediction
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains ML models for bug prediction
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize ModelTrainer
        
        Args:
            config: Training configuration
        """
        self.config = config or {}
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
        logger.info("ModelTrainer initialized")
    
    def prepare_data(
        self,
        features_df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        use_time_split: bool = True
    ):
        """
        Prepare data for training
        
        Args:
            features_df: DataFrame with features and labels
            test_size: Proportion of data for testing
            random_state: Random seed
            use_time_split: Use time-based split instead of random
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Preparing data: {len(features_df)} samples")
        
        # Separate features and labels
        label_col = 'is_buggy'
        exclude_cols = ['commit_hash', 'author', 'is_buggy', 'timestamp']
        
        feature_cols = [col for col in features_df.columns 
                       if col not in exclude_cols]
        
        X = features_df[feature_cols].copy()
        y = features_df[label_col].copy()
        
        logger.info(f"Features: {len(feature_cols)} columns")
        logger.info(f"Labels: {len(y)} samples")
        
        # Handle missing values
        X = X.fillna(0)
        
        # Time-based or random split
        if use_time_split and 'timestamp' in features_df.columns:
            logger.info("Using time-based train/test split")
            # Sort by timestamp
            features_df = features_df.sort_values('timestamp')
            split_idx = int(len(features_df) * (1 - test_size))
            
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
        else:
            logger.info("Using random train/test split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=y
            )
        
        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Training bug ratio: {y_train.mean():.2%}")
        logger.info(f"Test bug ratio: {y_test.mean():.2%}")
        
        # Store feature names
        self.feature_names = feature_cols
        
        return X_train, X_test, y_train, y_test
    
    def handle_imbalance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        method: str = 'class_weight'
    ):
        """
        Handle class imbalance
        
        Args:
            X_train: Training features
            y_train: Training labels
            method: 'class_weight' or 'smote'
        
        Returns:
            X_train, y_train (potentially resampled)
        """
        logger.info(f"Handling class imbalance using: {method}")
        
        if method == 'smote':
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"After SMOTE: {len(X_train)} samples")
            logger.info(f"New bug ratio: {y_train.mean():.2%}")
        
        return X_train, y_train
    
    def train_baseline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        class_weight: str = 'balanced'
    ):
        """
        Train baseline Logistic Regression model
        
        Args:
            X_train: Training features
            y_train: Training labels
            class_weight: How to handle class imbalance
        
        Returns:
            Trained model
        """
        logger.info("Training baseline model (Logistic Regression)...")
        
        model = LogisticRegression(
            class_weight=class_weight,
            max_iter=1000,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        self.models['logistic_regression'] = model
        logger.info("✅ Baseline model trained")
        
        return model
    
    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: dict = None
    ):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            params: XGBoost parameters
        
        Returns:
            Trained model
        """
        logger.info("Training advanced model (XGBoost)...")
        
        # Default parameters
        default_params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            'random_state': 42
        }
        
        if params:
            default_params.update(params)
        
        model = XGBClassifier(**default_params)
        model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        logger.info("✅ XGBoost model trained")
        
        return model
    
    def save_model(self, model, model_name: str, output_dir: str = "models"):
        """
        Save trained model to disk
        
        Args:
            model: Trained model
            model_name: Name for the model file
            output_dir: Directory to save model
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        filepath = f"{output_dir}/{model_name}.pkl"
        joblib.dump(model, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def get_feature_importance(self, model_name: str, top_n: int = 10) -> pd.DataFrame:
        """
        Get feature importance from trained model
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to return
        
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.models:
            logger.error(f"Model '{model_name}' not found")
            return None
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Tree-based models (XGBoost)
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Linear models (Logistic Regression)
            importance = np.abs(model.coef_[0])
        else:
            logger.warning(f"Model '{model_name}' doesn't support feature importance")
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("TESTING MODEL TRAINER")
    logger.info("=" * 70)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    sample_data = pd.DataFrame({
        'commit_hash': [f'commit_{i}' for i in range(n_samples)],
        'lines_added': np.random.randint(10, 500, n_samples),
        'lines_deleted': np.random.randint(5, 100, n_samples),
        'files_changed': np.random.randint(1, 20, n_samples),
        'total_churn': np.random.randint(20, 600, n_samples),
        'bug_rate': np.random.uniform(0, 0.5, n_samples),
        'complexity_score': np.random.uniform(0, 1, n_samples),
        'hour_of_day': np.random.randint(0, 24, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
        'is_buggy': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    print(f"\nSample data: {len(sample_data)} commits")
    print(f"Bug ratio: {sample_data['is_buggy'].mean():.2%}")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        sample_data,
        use_time_split=False
    )
    
    # Train baseline
    baseline_model = trainer.train_baseline(X_train, y_train)
    
    # Train XGBoost
    xgb_model = trainer.train_xgboost(X_train, y_train)
    
    # Get feature importance
    print("\nTop 5 important features (XGBoost):")
    importance = trainer.get_feature_importance('xgboost', top_n=5)
    print(importance)
    
    print("\n✅ Model training test complete!")