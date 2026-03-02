"""
Run Model Training Pipeline
Trains and evaluates ML models for bug prediction
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime
from src.training.train_model import ModelTrainer
from src.training.evaluate import ModelEvaluator
from src.utils.config_loader import ConfigLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main training pipeline
    """
    logger.info("=" * 70)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 70)
    
    try:
        # Load configuration
        config_loader = ConfigLoader()
        config = config_loader.load_main_config()
        
        # Get settings
        training_config = config['training']
        features_path = "data/features/commit_features.csv"
        models_dir = "models"
        
        logger.info(f"Features input: {features_path}")
        logger.info(f"Models output: {models_dir}")
        
        # Load features
        logger.info("Loading features...")
        features_df = pd.read_csv(features_path)
        logger.info(f"Loaded {len(features_df)} samples with {len(features_df.columns)} features")
        
        # Check label distribution
        if 'is_buggy' in features_df.columns:
            bug_ratio = features_df['is_buggy'].mean()
            logger.info(f"Bug ratio: {bug_ratio:.2%}")
            
            if bug_ratio < 0.05:
                logger.warning("⚠️  Very low bug ratio - model may struggle")
        else:
            logger.error("Label column 'is_buggy' not found!")
            raise ValueError("Missing label column")
        
        # Initialize MLflow
        mlflow_config = config.get('mlflow', {})
        mlflow.set_tracking_uri(mlflow_config.get('tracking_uri', 'file:./models/registry'))
        mlflow.set_experiment(mlflow_config.get('experiment_name', 'commit_risk_scoring'))
        
        # Initialize trainer
        trainer = ModelTrainer(config=training_config)
        
        # Prepare data
        logger.info("Preparing data...")
        X_train, X_test, y_train, y_test = trainer.prepare_data(
            features_df,
            test_size=training_config.get('test_size', 0.2),
            random_state=training_config.get('random_state', 42),
            use_time_split=training_config.get('use_time_split', True)
        )
        
        # Handle class imbalance
        imbalance_method = training_config.get('imbalance_method', 'class_weight')
        if training_config.get('handle_imbalance', True):
            if imbalance_method == 'SMOTE':
                X_train, y_train = trainer.handle_imbalance(X_train, y_train, method='smote')
        
        # Initialize evaluator
        evaluator = ModelEvaluator()
        
        # Train and evaluate models
        logger.info("=" * 70)
        logger.info("TRAINING MODELS")
        logger.info("=" * 70)
        
        # Model 1: Logistic Regression
        if training_config['models']['logistic_regression']['enabled']:
            logger.info("\n--- Training Logistic Regression (Baseline) ---")
            
            with mlflow.start_run(run_name="logistic_regression"):
                # Train
                lr_model = trainer.train_baseline(
                    X_train, y_train,
                    class_weight='balanced'
                )
                
                # Evaluate
                lr_metrics = evaluator.evaluate_model(
                    lr_model, X_test, y_test,
                    model_name="Logistic Regression"
                )
                
                # Log to MLflow
                mlflow.log_params(training_config['models']['logistic_regression']['params'])
                mlflow.log_metrics({
                    'accuracy': lr_metrics['accuracy'],
                    'precision': lr_metrics['precision'],
                    'recall': lr_metrics['recall'],
                    'f1_score': lr_metrics['f1_score'],
                    'roc_auc': lr_metrics['roc_auc']
                })
                
                # Save model
                trainer.save_model(lr_model, 'baseline_logistic', models_dir)
                mlflow.sklearn.log_model(lr_model, "model")
                
                # Print confusion matrix
                evaluator.print_confusion_matrix("Logistic Regression")
        
        # Model 2: XGBoost
        if training_config['models']['xgboost']['enabled']:
            logger.info("\n--- Training XGBoost (Advanced) ---")
            
            with mlflow.start_run(run_name="xgboost"):
                # Train
                xgb_params = training_config['models']['xgboost']['params']
                xgb_model = trainer.train_xgboost(
                    X_train, y_train,
                    params=xgb_params
                )
                
                # Evaluate
                xgb_metrics = evaluator.evaluate_model(
                    xgb_model, X_test, y_test,
                    model_name="XGBoost"
                )
                
                # Log to MLflow
                mlflow.log_params(xgb_params)
                mlflow.log_metrics({
                    'accuracy': xgb_metrics['accuracy'],
                    'precision': xgb_metrics['precision'],
                    'recall': xgb_metrics['recall'],
                    'f1_score': xgb_metrics['f1_score'],
                    'roc_auc': xgb_metrics['roc_auc']
                })
                
                # Save model
                trainer.save_model(xgb_model, 'advanced_xgboost', models_dir)
                mlflow.sklearn.log_model(xgb_model, "model")
                
                # Print confusion matrix
                evaluator.print_confusion_matrix("XGBoost")
                
                # Feature importance
                importance = trainer.get_feature_importance('xgboost', top_n=10)
                logger.info("\nTop 10 Important Features (XGBoost):")
                for idx, row in importance.iterrows():
                    logger.info(f"  {row['feature']:25s}: {row['importance']:.4f}")
        
        # Compare models
        logger.info("\n" + "=" * 70)
        logger.info("MODEL COMPARISON")
        logger.info("=" * 70)
        
        comparison = evaluator.compare_models()
        logger.info("\n" + str(comparison))
        
        # Select best model
        primary_metric = config['evaluation'].get('primary_metric', 'recall')
        best_model_name = evaluator.select_best_model(primary_metric)
        
        # Generate summary report
        summary = evaluator.generate_summary_report()
        logger.info("\n" + summary)
        
        # Save summary to file
        summary_path = f"{models_dir}/training_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(summary)
            f.write(f"\n\nTraining completed at: {datetime.now()}")
        logger.info(f"Summary saved to {summary_path}")
        
        logger.info("\n" + "=" * 70)
        logger.info("✅ MODEL TRAINING COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\nBest Model: {best_model_name}")
        logger.info(f"Primary Metric ({primary_metric}): {evaluator.results[best_model_name][primary_metric]:.4f}")
        logger.info(f"\nModels saved in: {models_dir}/")
        logger.info(f"MLflow tracking: {mlflow.get_tracking_uri()}")
        
    except FileNotFoundError as e:
        logger.error(f"❌ Input file not found: {e}")
        logger.error("Please run feature engineering first:")
        logger.error("  python scripts/run_feature_engineering.py")
        raise
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()