"""
Model Evaluation
Evaluates trained models with comprehensive metrics
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluates ML model performance
    """
    
    def __init__(self):
        """Initialize ModelEvaluator"""
        self.results = {}
        logger.info("ModelEvaluator initialized")
    
    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model"
    ) -> dict:
        """
        Evaluate a trained model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
        
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Store results
        self.results[model_name] = metrics
        
        # Log metrics
        logger.info(f"Results for {model_name}:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all evaluated models
        
        Returns:
            DataFrame with comparison
        """
        if not self.results:
            logger.warning("No models to compare")
            return None
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df.drop('confusion_matrix', axis=1)
        
        return comparison_df
    
    def print_confusion_matrix(self, model_name: str):
        """
        Print confusion matrix in readable format
        
        Args:
            model_name: Name of the model
        """
        if model_name not in self.results:
            logger.error(f"Model '{model_name}' not found")
            return
        
        cm = np.array(self.results[model_name]['confusion_matrix'])
        
        print(f"\nConfusion Matrix for {model_name}:")
        print("=" * 40)
        print(f"                Predicted")
        print(f"              Clean  Buggy")
        print(f"Actual Clean    {cm[0,0]:4d}   {cm[0,1]:4d}")
        print(f"       Buggy    {cm[1,0]:4d}   {cm[1,1]:4d}")
        print("=" * 40)
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\nBreakdown:")
        print(f"  True Negatives  (TN): {tn:4d} - Correctly predicted clean")
        print(f"  False Positives (FP): {fp:4d} - Clean predicted as buggy")
        print(f"  False Negatives (FN): {fn:4d} - Buggy predicted as clean ⚠️")
        print(f"  True Positives  (TP): {tp:4d} - Correctly predicted buggy")
        
        if fn > 0:
            print(f"\n⚠️  Warning: {fn} bugs were missed!")
    
    def plot_confusion_matrix(
        self,
        model_name: str,
        save_path: str = None
    ):
        """
        Plot confusion matrix as heatmap
        
        Args:
            model_name: Name of the model
            save_path: Path to save plot (optional)
        """
        if model_name not in self.results:
            logger.error(f"Model '{model_name}' not found")
            return
        
        cm = np.array(self.results[model_name]['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Clean', 'Buggy'],
            yticklabels=['Clean', 'Buggy']
        )
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.close()
    
    def get_classification_report(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> str:
        """
        Get detailed classification report
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Classification report string
        """
        y_pred = model.predict(X_test)
        report = classification_report(
            y_test,
            y_pred,
            target_names=['Clean', 'Buggy'],
            zero_division=0
        )
        
        return report
    
    def select_best_model(self, primary_metric: str = 'recall') -> str:
        """
        Select best model based on primary metric
        
        Args:
            primary_metric: Metric to optimize ('recall', 'precision', 'f1_score', 'roc_auc')
        
        Returns:
            Name of best model
        """
        if not self.results:
            logger.error("No models evaluated")
            return None
        
        best_model = max(
            self.results.items(),
            key=lambda x: x[1][primary_metric]
        )[0]
        
        logger.info(f"Best model (by {primary_metric}): {best_model}")
        logger.info(f"  {primary_metric}: {self.results[best_model][primary_metric]:.4f}")
        
        return best_model
    
    def generate_summary_report(self) -> str:
        """
        Generate a comprehensive summary report
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("MODEL EVALUATION SUMMARY")
        report.append("=" * 70)
        
        if not self.results:
            report.append("No models evaluated yet.")
            return "\n".join(report)
        
        # Comparison table
        comparison = self.compare_models()
        report.append("\nModel Comparison:")
        report.append(str(comparison))
        
        # Best model by different metrics
        report.append("\nBest Models by Metric:")
        for metric in ['recall', 'precision', 'f1_score', 'roc_auc']:
            best = max(self.results.items(), key=lambda x: x[1][metric])
            report.append(f"  {metric.upper()}: {best[0]} ({best[1][metric]:.4f})")
        
        # Recommendations
        report.append("\n" + "=" * 70)
        report.append("RECOMMENDATIONS")
        report.append("=" * 70)
        
        best_recall = self.select_best_model('recall')
        report.append(f"\nFor bug prediction (catching all bugs):")
        report.append(f"  → Use: {best_recall}")
        report.append(f"    Recall: {self.results[best_recall]['recall']:.2%} of bugs caught")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("TESTING MODEL EVALUATOR")
    logger.info("=" * 70)
    
    # Create sample predictions
    np.random.seed(42)
    n_samples = 50
    
    y_test = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Simulate two models with different performance
    # Model 1: High recall, lower precision
    y_pred_model1 = y_test.copy()
    # Introduce some errors
    flip_indices = np.random.choice(n_samples, 10, replace=False)
    y_pred_model1[flip_indices] = 1 - y_pred_model1[flip_indices]
    y_pred_proba1 = np.random.uniform(0.3, 0.9, n_samples)
    
    # Model 2: Balanced
    y_pred_model2 = y_test.copy()
    flip_indices = np.random.choice(n_samples, 5, replace=False)
    y_pred_model2[flip_indices] = 1 - y_pred_model2[flip_indices]
    y_pred_proba2 = np.random.uniform(0.2, 0.8, n_samples)
    
    # Create mock model objects
    class MockModel:
        def __init__(self, y_pred, y_pred_proba):
            self.y_pred = y_pred
            self.y_pred_proba = y_pred_proba
        
        def predict(self, X):
            return self.y_pred
        
        def predict_proba(self, X):
            return np.column_stack([1 - self.y_pred_proba, self.y_pred_proba])
    
    model1 = MockModel(y_pred_model1, y_pred_proba1)
    model2 = MockModel(y_pred_model2, y_pred_proba2)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Create dummy X_test
    X_test = pd.DataFrame(np.random.randn(n_samples, 5))
    
    # Evaluate models
    evaluator.evaluate_model(model1, X_test, y_test, "Model_1")
    evaluator.evaluate_model(model2, X_test, y_test, "Model_2")
    
    # Compare models
    print("\nModel Comparison:")
    comparison = evaluator.compare_models()
    print(comparison)
    
    # Print confusion matrices
    evaluator.print_confusion_matrix("Model_1")
    evaluator.print_confusion_matrix("Model_2")
    
    # Select best model
    best = evaluator.select_best_model('recall')
    print(f"\nBest model for bug catching: {best}")
    
    # Generate summary report
    print("\n" + evaluator.generate_summary_report())
    
    print("\n✅ Model evaluation test complete!")