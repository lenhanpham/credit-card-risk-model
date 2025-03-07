import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import os
import sys
import argparse
import json

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from credit_risk_model.config.model_config import (
    FIGURES_DIR, CONFUSION_MATRIX_PATH, ROC_CURVE_PATH, 
    PRECISION_RECALL_CURVE_PATH, TRAINING_HISTORY_PATH
)


def plot_confusion_matrix(y_true, y_pred, output_path=None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Bad', 'Good'], 
                yticklabels=['Bad', 'Good'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()


def plot_roc_curve(y_true, y_pred_proba, output_path=None):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {output_path}")
    else:
        plt.show()


def plot_precision_recall_curve(y_true, y_pred_proba, output_path=None):
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save the plot
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Precision-recall curve saved to {output_path}")
    else:
        plt.show()


def plot_feature_importance(feature_names, importance_values, output_path=None):
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importance_values: List of importance values
        output_path: Path to save the plot
    """
    # Sort features by importance
    indices = np.argsort(importance_values)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importance_values[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance')
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {output_path}")
    else:
        plt.show()


def plot_training_history(history, output_path=None):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {output_path}")
    else:
        plt.show()


def main():
    """Main function to create visualizations."""
    parser = argparse.ArgumentParser(description='Create visualizations for credit risk model')
    parser.add_argument('--predictions_path', type=str, required=True,
                        help='Path to the predictions CSV file')
    parser.add_argument('--history_path', type=str, default=None,
                        help='Path to the training history JSON file')
    parser.add_argument('--output_dir', type=str, default=FIGURES_DIR,
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load predictions
    predictions_df = pd.read_csv(args.predictions_path)
    
    # Extract true labels and predicted probabilities
    y_true = predictions_df['class'].map({'good': 1, 'bad': 0}).values
    y_pred_proba = predictions_df['prediction'].values
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Create visualizations
    plot_confusion_matrix(
        y_true, 
        y_pred, 
        output_path=os.path.join(args.output_dir, os.path.basename(CONFUSION_MATRIX_PATH))
    )
    
    plot_roc_curve(
        y_true, 
        y_pred_proba, 
        output_path=os.path.join(args.output_dir, os.path.basename(ROC_CURVE_PATH))
    )
    
    plot_precision_recall_curve(
        y_true, 
        y_pred_proba, 
        output_path=os.path.join(args.output_dir, os.path.basename(PRECISION_RECALL_CURVE_PATH))
    )
    
    # Plot training history if provided
    if args.history_path and os.path.exists(args.history_path):
        with open(args.history_path, 'r') as f:
            history = json.load(f)
        
        plot_training_history(
            history, 
            output_path=os.path.join(args.output_dir, os.path.basename(TRAINING_HISTORY_PATH))
        )


if __name__ == "__main__":
    main() 