import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, classification_report, confusion_matrix

def plot_evaluation_metrics(config,y_true, y_pred_prob, y_pred) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'TPR')
    axes[0].set_xlabel('Thresholds')
    axes[0].set_ylabel('Rate')
    axes[0].set_title(f'ROC Curve (AUC = {roc_auc:.2f})')
    axes[0].legend(loc="lower right")

    # Precision-Recall Curve
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = average_precision_score(y_true, y_pred_prob)
    axes[1].plot(recall[:-1], precision[:-1], color='darkorange', lw=2, label=f'Precision-Recall Curve')
    axes[1].set_xlabel('Thresholds')
    axes[1].set_ylabel('Score')
    axes[1].set_title(f'Precision-Recall Curve (AUC = {pr_auc:.2f})')
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(config["PATHS_CONFIG"]["metrics_directory"] + 'evaluation_metrics.png')

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(config["PATHS_CONFIG"]["metrics_directory"] + 'confusion_matrix.png')

    # Create a single figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Precision-Recall vs Threshold plot
    axes[0].plot(thresholds_pr, precision[:-1], color='darkorange', lw=2, label='Precision')
    axes[0].plot(thresholds_pr, recall[:-1], color='blue', lw=2, label='Recall')
    axes[0].set_xlabel('Thresholds')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Precision-Recall vs Threshold Curve')
    axes[0].legend(loc="lower left")

    # TPR-FPR vs Threshold plot
    axes[1].plot(thresholds_roc, tpr, color='darkorange', lw=2, label='TPR')
    axes[1].plot(thresholds_roc, fpr, color='blue', lw=2, label='FPR')
    axes[1].set_xlabel('Thresholds')
    axes[1].set_ylabel('Rate')
    axes[1].set_title('TPR-FPR vs Threshold Curve')
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(config["PATHS_CONFIG"]["metrics_directory"] + 'threshold_curves.png')

    # Classification Report
    print(classification_report(y_true, y_pred))
