"""
Utilities for MLFlow experiment tracking.
"""
import os
import mlflow
import tensorflow as tf
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, precision_recall_curve
)
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from config.model_config import (
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_ENABLED, MLFLOW_TAGS
)

def setup_mlflow():
    """
    Set up MLFlow tracking.
    
    Returns:
        bool: Whether MLFlow tracking is enabled.
    """
    if not MLFLOW_TRACKING_ENABLED:
        return False
    
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create experiment if it doesn't exist
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    return True

class MLFlowCallback(tf.keras.callbacks.Callback):
    """
    Callback for logging metrics to MLFlow during training.
    """
    def __init__(self):
        super(MLFlowCallback, self).__init__()
        self.run_id = None
        self.active_run = None
    
    def on_train_begin(self, logs=None):
        """Start MLFlow run at the beginning of training."""
        if mlflow.active_run() is None:
            self.active_run = mlflow.start_run()
            self.run_id = self.active_run.info.run_id
        else:
            self.active_run = mlflow.active_run()
            self.run_id = self.active_run.info.run_id
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at the end of each epoch."""
        if logs is not None:
            for metric_name, metric_value in logs.items():
                mlflow.log_metric(metric_name, metric_value, step=epoch)
    
    def on_train_end(self, logs=None):
        """End MLFlow run at the end of training."""
        if self.active_run is not None and mlflow.active_run() is not None:
            mlflow.end_run()
            self.active_run = None
            self.run_id = None

def log_model_parameters(model, hyperparameters=None):
    """
    Log model parameters to MLFlow.
    
    Args:
        model: TensorFlow model
        hyperparameters: Optional dictionary of hyperparameters
    """
    # Log model architecture parameters
    config = model.get_config() if hasattr(model, 'get_config') else {}
    
    # Log number of layers and parameters
    mlflow.log_param("num_layers", len(model.layers))
    mlflow.log_param("num_parameters", model.count_params())
    
    # Log layer information
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'units'):
            mlflow.log_param(f"layer_{i}_units", layer.units)
        if hasattr(layer, 'activation'):
            if hasattr(layer.activation, '__name__'):
                mlflow.log_param(f"layer_{i}_activation", layer.activation.__name__)
            else:
                mlflow.log_param(f"layer_{i}_activation", str(layer.activation))
    
    # Log hyperparameters if provided
    if hyperparameters:
        for param_name, param_value in hyperparameters.items():
            mlflow.log_param(param_name, param_value)

def log_model_metrics(y_true, y_pred, y_pred_proba):
    """
    Log model metrics to MLFlow.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
    """
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    
    # Log confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    mlflow.log_metric("true_negatives", tn)
    mlflow.log_metric("false_positives", fp)
    mlflow.log_metric("false_negatives", fn)
    mlflow.log_metric("true_positives", tp)
    
    # Calculate and log ROC AUC if applicable
    if len(np.unique(y_true)) > 1:
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        mlflow.log_metric("roc_auc", roc_auc)
    
    # Plot and log confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Negative', 'Positive'], rotation=45)
    plt.yticks(tick_marks, ['Negative', 'Positive'])
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save figure
    confusion_matrix_path = "confusion_matrix.png"
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # Log figure as artifact
    mlflow.log_artifact(confusion_matrix_path)
    
    # Plot and log ROC curve
    if len(np.unique(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Save figure
        roc_curve_path = "roc_curve.png"
        plt.savefig(roc_curve_path)
        plt.close()
        
        # Log figure as artifact
        mlflow.log_artifact(roc_curve_path)
        
        # Plot and log precision-recall curve
        precision_values, recall_values, _ = precision_recall_curve(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(recall_values, precision_values, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        # Save figure
        pr_curve_path = "precision_recall_curve.png"
        plt.savefig(pr_curve_path)
        plt.close()
        
        # Log figure as artifact
        mlflow.log_artifact(pr_curve_path)

def log_model(model, preprocessor=None, X_sample=None, signature=None, registered_model_name=None):
    """
    Log model to MLFlow.
    
    Args:
        model: TensorFlow model
        preprocessor: Optional preprocessor
        X_sample: Sample input data for model signature
        signature: Optional model signature
        registered_model_name: Optional name for model registry
    """
    # Create model signature if not provided
    if signature is None and X_sample is not None:
        # Create a sample input
        if preprocessor:
            # Convert to TF dataset
            sample_dataset = tf.data.Dataset.from_tensor_slices((dict(X_sample), np.zeros(len(X_sample)))).batch(1)
            # Prepare dataset
            processed_dataset = preprocessor.prepare_dataset(sample_dataset)
            # Get first batch
            for features_batch, _ in processed_dataset.take(1):
                # Infer signature from sample input and output
                sample_input = features_batch
                sample_output = model.predict(processed_dataset.take(1))
                signature = infer_signature(sample_input, sample_output)
                break
    
    # Log input feature names
    if X_sample is not None:
        feature_names = X_sample.columns.tolist()
        mlflow.log_param("input_features", str(feature_names))
        
        # Log feature statistics
        feature_stats = {}
        for col in feature_names:
            if pd.api.types.is_numeric_dtype(X_sample[col]):
                feature_stats[f"{col}_mean"] = X_sample[col].mean()
                feature_stats[f"{col}_std"] = X_sample[col].std()
                feature_stats[f"{col}_min"] = X_sample[col].min()
                feature_stats[f"{col}_max"] = X_sample[col].max()
            else:
                value_counts = X_sample[col].value_counts(normalize=True)
                for val, count in value_counts.items():
                    feature_stats[f"{col}_{val}_frequency"] = count
        
        # Log feature statistics as a JSON artifact
        with open("feature_stats.json", "w") as f:
            json.dump(feature_stats, f)
        mlflow.log_artifact("feature_stats.json")
    
    # Save model
    model_path = "model"
    model.save(model_path)
    
    # Log model
    mlflow.tensorflow.log_model(
        tf_saved_model_dir=model_path,
        tf_meta_graph_tags=None,
        tf_signature_def_key="serving_default",
        artifact_path="model",
        signature=signature,
        registered_model_name=registered_model_name
    )
    
    # Log model as artifact
    mlflow.log_artifact(model_path)
    
    # Log preprocessor if provided
    if preprocessor:
        # Save preprocessor configuration
        preprocessor_config = {
            "discrete_features": preprocessor.discrete_features,
            "categorical_features": preprocessor.categorical_features,
            "continuous_features": preprocessor.continuous_features
        }
        
        with open("preprocessor_config.json", "w") as f:
            json.dump(preprocessor_config, f)
        
        mlflow.log_artifact("preprocessor_config.json")

def log_training_history(history):
    """
    Log training history to MLFlow.
    
    Args:
        history: Training history dictionary
    """
    # Log final metrics
    for metric_name, metric_values in history.items():
        if len(metric_values) > 0:
            mlflow.log_metric(f"final_{metric_name}", metric_values[-1])
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy values
    plt.subplot(1, 2, 1)
    if 'accuracy' in history:
        plt.plot(history['accuracy'])
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    if 'loss' in history:
        plt.plot(history['loss'])
    if 'val_loss' in history:
        plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    
    # Save figure
    history_plot_path = "training_history.png"
    plt.savefig(history_plot_path)
    plt.close()
    
    # Log figure as artifact
    mlflow.log_artifact(history_plot_path)
    
    # Save history as JSON
    history_json_path = "training_history.json"
    with open(history_json_path, "w") as f:
        json.dump(history, f)
    
    # Log JSON as artifact
    mlflow.log_artifact(history_json_path)

def start_run(run_name=None, tags=None):
    """
    Start MLFlow run.
    
    Args:
        run_name: Optional run name
        tags: Optional tags
    
    Returns:
        MLFlow run
    """
    if not MLFLOW_TRACKING_ENABLED:
        return None
    
    # Set up MLFlow
    setup_mlflow()
    
    # Set default run name if not provided
    if run_name is None:
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Combine default tags with provided tags
    all_tags = MLFLOW_TAGS.copy()
    if tags:
        all_tags.update(tags)
    
    # Add timestamp tag
    all_tags["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Start run
    run = mlflow.start_run(run_name=run_name)
    
    # Set tags
    for tag_name, tag_value in all_tags.items():
        mlflow.set_tag(tag_name, tag_value)
    
    # Log system info
    mlflow.log_param("tensorflow_version", tf.__version__)
    mlflow.log_param("mlflow_version", mlflow.__version__)
    
    # Log requirements
    try:
        import pkg_resources
        installed_packages = pkg_resources.working_set
        installed_packages_list = sorted([f"{i.key}=={i.version}" for i in installed_packages])
        
        with open("requirements.txt", "w") as f:
            for package in installed_packages_list:
                f.write(f"{package}\n")
        
        mlflow.log_artifact("requirements.txt")
    except Exception as e:
        print(f"Error logging requirements: {e}")
    
    return run

def end_run():
    """End MLFlow run."""
    if MLFLOW_TRACKING_ENABLED and mlflow.active_run():
        mlflow.end_run()

def get_best_run(metric, ascending=False, experiment_name=None):
    """
    Get the best run based on a metric.
    
    Args:
        metric: Metric to use for comparison
        ascending: Whether to sort in ascending order
        experiment_name: Optional experiment name
    
    Returns:
        Best run as a dictionary
    """
    if not MLFLOW_TRACKING_ENABLED:
        return None
    
    # Set up MLFlow
    setup_mlflow()
    
    # Set experiment name
    if experiment_name is None:
        experiment_name = MLFLOW_EXPERIMENT_NAME
    
    # Get experiment ID
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        return None
    
    # Search for runs
    order_by = f"metrics.{metric}" if not ascending else f"metrics.{metric} ASC"
    runs = mlflow.search_runs(
        experiment_ids=experiment.experiment_id,
        order_by=[order_by]
    )
    
    # Return best run
    if len(runs) == 0:
        return None
    
    return runs.iloc[0].to_dict()

def register_model(run_id, model_name, model_version_description=None, stage=None):
    """
    Register a model in the MLFlow Model Registry.
    
    Args:
        run_id: Run ID
        model_name: Model name
        model_version_description: Optional model version description
        stage: Optional stage (None, "Staging", "Production", "Archived")
    
    Returns:
        Registered model version
    """
    if not MLFLOW_TRACKING_ENABLED:
        return None
    
    # Set up MLFlow
    setup_mlflow()
    
    # Create client
    client = MlflowClient()
    
    # Check if model exists
    try:
        client.get_registered_model(model_name)
    except:
        # Create model if it doesn't exist
        client.create_registered_model(model_name)
    
    # Get model URI
    model_uri = f"runs:/{run_id}/model"
    
    # Register model
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )
    
    # Set model version description
    if model_version_description:
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=model_version_description
        )
    
    # Set stage
    if stage:
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=stage
        )
    
    return model_version

def compare_runs(run_ids, metrics=None):
    """
    Compare multiple runs based on metrics.
    
    Args:
        run_ids: List of run IDs
        metrics: Optional list of metrics to compare
    
    Returns:
        DataFrame with run metrics
    """
    if not MLFLOW_TRACKING_ENABLED:
        return None
    
    # Set up MLFlow
    setup_mlflow()
    
    # Create client
    client = MlflowClient()
    
    # Get runs
    runs_data = []
    for run_id in run_ids:
        run = client.get_run(run_id)
        run_data = {
            "run_id": run_id,
            "start_time": datetime.fromtimestamp(run.info.start_time / 1000.0).strftime("%Y-%m-%d %H:%M:%S"),
            "status": run.info.status
        }
        
        # Add tags
        for key, value in run.data.tags.items():
            run_data[f"tag_{key}"] = value
        
        # Add metrics
        for key, value in run.data.metrics.items():
            if metrics is None or key in metrics:
                run_data[f"metric_{key}"] = value
        
        # Add parameters
        for key, value in run.data.params.items():
            run_data[f"param_{key}"] = value
        
        runs_data.append(run_data)
    
    # Create DataFrame
    df = pd.DataFrame(runs_data)
    
    return df

def log_data_validation(X, validation_results=None):
    """
    Log data validation results.
    
    Args:
        X: Input data
        validation_results: Optional validation results dictionary
    """
    # Log data shape
    mlflow.log_param("data_shape", str(X.shape))
    
    # Log data statistics
    data_stats = {
        "num_rows": len(X),
        "num_columns": len(X.columns),
        "missing_values": X.isna().sum().sum(),
        "missing_percentage": (X.isna().sum().sum() / (len(X) * len(X.columns))) * 100
    }
    
    for stat_name, stat_value in data_stats.items():
        mlflow.log_param(f"data_{stat_name}", stat_value)
    
    # Log column statistics
    column_stats = {}
    for col in X.columns:
        column_stats[col] = {
            "dtype": str(X[col].dtype),
            "missing": X[col].isna().sum(),
            "missing_percentage": (X[col].isna().sum() / len(X)) * 100
        }
        
        if pd.api.types.is_numeric_dtype(X[col]):
            column_stats[col].update({
                "mean": X[col].mean(),
                "std": X[col].std(),
                "min": X[col].min(),
                "25%": X[col].quantile(0.25),
                "50%": X[col].quantile(0.5),
                "75%": X[col].quantile(0.75),
                "max": X[col].max()
            })
        else:
            column_stats[col].update({
                "unique_values": X[col].nunique(),
                "top_value": X[col].value_counts().index[0] if not X[col].value_counts().empty else None,
                "top_count": X[col].value_counts().iloc[0] if not X[col].value_counts().empty else 0
            })
    
    # Save column statistics as JSON
    with open("column_stats.json", "w") as f:
        json.dump(column_stats, f)
    
    # Log JSON as artifact
    mlflow.log_artifact("column_stats.json")
    
    # Log validation results if provided
    if validation_results:
        # Save validation results as JSON
        with open("validation_results.json", "w") as f:
            json.dump(validation_results, f)
        
        # Log JSON as artifact
        mlflow.log_artifact("validation_results.json")
        
        # Log validation status
        mlflow.log_param("validation_status", validation_results.get("status", "unknown"))

def get_model_from_registry(model_name, stage=None, version=None):
    """
    Get a model from the MLFlow Model Registry.
    
    Args:
        model_name: Model name
        stage: Optional stage (None, "Staging", "Production", "Archived")
        version: Optional version
    
    Returns:
        Loaded model
    """
    if not MLFLOW_TRACKING_ENABLED:
        return None
    
    # Set up MLFlow
    setup_mlflow()
    
    # Determine model URI
    if version is not None:
        model_uri = f"models:/{model_name}/{version}"
    elif stage is not None:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        model_uri = f"models:/{model_name}/latest"
    
    # Load model
    model = mlflow.tensorflow.load_model(model_uri)
    
    return model 