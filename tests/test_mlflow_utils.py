import pytest
import mlflow
import tensorflow as tf
import numpy as np
from src.utils.mlflow_utils import (
    setup_mlflow, MLFlowCallback, log_model_parameters,
    log_model_metrics, log_model, log_training_history,
    start_run, end_run, get_best_run
)

def test_setup_mlflow(monkeypatch):
    """Test MLFlow setup."""
    # Mock MLFlow configuration
    monkeypatch.setenv('MLFLOW_TRACKING_URI', 'mlruns')
    
    # Test setup
    is_enabled = setup_mlflow()
    assert is_enabled is True
    
    # Check experiment exists
    experiment = mlflow.get_experiment_by_name('credit_risk_model')
    assert experiment is not None

def test_mlflow_callback():
    """Test MLFlow callback functionality."""
    callback = MLFlowCallback()
    
    # Test initialization
    assert callback.run_id is None
    assert callback.active_run is None
    
    # Test on_train_begin
    with mlflow.start_run() as run:
        callback.on_train_begin()
        assert callback.active_run is not None
    
    # Test on_epoch_end
    with mlflow.start_run() as run:
        callback.on_train_begin()
        logs = {'loss': 0.5, 'accuracy': 0.8}
        callback.on_epoch_end(0, logs)
        
        # Check metrics were logged
        run_data = mlflow.get_run(run.info.run_id)
        assert 'loss' in run_data.data.metrics
        assert 'accuracy' in run_data.data.metrics

def test_log_model_parameters(preprocessor):
    """Test logging model parameters."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    with mlflow.start_run() as run:
        log_model_parameters(model)
        
        # Check parameters were logged
        run_data = mlflow.get_run(run.info.run_id)
        assert len(run_data.data.params) > 0

def test_log_model_metrics():
    """Test logging model metrics."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.6, 0.8])
    
    with mlflow.start_run() as run:
        log_model_metrics(y_true, y_pred, y_pred_proba)
        
        # Check metrics were logged
        run_data = mlflow.get_run(run.info.run_id)
        assert 'accuracy' in run_data.data.metrics
        assert 'f1_score' in run_data.data.metrics
        assert 'roc_auc' in run_data.data.metrics

def test_log_training_history():
    """Test logging training history."""
    history = {
        'loss': [0.5, 0.4],
        'accuracy': [0.8, 0.85],
        'val_loss': [0.45, 0.35],
        'val_accuracy': [0.82, 0.87]
    }
    
    with mlflow.start_run() as run:
        log_training_history(history)
        
        # Check metrics were logged
        run_data = mlflow.get_run(run.info.run_id)
        assert 'accuracy' in run_data.data.metrics
        assert 'val_accuracy' in run_data.data.metrics

def test_start_and_end_run():
    """Test run management functions."""
    # Test start_run
    run = start_run(run_name='test_run')
    assert run is not None
    assert mlflow.active_run() is not None
    
    # Test end_run
    end_run()
    assert mlflow.active_run() is None

def test_get_best_run():
    """Test retrieving best run."""
    # Create some test runs
    with mlflow.start_run() as run1:
        mlflow.log_metric('val_accuracy', 0.8)
    
    with mlflow.start_run() as run2:
        mlflow.log_metric('val_accuracy', 0.9)
    
    # Get best run
    best_run = get_best_run('val_accuracy', ascending=False)
    assert best_run is not None
    assert best_run['metrics.val_accuracy'] == 0.9 