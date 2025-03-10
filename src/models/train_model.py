import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import keras_tuner as kt
import os
import sys
import argparse
import json
from datetime import datetime
import numpy as np

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.data.make_dataset import load_data, create_tf_datasets, CreditDataPreprocessor
from src.models.model import CreditRiskModel
from src.utils.mlflow_utils import (
    start_run, end_run, log_model_parameters, log_model_metrics, 
    log_model, log_training_history, MLFlowCallback, 
    register_model, log_data_validation
)
from config.model_config import (
    DISCRETE_FEATURES, CATEGORICAL_FEATURES, CONTINUOUS_FEATURES,
    BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, RANDOM_SEED,
    HP_MAX_TRIALS, HP_EXECUTIONS_PER_TRIAL, HP_DIRECTORY,
    HP_UNITS1_MIN, HP_UNITS1_MAX, HP_UNITS1_STEP,
    HP_UNITS2_MIN, HP_UNITS2_MAX, HP_UNITS2_STEP,
    HP_DROPOUT_MIN, HP_DROPOUT_MAX, HP_DROPOUT_STEP,
    HP_LR_MIN, HP_LR_MAX,
    EARLY_STOPPING_PATIENCE, REDUCE_LR_PATIENCE, REDUCE_LR_FACTOR, MIN_DELTA,
    CHECKPOINT_PATH, DEFAULT_MODEL_PATH, BEST_TUNED_MODEL_PATH,
    PREPROCESSOR_CONFIG_PATH, EVALUATION_METRICS_PATH,
    MLFLOW_TRACKING_ENABLED
)


def build_model_for_tuning(hp, preprocessor):
    """
    Build model for hyperparameter tuning.
    
    Args:
        hp: Hyperparameters
        preprocessor: Data preprocessor
    
    Returns:
        Compiled model
    """
    # Create model wrapper
    model_wrapper = CreditRiskModel(
        preprocessor,
        embedding_size=hp.Int('embedding_size', min_value=4, max_value=32, step=4)
    )
    
    # Get model
    model = model_wrapper.model
    
    # Add custom layers
    x = model.output
    
    # First dense layer
    units_1 = hp.Int('units_1', min_value=16, max_value=256, step=16)
    x = tf.keras.layers.Dense(units_1, activation='relu')(x)
    
    # Dropout
    dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Second dense layer (optional)
    if hp.Boolean('second_layer'):
        units_2 = hp.Int('units_2', min_value=8, max_value=128, step=8)
        x = tf.keras.layers.Dense(units_2, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    # Create model
    tuned_model = tf.keras.Model(inputs=model.inputs, outputs=outputs)
    
    # Compile model
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    tuned_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return tuned_model


def train_model(
    train_dataset, 
    val_dataset, 
    preprocessor, 
    model_dir, 
    epochs=MAX_EPOCHS, 
    batch_size=BATCH_SIZE,
    use_tuner=False,
    max_trials=HP_MAX_TRIALS,
    executions_per_trial=HP_EXECUTIONS_PER_TRIAL,
    register_to_mlflow=True
):
    """
    Train the credit risk model.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        preprocessor: Data preprocessor
        model_dir: Directory to save model artifacts
        epochs: Number of training epochs
        batch_size: Batch size
        use_tuner: Whether to use hyperparameter tuning
        max_trials: Maximum number of trials for hyperparameter tuning
        executions_per_trial: Number of executions per trial
        register_to_mlflow: Whether to register the model to MLFlow registry
    
    Returns:
        Trained model
    """
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Start MLFlow run
    run_name = f"hyperparameter_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if use_tuner else f"standard_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run = start_run(run_name=run_name, tags={
        "model_type": "tuned_model" if use_tuner else "standard_model",
        "phase": "training"
    })
    
    try:
        # Log data validation information
        # Extract a sample of the data for validation
        for features_batch, labels_batch in train_dataset.take(1):
            # Convert to pandas DataFrame for validation
            features_dict = {key: value.numpy() for key, value in features_batch.items()}
            X_sample = {}
            for feature in preprocessor.continuous_features:
                X_sample[feature] = features_dict.get(f"{feature}_input", [])
            for feature in preprocessor.discrete_features:
                X_sample[feature] = features_dict.get(f"{feature}_input", [])
            for feature in preprocessor.categorical_features:
                X_sample[feature] = features_dict.get(f"{feature}_input", [])
            
            import pandas as pd
            X_sample_df = pd.DataFrame(X_sample)
            log_data_validation(X_sample_df)
            break
        
        # Create callbacks
        callbacks = []
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_delta=MIN_DELTA
        )
        callbacks.append(reduce_lr)
        
        # Model checkpoint
        checkpoint_path = os.path.join(model_dir, 'checkpoint.keras')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False
        )
        callbacks.append(checkpoint)
        
        # MLFlow callback
        if MLFLOW_TRACKING_ENABLED:
            mlflow_callback = MLFlowCallback()
            callbacks.append(mlflow_callback)
        
        # Use hyperparameter tuning if specified
        if use_tuner:
            # Log hyperparameter tuning parameters
            tuner_params = {
                'max_trials': max_trials,
                'executions_per_trial': executions_per_trial
            }
            log_model_parameters(None, tuner_params)
            
            # Create tuner
            hp_dir = os.path.join(model_dir, HP_DIRECTORY)
            tuner = kt.Hyperband(
                lambda hp: build_model_for_tuning(hp, preprocessor),
                objective='val_accuracy',
                max_epochs=epochs,
                factor=3,
                directory=hp_dir,
                project_name='credit_risk'
            )
            
            # Search for best hyperparameters
            tuner.search(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks
            )
            
            # Get best hyperparameters
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            
            # Log best hyperparameters
            best_hp_dict = {
                'units_1': best_hps.get('units_1'),
                'units_2': best_hps.get('units_2'),
                'learning_rate': best_hps.get('learning_rate'),
                'dropout_rate': best_hps.get('dropout_rate')
            }
            log_model_parameters(None, best_hp_dict)
            
            # Build model with best hyperparameters
            model = build_model_for_tuning(best_hps, preprocessor)
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate')),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model with best hyperparameters
            history = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks
            )
            
            # Save model
            model_path = os.path.join(model_dir, os.path.basename(BEST_TUNED_MODEL_PATH))
            model.save(model_path)
            
            # Create model wrapper
            model_wrapper = CreditRiskModel(preprocessor)
            model_wrapper.model = model
            
            # Log model to MLFlow with signature
            for features_batch, _ in train_dataset.take(1):
                log_model(
                    model, 
                    preprocessor=preprocessor, 
                    X_sample=X_sample_df,
                    registered_model_name="credit_risk_model" if register_to_mlflow else None
                )
                break
            
            # Register model to MLFlow registry if enabled
            if register_to_mlflow and MLFLOW_TRACKING_ENABLED and run:
                register_model(
                    run_id=run.info.run_id,
                    model_name="credit_risk_model",
                    model_version_description=f"Tuned model with {best_hp_dict}",
                    stage="Staging"
                )
        else:
            # Create model
            model_wrapper = CreditRiskModel(preprocessor)
            
            # Compile model
            model_wrapper.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Log model parameters
            log_model_parameters(model_wrapper.model, {
                'learning_rate': LEARNING_RATE,
                'batch_size': batch_size,
                'epochs': epochs
            })
            
            # Train model
            history = model_wrapper.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=epochs,
                callbacks=callbacks
            )
            
            # Save model
            model_path = os.path.join(model_dir, os.path.basename(DEFAULT_MODEL_PATH))
            model_wrapper.save_model(model_path)
            
            # Log model to MLFlow with signature
            for features_batch, _ in train_dataset.take(1):
                log_model(
                    model_wrapper.model, 
                    preprocessor=preprocessor, 
                    X_sample=X_sample_df,
                    registered_model_name="credit_risk_model" if register_to_mlflow else None
                )
                break
            
            # Register model to MLFlow registry if enabled
            if register_to_mlflow and MLFLOW_TRACKING_ENABLED and run:
                register_model(
                    run_id=run.info.run_id,
                    model_name="credit_risk_model",
                    model_version_description="Standard model",
                    stage="Staging"
                )
        
        # Save training history
        history_path = os.path.join(model_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history.history, f)
        
        # Log training history to MLFlow
        log_training_history(history.history)
        
        return model_wrapper
    
    finally:
        # End MLFlow run
        end_run()


def main():
    """Main function to train the model."""
    parser = argparse.ArgumentParser(description='Train credit risk model')
    parser.add_argument('--data_path', type=str, default=None, 
                        help='Path to the data file. If not provided, data will be fetched from OpenML.')
    parser.add_argument('--model_dir', type=str, default='../../models',
                        help='Directory to save model artifacts')
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='Batch size for training')
    parser.add_argument('--use_tuner', action='store_true',
                        help='Use hyperparameter tuning')
    parser.add_argument('--max_trials', type=int, default=HP_MAX_TRIALS,
                        help='Maximum number of trials for hyperparameter tuning')
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create model directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                            f'../../../models/{timestamp}'))
    os.makedirs(model_dir, exist_ok=True)
    
    # Resolve data path
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = os.path.join(os.path.dirname(__file__), '../../../data/raw/credit-g.csv')
        if not os.path.exists(data_path):
            data_path = None
    
    # Load data
    X, y = load_data(data_path)
    
    # Define feature columns
    discrete_features = DISCRETE_FEATURES
    continuous_features = CONTINUOUS_FEATURES
    categorical_features = CATEGORICAL_FEATURES if CATEGORICAL_FEATURES is not None else X.select_dtypes(exclude='number').columns.tolist()
    
    # Save preprocessor configuration
    preprocessor_config = {
        'discrete_features': discrete_features,
        'categorical_features': categorical_features,
        'continuous_features': continuous_features
    }
    
    with open(os.path.join(model_dir, os.path.basename(PREPROCESSOR_CONFIG_PATH)), 'w') as f:
        json.dump(preprocessor_config, f)
    
    # Create datasets
    train_dataset_raw, val_dataset_raw, test_dataset_raw = create_tf_datasets(
        X, y, batch_size=args.batch_size, seed=args.seed
    )
    
    # Initialize and adapt preprocessor
    preprocessor = CreditDataPreprocessor(
        discrete_features=discrete_features,
        categorical_features=categorical_features,
        continuous_features=continuous_features
    )
    preprocessor.adapt(train_dataset_raw)
    
    # Prepare datasets
    train_dataset = preprocessor.prepare_dataset(train_dataset_raw)
    val_dataset = preprocessor.prepare_dataset(val_dataset_raw)
    test_dataset = preprocessor.prepare_dataset(test_dataset_raw)
    
    # Start MLFlow run for the entire training process
    run_name = f"training_{timestamp}"
    run = start_run(run_name=run_name, tags={"phase": "training"})
    
    try:
        # Train model
        model = train_model(
            train_dataset, 
            val_dataset, 
            preprocessor, 
            model_dir, 
            epochs=args.epochs, 
            batch_size=args.batch_size,
            use_tuner=args.use_tuner,
            max_trials=args.max_trials
        )
        
        # Evaluate model
        test_loss, test_acc = model.model.evaluate(test_dataset)
        print(f"Test loss: {test_loss} - Test accuracy: {test_acc}")
        
        # Save evaluation metrics
        with open(os.path.join(model_dir, os.path.basename(EVALUATION_METRICS_PATH)), 'w') as f:
            json.dump({'test_loss': float(test_loss), 'test_accuracy': float(test_acc)}, f)
        
        # Log evaluation metrics to MLFlow
        if MLFLOW_TRACKING_ENABLED:
            # Get predictions for test dataset
            y_pred_proba = model.model.predict(test_dataset)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Extract true labels from test dataset
            y_true = []
            for batch in test_dataset:
                _, labels = batch
                y_true.extend(labels.numpy())
            
            # Log metrics
            log_model_metrics(y_true, y_pred, y_pred_proba)
        
        print(f"Model artifacts saved to {model_dir}")
    finally:
        # End MLFlow run
        end_run()


if __name__ == "__main__":
    main() 