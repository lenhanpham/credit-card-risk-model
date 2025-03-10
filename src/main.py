import argparse
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import mlflow

# Add the project root directory to the Python path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.make_dataset import load_data, create_tf_datasets, CreditDataPreprocessor
from src.models.train_model import train_model
from src.models.predict_model import predict, load_model
from src.visualization.visualize import (
    plot_confusion_matrix, 
    plot_roc_curve, 
    plot_precision_recall_curve, 
    plot_training_history
)
from src.utils.mlflow_utils import (
    start_run, end_run, log_model_metrics, log_training_history,
    get_best_run, register_model, get_model_from_registry, compare_runs
)
from config.model_config import (
    DISCRETE_FEATURES, CATEGORICAL_FEATURES, CONTINUOUS_FEATURES,
    BATCH_SIZE, MAX_EPOCHS, RANDOM_SEED,
    HP_MAX_TRIALS, DEFAULT_MODEL_PATH, BEST_TUNED_MODEL_PATH,
    PREPROCESSOR_CONFIG_PATH, EVALUATION_METRICS_PATH,
    FIGURES_DIR, CONFUSION_MATRIX_PATH, ROC_CURVE_PATH,
    PRECISION_RECALL_CURVE_PATH, TRAINING_HISTORY_PATH,
    MLFLOW_TRACKING_ENABLED, MLFLOW_EXPERIMENT_NAME
)


def main():
    """Main function to run the credit risk model pipeline."""
    parser = argparse.ArgumentParser(description='Credit Risk Model Pipeline')
    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'visualize', 'full', 'mlflow_best', 'compare_models', 'promote_model'],
                        default='full', help='Mode to run the pipeline')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the data file. If not provided, data will be fetched from OpenML.')
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory to save model artifacts')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained model for prediction or visualization')
    parser.add_argument('--input_data', type=str, default=None,
                        help='Path to the input data CSV file for prediction')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save predictions')
    parser.add_argument('--preprocessor_config', type=str, default=None,
                        help='Path to the preprocessor configuration JSON file')
    parser.add_argument('--history_path', type=str, default=None,
                        help='Path to the training history JSON file for visualization')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualizations')
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
    parser.add_argument('--track_with_mlflow', action='store_true',
                        help='Whether to track with MLFlow')
    parser.add_argument('--mlflow_metric', type=str, default='val_accuracy',
                        help='Metric to use for finding the best model in MLFlow')
    parser.add_argument('--register_model', action='store_true',
                        help='Whether to register the model to MLFlow registry')
    parser.add_argument('--model_stage', type=str, choices=['None', 'Staging', 'Production', 'Archived'],
                        default='Staging', help='Stage for the registered model')
    parser.add_argument('--model_version', type=str, default=None,
                        help='Version of the model to use from registry')
    parser.add_argument('--compare_runs_count', type=int, default=5,
                        help='Number of runs to compare')
    parser.add_argument('--model_name', type=str, default='credit_card_risk_model',
                        help='Name of the model in the registry')
    
    args = parser.parse_args()
    
    # Create timestamp for model directory if not provided
    if args.model_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                                    f'../../models/{timestamp}'))
    
    # Create model directory if it doesn't exist
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Create output directory for visualizations if not provided
    if args.output_dir is None:
        args.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 
                                                     f'../../{FIGURES_DIR}'))
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Resolve data path
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = os.path.join(os.path.dirname(__file__), '../../data/raw/credit-g.csv')
        if not os.path.exists(data_path):
            data_path = None
    
    # MLFlow best model mode
    if args.mode == 'mlflow_best':
        if not MLFLOW_TRACKING_ENABLED:
            print("MLFlow tracking is not enabled. Please enable it in the configuration.")
            return
        
        print("=== MLFlow Best Model Mode ===")
        
        # Get best run
        best_run = get_best_run(args.mlflow_metric, ascending=False)
        if best_run is None:
            print("No runs found in MLFlow.")
            return
        
        print(f"Best run found: {best_run['run_id']} with {args.mlflow_metric} = {best_run[f'metrics.{args.mlflow_metric}']}")
        
        # Get model artifacts
        model_uri = f"runs:/{best_run['run_id']}/best_tuned_model"
        if 'standard_model' in best_run['tags.model_type']:
            model_uri = f"runs:/{best_run['run_id']}/standard_model"
        
        # Load model
        model = mlflow.tensorflow.load_model(model_uri)
        
        # Save model
        model_path = os.path.join(args.model_dir, 'best_mlflow_model.keras')
        model.save(model_path)
        
        print(f"Best model saved to {model_path}")
        
        # Set model path for prediction or visualization
        args.model_path = model_path
    
    # Compare models mode
    if args.mode == 'compare_models':
        if not MLFLOW_TRACKING_ENABLED:
            print("MLFlow tracking is not enabled. Please enable it in the configuration.")
            return
        
        print("=== Compare Models Mode ===")
        
        # Get runs
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            print("No experiment found.")
            return
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{args.mlflow_metric} DESC"],
            max_results=args.compare_runs_count
        )
        
        if runs.empty:
            print("No runs found.")
            return
        
        # Compare runs
        run_ids = runs['run_id'].tolist()
        comparison = compare_runs(run_ids, metrics=[args.mlflow_metric, 'accuracy', 'f1_score', 'roc_auc'])
        
        # Print comparison
        print("\nModel Comparison:")
        print(comparison)
        
        # Save comparison to CSV
        comparison_path = os.path.join(args.model_dir, 'model_comparison.csv')
        comparison.to_csv(comparison_path, index=False)
        print(f"Comparison saved to {comparison_path}")
    
    # Promote model mode
    if args.mode == 'promote_model':
        if not MLFLOW_TRACKING_ENABLED:
            print("MLFlow tracking is not enabled. Please enable it in the configuration.")
            return
        
        print("=== Promote Model Mode ===")
        
        # Get model version
        if args.model_version is None:
            # Get best run
            best_run = get_best_run(args.mlflow_metric, ascending=False)
            if best_run is None:
                print("No runs found.")
                return
            
            run_id = best_run['run_id']
            
            # Register model if not already registered
            client = mlflow.tracking.MlflowClient()
            model_versions = client.search_model_versions(f"run_id='{run_id}'")
            
            if not model_versions:
                print(f"Registering model from run {run_id}...")
                model_version = register_model(
                    run_id=run_id,
                    model_name=args.model_name,
                    model_version_description=f"Best model with {args.mlflow_metric}={best_run[f'metrics.{args.mlflow_metric}']}",
                    stage=args.model_stage
                )
                version = model_version.version
            else:
                version = model_versions[0].version
        else:
            version = args.model_version
        
        # Promote model to specified stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=args.model_name,
            version=version,
            stage=args.model_stage
        )
        
        print(f"Model {args.model_name} version {version} promoted to {args.model_stage}")
    
    # Training mode
    if args.mode in ['train', 'full']:
        print("=== Training Mode ===")
        
        # Start MLFlow run if tracking is enabled
        if args.track_with_mlflow and MLFLOW_TRACKING_ENABLED:
            run_name = f"full_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            run = start_run(run_name=run_name, tags={"phase": "full_training"})
        
        try:
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
            
            with open(os.path.join(args.model_dir, os.path.basename(PREPROCESSOR_CONFIG_PATH)), 'w') as f:
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
            
            # Train model
            model = train_model(
                train_dataset, 
                val_dataset, 
                preprocessor, 
                args.model_dir, 
                epochs=args.epochs, 
                batch_size=args.batch_size,
                use_tuner=args.use_tuner,
                max_trials=args.max_trials,
                register_to_mlflow=args.register_model
            )
            
            # Evaluate model
            test_loss, test_acc = model.model.evaluate(test_dataset)
            print(f"Test loss: {test_loss} - Test accuracy: {test_acc}")
            
            # Save evaluation metrics
            with open(os.path.join(args.model_dir, os.path.basename(EVALUATION_METRICS_PATH)), 'w') as f:
                json.dump({'test_loss': float(test_loss), 'test_accuracy': float(test_acc)}, f)
            
            print(f"Model artifacts saved to {args.model_dir}")
            
            # Set model path for prediction or visualization
            if args.use_tuner:
                args.model_path = os.path.join(args.model_dir, os.path.basename(BEST_TUNED_MODEL_PATH))
            else:
                args.model_path = os.path.join(args.model_dir, os.path.basename(DEFAULT_MODEL_PATH))
            
            # Set preprocessor config path
            args.preprocessor_config = os.path.join(args.model_dir, os.path.basename(PREPROCESSOR_CONFIG_PATH))
            
            # Set history path
            args.history_path = os.path.join(args.model_dir, 'training_history.json')
        finally:
            # End MLFlow run if tracking is enabled
            if args.track_with_mlflow and MLFLOW_TRACKING_ENABLED:
                end_run()
    
    # Prediction mode
    if args.mode in ['predict', 'full']:
        if args.model_path or args.model_stage or args.model_version:
            print("=== Prediction Mode ===")
            
            # Start MLFlow run if tracking is enabled
            if args.track_with_mlflow and MLFLOW_TRACKING_ENABLED:
                run_name = f"prediction_{os.path.basename(args.model_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                run = start_run(run_name=run_name, tags={"phase": "prediction"})
            
            try:
                # Set default preprocessor config path if not provided
                if args.preprocessor_config is None:
                    if os.path.exists(PREPROCESSOR_CONFIG_PATH):
                        args.preprocessor_config = PREPROCESSOR_CONFIG_PATH
                    else:
                        args.preprocessor_config = os.path.join(os.path.dirname(args.model_path), 
                                                              os.path.basename(PREPROCESSOR_CONFIG_PATH))
                
                # Load model - either from file or from registry
                if args.model_path:
                    model = load_model(args.model_path)
                elif args.model_stage or args.model_version:
                    model = get_model_from_registry(
                        model_name=args.model_name,
                        stage=args.model_stage if args.model_stage != 'None' else None,
                        version=args.model_version
                    )
                    if model is None:
                        print("Failed to load model from registry.")
                        return
                
                # Load input data
                data = load_data(args.input_data)[0]  # Get only features
                
                # Load preprocessor configuration
                with open(args.preprocessor_config, 'r') as f:
                    config = json.load(f)
                
                # Initialize preprocessor
                preprocessor = CreditDataPreprocessor(
                    discrete_features=config['discrete_features'],
                    categorical_features=config['categorical_features'],
                    continuous_features=config['continuous_features']
                )
                
                # Make predictions
                predictions = predict(model, data, preprocessor)
                
                # Set output path if not provided
                if args.output_path is None:
                    args.output_path = os.path.join(args.model_dir, 'predictions.csv')
                
                # Save predictions
                results = pd.DataFrame({
                    'prediction': predictions.flatten(),
                    'credit_risk': np.where(predictions.flatten() > 0.5, 'good', 'bad')
                })
                
                # Combine with input data if needed
                output = pd.concat([data, results], axis=1)
                output.to_csv(args.output_path, index=False)
                
                # Log metrics to MLFlow if tracking is enabled and true labels are available
                if args.track_with_mlflow and MLFLOW_TRACKING_ENABLED and 'class' in data.columns:
                    y_true = data['class'].map({'good': 1, 'bad': 0}).values
                    y_pred = (predictions.flatten() > 0.5).astype(int)
                    log_model_metrics(y_true, y_pred, predictions.flatten())
                
                print(f"Predictions saved to {args.output_path}")
            finally:
                # End MLFlow run if tracking is enabled
                if args.track_with_mlflow and MLFLOW_TRACKING_ENABLED:
                    end_run()
        else:
            print("Skipping prediction mode. Missing required arguments.")
    
    # Visualization mode
    if args.mode in ['visualize', 'full']:
        if args.model_path and args.output_path:
            print("=== Visualization Mode ===")
            
            # Start MLFlow run if tracking is enabled
            if args.track_with_mlflow and MLFLOW_TRACKING_ENABLED:
                run_name = f"visualization_{os.path.basename(args.model_path)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                run = start_run(run_name=run_name, tags={"phase": "visualization"})
            
            try:
                # Load predictions
                predictions_df = pd.read_csv(args.output_path)
                
                # Extract true labels and predicted probabilities
                if 'class' in predictions_df.columns:
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
                    
                    # Log metrics to MLFlow if tracking is enabled
                    if args.track_with_mlflow and MLFLOW_TRACKING_ENABLED:
                        log_model_metrics(y_true, y_pred, y_pred_proba)
                
                # Plot training history if provided
                if args.history_path and os.path.exists(args.history_path):
                    with open(args.history_path, 'r') as f:
                        history = json.load(f)
                    
                    plot_training_history(
                        history, 
                        output_path=os.path.join(args.output_dir, os.path.basename(TRAINING_HISTORY_PATH))
                    )
                    
                    # Log training history to MLFlow if tracking is enabled
                    if args.track_with_mlflow and MLFLOW_TRACKING_ENABLED:
                        log_training_history(history)
                
                print(f"Visualizations saved to {args.output_dir}")
            finally:
                # End MLFlow run if tracking is enabled
                if args.track_with_mlflow and MLFLOW_TRACKING_ENABLED:
                    end_run()
        else:
            print("Skipping visualization mode. Missing required arguments.")


if __name__ == "__main__":
    main() 