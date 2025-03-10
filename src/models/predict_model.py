import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import argparse
import json

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.data.make_dataset import CreditDataPreprocessor
from src.models.model import CreditRiskModel
from src.utils.mlflow_utils import (
    start_run, end_run, log_model_metrics
)
from config.model_config import (
    DEFAULT_MODEL_PATH, PREPROCESSOR_CONFIG_PATH,
    MLFLOW_TRACKING_ENABLED
)


def load_model(model_path=DEFAULT_MODEL_PATH):
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    return tf.keras.models.load_model(model_path)


def prepare_input_data(data, preprocessor):
    """
    Prepare input data for prediction.
    
    Args:
        data: DataFrame with input features
        preprocessor: Data preprocessor instance
        
    Returns:
        Processed TensorFlow dataset ready for prediction
    """
    # Convert to TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(data), np.zeros(len(data))))
    
    # Apply preprocessing
    processed_dataset = preprocessor.prepare_dataset(dataset)
    
    return processed_dataset.batch(len(data))


def predict(model, data, preprocessor):
    """
    Make predictions using the trained model.
    
    Args:
        model: Trained model
        data: DataFrame with input features
        preprocessor: Data preprocessor instance
        
    Returns:
        Numpy array of predictions
    """
    # Prepare input data
    input_data = prepare_input_data(data, preprocessor)
    
    # Make predictions
    predictions = model.predict(input_data)
    
    return predictions


def main():
    """Main function to make predictions."""
    parser = argparse.ArgumentParser(description='Make predictions with credit risk model')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to the trained model')
    parser.add_argument('--input_data', type=str, required=True,
                        help='Path to the input data CSV file')
    parser.add_argument('--output_path', type=str, default='predictions.csv',
                        help='Path to save predictions')
    parser.add_argument('--preprocessor_config', type=str, default=PREPROCESSOR_CONFIG_PATH,
                        help='Path to the preprocessor configuration JSON file')
    parser.add_argument('--track_predictions', action='store_true',
                        help='Whether to track predictions with MLFlow')
    
    args = parser.parse_args()
    
    # Start MLFlow run if tracking is enabled
    if args.track_predictions and MLFLOW_TRACKING_ENABLED:
        run_name = f"prediction_{os.path.basename(args.model_path)}_{os.path.basename(args.input_data)}"
        run = start_run(run_name=run_name, tags={"phase": "prediction"})
    
    try:
        # Load model
        model = load_model(args.model_path)
        
        # Load input data
        data = pd.read_csv(args.input_data)
        
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
        
        # Save predictions
        results = pd.DataFrame({
            'prediction': predictions.flatten(),
            'credit_risk': np.where(predictions.flatten() > 0.5, 'good', 'bad')
        })
        
        # Combine with input data if needed
        output = pd.concat([data, results], axis=1)
        output.to_csv(args.output_path, index=False)
        
        # Log metrics to MLFlow if tracking is enabled and true labels are available
        if args.track_predictions and MLFLOW_TRACKING_ENABLED and 'class' in data.columns:
            y_true = data['class'].map({'good': 1, 'bad': 0}).values
            y_pred = (predictions.flatten() > 0.5).astype(int)
            log_model_metrics(y_true, y_pred, predictions.flatten())
        
        print(f"Predictions saved to {args.output_path}")
    finally:
        # End MLFlow run if tracking is enabled
        if args.track_predictions and MLFLOW_TRACKING_ENABLED:
            end_run()


if __name__ == "__main__":
    main() 