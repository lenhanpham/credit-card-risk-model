import tensorflow as tf
from sklearn.datasets import fetch_openml
import pandas as pd
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from credit_risk_model.src.utils.preprocessing import LogTransform, Standardize, DiscreteFeatureEncoder
from credit_risk_model.config.model_config import (
    DISCRETE_FEATURES, CATEGORICAL_FEATURES, CONTINUOUS_FEATURES,
    BATCH_SIZE, RANDOM_SEED, TRAIN_SIZE, VAL_SIZE
)


class CreditDataPreprocessor:
    """
    Preprocessor for credit risk data that handles discrete, categorical, and continuous features.
    """
    def __init__(self, discrete_features, categorical_features, continuous_features):
        self.discrete_features = discrete_features
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        
        # Initialize encoders
        self.ordinal_encoders = {
            col: tf.keras.layers.IntegerLookup(output_mode='int', num_oov_indices=1) 
            for col in discrete_features
        }
        self.categorical_encoders = {
            col: tf.keras.layers.StringLookup(output_mode='int', num_oov_indices=1) 
            for col in categorical_features
        }
        
    def adapt(self, dataset):
        """Adapt all encoders to the data"""
        for batch in dataset:
            features, _ = batch
            for col in self.discrete_features:
                self.ordinal_encoders[col].adapt(features[col])
            for col in self.categorical_features:
                self.categorical_encoders[col].adapt(features[col])
    
    def preprocess_batch(self, features, labels):
        """Transform a batch of data"""
        inputs = {
            **{f"{col}_input": features[col] for col in self.continuous_features},
            **{f"{col}_input": tf.cast(features[col], tf.int32) for col in self.discrete_features},
            **{f"{col}_input": features[col] for col in self.categorical_features}
        }
        return inputs, labels
    
    def prepare_dataset(self, dataset):
        """Prepare a dataset for training"""
        return dataset.map(self.preprocess_batch).cache()


def load_data(data_path=None):
    """
    Load credit risk data either from a local file or from OpenML.
    
    Args:
        data_path: Path to the local CSV file. If None, data is fetched from OpenML.
        
    Returns:
        X: Features DataFrame
        y: Target Series
    """
    if data_path and os.path.exists(data_path):
        # Load from local file
        data = pd.read_csv(data_path)
        # Assuming the target column is named 'class'
        X = data.drop('class', axis=1)
        y = data['class'].map({'good': 1, 'bad': 0})
    else:
        # Load from OpenML
        credit_data = fetch_openml(name='credit-g', version=1, as_frame=True)
        X = credit_data.data
        y = credit_data.target.map({'good': 1, 'bad': 0}).values
    
    return X, y


def create_tf_datasets(X, y, train_size=TRAIN_SIZE, val_size=VAL_SIZE, batch_size=BATCH_SIZE, seed=RANDOM_SEED):
    """
    Create TensorFlow datasets for training, validation, and testing.
    
    Args:
        X: Features DataFrame
        y: Target Series
        train_size: Proportion of data for training
        val_size: Proportion of data for validation
        batch_size: Batch size for training
        seed: Random seed for reproducibility
        
    Returns:
        train_dataset: TensorFlow dataset for training
        val_dataset: TensorFlow dataset for validation
        test_dataset: TensorFlow dataset for testing
    """
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    dataset = dataset.shuffle(buffer_size=len(X), seed=seed)
    n = len(X)
    train_size_n = int(n * train_size)
    val_size_n = int(n * val_size)
    
    train_dataset = dataset.take(train_size_n)
    val_dataset = dataset.skip(train_size_n).take(val_size_n)
    test_dataset = dataset.skip(train_size_n + val_size_n)
    
    return (
        train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE),
        val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE),
        test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    )


if __name__ == "__main__":
    # Example usage
    data_path = os.path.join(os.path.dirname(__file__), '../../../data/raw/credit-g.csv')
    X, y = load_data(data_path)
    
    # Define feature columns if not already defined in config
    discrete_features = DISCRETE_FEATURES
    continuous_features = CONTINUOUS_FEATURES
    categorical_features = CATEGORICAL_FEATURES if CATEGORICAL_FEATURES is not None else X.select_dtypes(exclude='number').columns.tolist()
    
    # Create datasets
    train_dataset_raw, val_dataset_raw, test_dataset_raw = create_tf_datasets(X, y, seed=RANDOM_SEED)
    
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
    
    n = len(X)
    train_size_n = int(n * TRAIN_SIZE)
    val_size_n = int(n * VAL_SIZE)
    print(f"Created datasets: train={train_size_n} samples, val={val_size_n} samples, test={n - train_size_n - val_size_n} samples") 