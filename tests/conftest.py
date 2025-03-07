import pytest
import pandas as pd
import numpy as np
import tensorflow as tf
from credit_risk_model.src.data.make_dataset import CreditDataPreprocessor

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'duration': np.random.randint(1, 100, n_samples),
        'credit_amount': np.random.randint(1000, 10000, n_samples),
        'installment_commitment': np.random.randint(1, 5, n_samples),
        'residence_since': np.random.randint(1, 5, n_samples),
        'num_dependents': np.random.randint(1, 5, n_samples),
        'existing_credits': np.random.randint(1, 5, n_samples),
        'checking_status': np.random.choice(['A11', 'A12', 'A13', 'A14'], n_samples),
        'credit_history': np.random.choice(['A30', 'A31', 'A32', 'A33', 'A34'], n_samples),
        'purpose': np.random.choice(['A40', 'A41', 'A42', 'A43'], n_samples),
        'savings_status': np.random.choice(['A61', 'A62', 'A63', 'A64', 'A65'], n_samples),
        'employment': np.random.choice(['A71', 'A72', 'A73', 'A74', 'A75'], n_samples),
    }
    
    X = pd.DataFrame(data)
    y = np.random.choice([0, 1], n_samples)
    
    return X, y

@pytest.fixture
def preprocessor(sample_data):
    """Create preprocessor instance with sample data."""
    X, _ = sample_data
    
    discrete_features = ['installment_commitment', 'residence_since', 'num_dependents', 'existing_credits']
    categorical_features = X.select_dtypes(exclude='number').columns.tolist()
    continuous_features = ['duration', 'credit_amount']
    
    preprocessor = CreditDataPreprocessor(
        discrete_features=discrete_features,
        categorical_features=categorical_features,
        continuous_features=continuous_features
    )
    
    return preprocessor

@pytest.fixture
def tf_dataset(sample_data):
    """Create TensorFlow dataset for testing."""
    X, y = sample_data
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    return dataset.batch(32)

@pytest.fixture
def model_config():
    """Create model configuration for testing."""
    return {
        'batch_size': 32,
        'epochs': 2,
        'learning_rate': 0.001,
        'embedding_size': 8
    } 