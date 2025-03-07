import pytest
import tensorflow as tf
import numpy as np
import pandas as pd

def test_preprocessor_initialization(preprocessor):
    """Test preprocessor initialization."""
    assert preprocessor.discrete_features == ['installment_commitment', 'residence_since', 'num_dependents', 'existing_credits']
    assert preprocessor.continuous_features == ['duration', 'credit_amount']
    assert len(preprocessor.categorical_features) > 0

def test_preprocessor_adapt(preprocessor, tf_dataset):
    """Test preprocessor adaptation to data."""
    preprocessor.adapt(tf_dataset)
    
    # Check that encoders are adapted
    for col in preprocessor.discrete_features:
        assert preprocessor.ordinal_encoders[col].is_adapted
    
    for col in preprocessor.categorical_features:
        assert preprocessor.categorical_encoders[col].is_adapted

def test_preprocessor_prepare_dataset(preprocessor, tf_dataset):
    """Test dataset preparation."""
    preprocessor.adapt(tf_dataset)
    processed_dataset = preprocessor.prepare_dataset(tf_dataset)
    
    # Check that the dataset has the correct structure
    for batch in processed_dataset.take(1):
        features, labels = batch
        
        # Check continuous features
        for feature in preprocessor.continuous_features:
            assert f"{feature}_input" in features
            assert features[f"{feature}_input"].dtype == tf.float32
        
        # Check discrete features
        for feature in preprocessor.discrete_features:
            assert f"{feature}_input" in features
            assert features[f"{feature}_input"].dtype == tf.int32
        
        # Check categorical features
        for feature in preprocessor.categorical_features:
            assert f"{feature}_input" in features
            assert features[f"{feature}_input"].dtype == tf.string
        
        # Check labels
        assert labels.dtype == tf.float32 or labels.dtype == tf.int32

def test_preprocessor_batch_processing(preprocessor, sample_data):
    """Test batch processing functionality."""
    X, y = sample_data
    batch_size = 32
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y)).batch(batch_size)
    
    preprocessor.adapt(dataset)
    processed_dataset = preprocessor.prepare_dataset(dataset)
    
    # Check batch size is maintained
    for batch in processed_dataset.take(1):
        features, labels = batch
        assert labels.shape[0] <= batch_size  # Less than or equal because last batch might be smaller

def test_preprocessor_feature_types(preprocessor, tf_dataset):
    """Test that feature types are correctly handled."""
    preprocessor.adapt(tf_dataset)
    processed_dataset = preprocessor.prepare_dataset(tf_dataset)
    
    for batch in processed_dataset.take(1):
        features, _ = batch
        
        # Test continuous features are float32
        for feature in preprocessor.continuous_features:
            assert features[f"{feature}_input"].dtype == tf.float32
        
        # Test discrete features are int32
        for feature in preprocessor.discrete_features:
            assert features[f"{feature}_input"].dtype == tf.int32
        
        # Test categorical features are string
        for feature in preprocessor.categorical_features:
            assert features[f"{feature}_input"].dtype == tf.string

def test_preprocessor_missing_values(preprocessor):
    """Test handling of missing values."""
    # Create data with missing values
    data = {
        'duration': [1, None, 3],
        'credit_amount': [1000, 2000, None],
        'installment_commitment': [1, None, 3],
        'checking_status': ['A11', None, 'A13']
    }
    X = pd.DataFrame(data)
    y = np.array([0, 1, 0])
    
    # This should raise an error due to missing values
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y)).batch(1)
    with pytest.raises(Exception):
        preprocessor.adapt(dataset) 