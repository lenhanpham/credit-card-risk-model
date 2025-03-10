import unittest
import os
import sys
import tensorflow as tf
import numpy as np
import pytest

# Add the project root directory to the Python path
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.make_dataset import CreditDataPreprocessor
from src.models.model import CreditRiskModel


class TestCreditRiskModel(unittest.TestCase):
    """Tests for the Credit Risk Model."""

    def setUp(self):
        """Set up test fixtures."""
        # Define simple test data
        self.discrete_features = ['feature1']
        self.categorical_features = ['feature2']
        self.continuous_features = ['feature3']
        
        # Create preprocessor
        self.preprocessor = CreditDataPreprocessor(
            discrete_features=self.discrete_features,
            categorical_features=self.categorical_features,
            continuous_features=self.continuous_features
        )
        
        # Create model
        self.model = CreditRiskModel(self.preprocessor)

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model, CreditRiskModel)
        self.assertEqual(self.model.embedding_size, 8)  # Default value

    def test_model_build(self):
        """Test that the model builds correctly."""
        # Check that the model has been built
        self.assertIsNotNone(self.model.model)
        
        # Check that the model has the correct input and output shapes
        expected_inputs = len(self.discrete_features) + len(self.categorical_features) + len(self.continuous_features)
        self.assertEqual(len(self.model.model.inputs), expected_inputs)
        self.assertEqual(self.model.model.outputs[0].shape[-1], 1)  # Binary output

    def test_process_features(self):
        """Test the feature processing method."""
        # Create dummy inputs
        continuous_inputs = {
            col: tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name=f"{col}_input") 
            for col in self.continuous_features
        }
        discrete_inputs = {
            col: tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name=f"{col}_input") 
            for col in self.discrete_features
        }
        categorical_inputs = {
            col: tf.keras.layers.Input(shape=(1,), dtype=tf.string, name=f"{col}_input") 
            for col in self.categorical_features
        }
        
        # Process features
        processed_features = self.model._process_features(
            continuous_inputs, discrete_inputs, categorical_inputs
        )
        
        # Check that the output is a tensor
        self.assertIsInstance(processed_features, tf.Tensor)


def test_model_initialization(preprocessor):
    """Test model initialization."""
    model = CreditRiskModel(preprocessor)
    assert isinstance(model.model, tf.keras.Model)
    assert model.embedding_size == 8  # Default value

def test_model_build(preprocessor, tf_dataset):
    """Test model building."""
    model = CreditRiskModel(preprocessor)
    
    # Check model structure
    assert len(model.model.layers) > 0
    
    # Check input layers
    input_layers = [layer for layer in model.model.layers if isinstance(layer, tf.keras.layers.InputLayer)]
    expected_inputs = (
        len(preprocessor.continuous_features) +
        len(preprocessor.discrete_features) +
        len(preprocessor.categorical_features)
    )
    assert len(input_layers) == expected_inputs

def test_model_compile_and_fit(preprocessor, tf_dataset):
    """Test model compilation and training."""
    model = CreditRiskModel(preprocessor)
    model.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Adapt preprocessor
    preprocessor.adapt(tf_dataset)
    processed_dataset = preprocessor.prepare_dataset(tf_dataset)
    
    # Train for 1 epoch
    history = model.model.fit(processed_dataset, epochs=1, verbose=0)
    
    # Check training history
    assert 'loss' in history.history
    assert 'accuracy' in history.history

def test_model_predict(preprocessor, tf_dataset):
    """Test model predictions."""
    model = CreditRiskModel(preprocessor)
    model.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Adapt preprocessor and prepare dataset
    preprocessor.adapt(tf_dataset)
    processed_dataset = preprocessor.prepare_dataset(tf_dataset)
    
    # Make predictions
    predictions = model.model.predict(processed_dataset)
    
    # Check predictions shape and values
    assert len(predictions.shape) == 2
    assert predictions.shape[1] == 1
    assert (predictions >= 0).all() and (predictions <= 1).all()

def test_model_save_and_load(preprocessor, tf_dataset, tmp_path):
    """Test model saving and loading."""
    model = CreditRiskModel(preprocessor)
    model.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Save model
    save_path = tmp_path / "test_model.keras"
    model.save_model(save_path)
    
    # Load model
    loaded_model = tf.keras.models.load_model(save_path)
    
    # Check model structure is preserved
    assert len(model.model.layers) == len(loaded_model.layers)
    
    # Check predictions are the same
    preprocessor.adapt(tf_dataset)
    processed_dataset = preprocessor.prepare_dataset(tf_dataset)
    
    original_predictions = model.model.predict(processed_dataset)
    loaded_predictions = loaded_model.predict(processed_dataset)
    
    assert tf.reduce_all(tf.equal(original_predictions, loaded_predictions))

def test_model_custom_layers(preprocessor):
    """Test custom layer functionality."""
    model = CreditRiskModel(preprocessor)
    
    # Check for custom layers
    log_transform_layers = [layer for layer in model.model.layers 
                          if layer.__class__.__name__ == 'LogTransform']
    standardize_layers = [layer for layer in model.model.layers 
                         if layer.__class__.__name__ == 'Standardize']
    
    assert len(log_transform_layers) > 0
    assert len(standardize_layers) > 0


if __name__ == '__main__':
    unittest.main() 