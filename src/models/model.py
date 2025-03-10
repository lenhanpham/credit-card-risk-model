import tensorflow as tf
import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.utils.preprocessing import LogTransform, Standardize, DiscreteFeatureEncoder
from config.model_config import (
    EMBEDDING_SIZE, DEFAULT_MODEL_PATH
)


class CreditRiskModel(tf.keras.Model):
    """
    Credit Risk prediction model that inherits from tf.keras.Model.
    
    This model processes continuous, discrete, and categorical features
    and combines them to predict credit risk.
    """
    def __init__(self, preprocessor, embedding_size=EMBEDDING_SIZE, **kwargs):
        super().__init__(**kwargs)  # Ensure proper initialization
        self.preprocessor = preprocessor
        self.embedding_size = embedding_size
        self.model = self.build_model()  # Store Keras model

    def build_model(self):
        """Builds and returns a Keras model"""
        continuous_inputs = {
            col: tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name=f"{col}_input") 
            for col in self.preprocessor.continuous_features
        }
        discrete_inputs = {
            col: tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name=f"{col}_input") 
            for col in self.preprocessor.discrete_features
        }
        categorical_inputs = {
            col: tf.keras.layers.Input(shape=(1,), dtype=tf.string, name=f"{col}_input") 
            for col in self.preprocessor.categorical_features
        }
        
        processed_features = self._process_features(
            continuous_inputs, discrete_inputs, categorical_inputs)
        
        x = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal')(processed_features)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        model_inputs = list(continuous_inputs.values()) + list(discrete_inputs.values()) + list(categorical_inputs.values())
        return tf.keras.models.Model(inputs=model_inputs, outputs=output)

    def call(self, inputs):
        """Forward pass for Keras"""
        return self.model(inputs)

    def get_config(self):
        """Required for serialization"""
        return {
            "embedding_size": self.embedding_size,
        }

    @classmethod
    def from_config(cls, config):
        """Load model from config"""
        return cls(**config)

    def save_model(self, path=DEFAULT_MODEL_PATH):
        """Save the model properly"""
        self.model.save(path)  # Save only the inner Keras model

    def _process_features(self, continuous_inputs, discrete_inputs, categorical_inputs):
        """
        Process and combine different types of features.
        
        Args:
            continuous_inputs: Dictionary of continuous feature inputs
            discrete_inputs: Dictionary of discrete feature inputs
            categorical_inputs: Dictionary of categorical feature inputs
            
        Returns:
            Concatenated processed features
        """
        log_transform = LogTransform()
        standardize = Standardize()

        processed_continuous = [
            standardize(log_transform(continuous_inputs[col]))
            for col in self.preprocessor.continuous_features
        ]

        processed_discrete = [
            DiscreteFeatureEncoder(self.preprocessor.ordinal_encoders[col])(discrete_inputs[col])
            for col in self.preprocessor.discrete_features
        ]

        embedded_features = [
            tf.keras.layers.Flatten()(tf.keras.layers.Embedding(
                input_dim=self.preprocessor.categorical_encoders[col].vocabulary_size(),
                output_dim=self.embedding_size
            )(self.preprocessor.categorical_encoders[col](categorical_inputs[col])))
            for col in self.preprocessor.categorical_features
        ]

        return tf.keras.layers.Concatenate()(processed_continuous + processed_discrete + embedded_features) 