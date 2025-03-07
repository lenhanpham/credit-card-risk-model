import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import (
    Dense, Input, Embedding, Flatten, Concatenate, 
    BatchNormalization, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

class CreditDataProcessor:
    """Efficient data processor for credit risk data"""
    
    def __init__(self, continuous_features, discrete_features, categorical_features):
        self.continuous_features = continuous_features
        self.discrete_features = discrete_features
        self.categorical_features = categorical_features
        
        # Initialize encoders
        self.ordinal_encoders = {
            col: tf.keras.layers.IntegerLookup(output_mode='int', num_oov_indices=1)
            for col in discrete_features
        }
        self.categorical_encoders = {
            col: tf.keras.layers.StringLookup(output_mode='int', num_oov_indices=1)
            for col in categorical_features
        }
        
        # Statistics for continuous features
        self.continuous_stats = {}

    def adapt(self, dataset):
        """Adapt encoders to the dataset"""
        # Compute statistics for continuous features
        continuous_data = []
        for batch in dataset:
            features, _ = batch
            continuous_data.append(
                tf.stack([features[col] for col in self.continuous_features], axis=1)
            )
        continuous_data = tf.concat(continuous_data, axis=0)
        
        self.continuous_stats['mean'] = tf.reduce_mean(continuous_data, axis=0)
        self.continuous_stats['std'] = tf.math.reduce_std(continuous_data, axis=0)
        
        # Adapt encoders
        for batch in dataset:
            features, _ = batch
            for col in self.discrete_features:
                self.ordinal_encoders[col].adapt(features[col])
            for col in self.categorical_features:
                self.categorical_encoders[col].adapt(features[col])

    def preprocess_batch(self, features, labels):
        """Preprocess a single batch of data"""
        # Process continuous features
        processed_continuous = []
        for i, col in enumerate(self.continuous_features):
            x = tf.cast(features[col], tf.float32)
            x = tf.math.log1p(x)
            x = (x - self.continuous_stats['mean'][i]) / self.continuous_stats['std'][i]
            processed_continuous.append(x)
        
        # Process discrete features
        processed_discrete = [
            tf.cast(self.ordinal_encoders[col](features[col]), tf.float32)
            for col in self.discrete_features
        ]
        
        # Process categorical features
        processed_categorical = [
            self.categorical_encoders[col](features[col])
            for col in self.categorical_features
        ]
        
        return {
            'continuous': tf.stack(processed_continuous, axis=1),
            'discrete': tf.stack(processed_discrete, axis=1),
            'categorical': processed_categorical
        }, labels

    def create_tf_datasets(self, X, y, train_size=0.8, val_size=0.1, batch_size=128, seed=None):
        """Create TensorFlow datasets with efficient splitting"""
        dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
        dataset = dataset.shuffle(buffer_size=len(X), seed=seed)
        
        n = len(X)
        train_size = int(n * train_size)
        val_size = int(n * val_size)
        
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size).take(val_size)
        test_dataset = dataset.skip(train_size + val_size)
        
        return (
            train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE),
            val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE),
            test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        )

class CreditRiskModel(tf.keras.Model):
    """Credit Risk Model with efficient architecture"""
    
    def __init__(self, feature_dims, embedding_dim=8):
        super().__init__()
        self.feature_dims = feature_dims
        self.embedding_dim = embedding_dim
        
        # Define layers
        self.embeddings = [
            Embedding(dim, embedding_dim) 
            for dim in feature_dims['categorical_dims']
        ]
        self.flatten = Flatten()
        self.concat = Concatenate()
        
        self.dense1 = Dense(128, activation='relu', kernel_initializer='he_normal')
        self.dropout1 = Dropout(0.1)
        self.dense2 = Dense(64, activation='relu', kernel_initializer='he_normal')
        self.dropout2 = Dropout(0.1)
        self.output_layer = Dense(1, activation='sigmoid')

    def call(self, inputs):
        # Process continuous and discrete features
        continuous = inputs['continuous']
        discrete = inputs['discrete']
        
        # Process categorical features
        categorical = inputs['categorical']
        embedded = [
            self.flatten(embedding(cat))
            for embedding, cat in zip(self.embeddings, categorical)
        ]
        
        # Combine all features
        x = self.concat([continuous, discrete] + embedded)
        
        # Dense layers
        x = self.dense1(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.output_layer(x)

    def get_config(self):
        return {
            'feature_dims': self.feature_dims,
            'embedding_dim': self.embedding_dim
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class CreditRiskTrainer:
    """Training pipeline for credit risk model"""
    
    def __init__(self, model, data_processor):
        self.model = model
        self.data_processor = data_processor

    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    def create_callbacks(self, model_path, patience=15):
        return [
            EarlyStopping(
                patience=patience,
                restore_best_weights=True,
                monitor='val_loss'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience,
                min_lr=1e-6
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True
            )
        ]

    def train(self, train_dataset, val_dataset, epochs=50, model_path='best_model.keras'):
        callbacks = self.create_callbacks(model_path)
        
        # Preprocess datasets
        train_dataset = train_dataset.map(
            self.data_processor.preprocess_batch
        ).cache()
        val_dataset = val_dataset.map(
            self.data_processor.preprocess_batch
        ).cache()
        
        # Train model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        return history

    def evaluate(self, test_dataset):
        test_dataset = test_dataset.map(
            self.data_processor.preprocess_batch
        ).cache()
        return self.model.evaluate(test_dataset)

    def save_model(self, path):
        self.model.save(path, save_format='tf')

    @staticmethod
    def load_model(path):
        return tf.keras.models.load_model(
            path,
            custom_objects={'CreditRiskModel': CreditRiskModel}
        )

# Example usage
if __name__ == "__main__":
    # Load data
    data = fetch_openml(name="credit-g", version=1, as_frame=True)
    X = data.data
    y = data.target
    
    # Handle categorical target variables by mapping them to integers
    if data.target.dtype == 'object' or data.target.dtype.name == 'category':
        # Map unique values to integers (e.g., 'good' -> 0, 'bad' -> 1)
        unique_values = data.target.unique()
        target_mapping = {val: i for i, val in enumerate(unique_values)}
        y = data.target.map(target_mapping)
    else:
        # If already numeric, just convert to int
        y = data.target.astype(int)
    
    # Analyze features
    continuous_cols = []
    discrete_cols = []
    categorical_cols = []
    
    # Known problematic columns to exclude
    exclude_columns = ['duration']  # Add other problematic columns here if needed
    
    for col in X.columns:
        # Skip columns that are known to cause issues
        if col in exclude_columns:
            continue
        
        if X[col].dtype == 'object':
            categorical_cols.append(col)
        elif X[col].nunique() < 10 and X[col].dtype != 'float':
            discrete_cols.append(col)
        else:
            continuous_cols.append(col)
    
    # Create data processor
    data_processor = CreditDataProcessor(continuous_cols, discrete_cols, categorical_cols)
    
    # Adapt data processor to the dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    data_processor.adapt(dataset)
    
    # Create feature dimensions
    feature_dims = {
        'continuous_dim': len(continuous_cols),
        'discrete_dim': len(discrete_cols),
        'categorical_dims': [len(data_processor.categorical_encoders[col].get_vocabulary()) for col in categorical_cols]
    }
    
    # Create model
    model = CreditRiskModel(feature_dims)
    
    # Create trainer
    trainer = CreditRiskTrainer(model, data_processor)
    
    # Compile model
    trainer.compile_model()
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = data_processor.create_tf_datasets(X, y)
    
    # Train model
    history = trainer.train(train_dataset, val_dataset, epochs=100)
    
    # Evaluate model
    metrics = trainer.evaluate(test_dataset)
    print(f"Model evaluation: {metrics}")
    
    # Save model
    trainer.save_model("credit_risk_model.keras")
    
    # Load model
    loaded_model = CreditRiskTrainer.load_model("credit_risk_model.keras")
    
    # Evaluate loaded model
    loaded_metrics = loaded_model.evaluate(test_dataset)
    print(f"Loaded model evaluation: {loaded_metrics}")
