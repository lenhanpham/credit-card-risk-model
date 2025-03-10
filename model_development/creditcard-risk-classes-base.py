### Augment class based code 
import tensorflow as tf
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import numpy as np
from sklearn.datasets import fetch_openml
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, Concatenate, IntegerLookup, StringLookup
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


@tf.keras.utils.register_keras_serializable(package="Custom", name="LogTransform")
class LogTransform(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.log1p(inputs)

    def get_config(self):  # Required for serialization
        return super().get_config()

@tf.keras.utils.register_keras_serializable(package="Custom", name="LogTransform")
class Standardize(tf.keras.layers.Layer):
    def call(self, inputs):
        return (inputs - tf.reduce_mean(inputs)) / tf.math.reduce_std(inputs)

    def get_config(self):  # Required for serialization
        return super().get_config()


class CreditDataPreprocessor:
    def __init__(self, discrete_features, categorical_features, continuous_features):
        self.discrete_features = discrete_features
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        
        # Initialize encoders
        self.ordinal_encoders = {
            col: IntegerLookup(output_mode='int', num_oov_indices=1) 
            for col in discrete_features
        }
        self.categorical_encoders = {
            col: StringLookup(output_mode='int', num_oov_indices=1) 
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

@tf.keras.utils.register_keras_serializable(package="Custom", name="DiscreteFeatureEncoder")
class DiscreteFeatureEncoder(tf.keras.layers.Layer):
    def __init__(self, encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
    
    def build(self, input_shape):
        # Mark the layer as built
        super().build(input_shape)    
    
    def call(self, inputs):
        return tf.cast(self.encoder(inputs), tf.float32)

    def get_config(self):
        # Serialize the encoder along with other configurations
        config = super().get_config()
        config.update({
            "encoder": self.encoder.get_config()  # Serialize the encoder
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize the encoder
        encoder_config = config.pop("encoder")
        encoder = IntegerLookup.from_config(encoder_config)  # Reconstruct the encoder
        return cls(encoder=encoder, **config)


class CreditRiskModel(tf.keras.Model):  # Inherit from tf.keras.Model
    def __init__(self, preprocessor, embedding_size=8, **kwargs):
        super().__init__(**kwargs)  # Ensure proper initialization
        self.preprocessor = preprocessor
        self.embedding_size = embedding_size
        self.model = self.build_model()  # Store Keras model

    def build_model(self):
        """Builds and returns a Keras model"""
        continuous_inputs = {
            col: Input(shape=(1,), dtype=tf.float32, name=f"{col}_input") 
            for col in self.preprocessor.continuous_features
        }
        discrete_inputs = {
            col: Input(shape=(1,), dtype=tf.int32, name=f"{col}_input") 
            for col in self.preprocessor.discrete_features
        }
        categorical_inputs = {
            col: Input(shape=(1,), dtype=tf.string, name=f"{col}_input") 
            for col in self.preprocessor.categorical_features
        }
        
        processed_features = self._process_features(
            continuous_inputs, discrete_inputs, categorical_inputs)
        
        x = Dense(128, activation='relu', kernel_initializer='he_normal')(processed_features)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        output = Dense(1, activation='sigmoid')(x)

        model_inputs = list(continuous_inputs.values()) + list(discrete_inputs.values()) + list(categorical_inputs.values())
        return Model(inputs=model_inputs, outputs=output)

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

    def save_model(self, path="logistic_credit_model_tf.keras"):
        """Save the model properly"""
        self.model.save(path)  # Save only the inner Keras model




    
    def _process_features(self, continuous_inputs, discrete_inputs, categorical_inputs):
        log_transform = LogTransform()
        standardize = Standardize()
    
        # Process continuous features
        processed_continuous = [
            standardize(log_transform(continuous_inputs[col]))
            for col in self.preprocessor.continuous_features
        ]
        
        processed_discrete = [
            DiscreteFeatureEncoder(self.preprocessor.ordinal_encoders[col])(discrete_inputs[col])
            for col in self.preprocessor.discrete_features
        ]

    
        # Process categorical features
        embedded_features = [
            Flatten()(Embedding(
                input_dim=self.preprocessor.categorical_encoders[col].vocabulary_size(),
                output_dim=self.embedding_size
            )(self.preprocessor.categorical_encoders[col](categorical_inputs[col])))
            for col in self.preprocessor.categorical_features
        ]
    
        return Concatenate()(processed_continuous + processed_discrete + embedded_features)

    



def create_tf_datasets(X, y, train_size=0.8, val_size=0.1, batch_size=128, seed=None):
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

# Load and prepare data
credit_data = fetch_openml(name='credit-g', version=1, as_frame=True)
X = credit_data.data
y = credit_data.target.map({'good': 1, 'bad': 0}).values

# Define feature columns
discrete_features = ['installment_commitment', 'residence_since', 'num_dependents', 'existing_credits']
categorical_features = X.select_dtypes(exclude='number').columns.tolist()
continuous_features = ['duration', 'credit_amount']

# Create datasets
seed = 2025
train_dataset_raw, val_dataset_raw, test_dataset_raw = create_tf_datasets(X, y, seed=seed)

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

# Create and compile model
credit_model = CreditRiskModel(preprocessor)  
credit_model.model.compile(optimizer=Adam(learning_rate=0.001),
                           loss='binary_crossentropy',
                           metrics=['accuracy'])


callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6),
    ModelCheckpoint('best_logistic_credit_model_tf.keras', monitor='val_loss', save_best_only=True)
]

credit_model.model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200,
    callbacks=callbacks
)

# Evaluate
test_loss, test_acc = credit_model.model.evaluate(test_dataset)
print(f"Test loss: {test_loss} - Test accuracy: {test_acc}")
credit_model.save_model()  # Now works correctly




