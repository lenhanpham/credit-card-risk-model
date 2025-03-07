### Augment
import tensorflow as tf
!pip install silence_tensorflow
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from sklearn.datasets import fetch_openml
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, Concatenate, IntegerLookup, StringLookup
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
! pip install keras-tuner
import keras_tuner as kt # Import kerastuner


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
        # Implement build method to avoid warnings
        # No weights to initialize, but we can call this to mark the layer as built
        super().build(input_shape)

    def call(self, inputs):
        return tf.cast(self.encoder(inputs), tf.float32)

    def get_config(self):
        config = super().get_config()
        # Save the encoder configuration
        config.update({
            'encoder_config': tf.keras.layers.serialize(self.encoder),
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Deserialize the encoder
        encoder_config = config.pop('encoder_config')
        encoder = tf.keras.layers.deserialize(encoder_config)
        return cls(encoder, **config)



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




    
    # Modify _process_features to accept preprocessor as an argument
    def _process_features(self, continuous_inputs, discrete_inputs, categorical_inputs):  # Note the 'self'
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
            Flatten()(Embedding(
                input_dim=self.preprocessor.categorical_encoders[col].vocabulary_size(),
                output_dim=self.embedding_size
            )(self.preprocessor.categorical_encoders[col](categorical_inputs[col])))
            for col in self.preprocessor.categorical_features
        ]

        return Concatenate()(processed_continuous + processed_discrete + embedded_features)

    



def create_tf_datasets(X, y, train_size=0.7, val_size=0.15, batch_size=128, seed=None):
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


# Hyperparameter search space
def build_model(hp):
    # Instantiate the model
    model_instance = CreditRiskModel(preprocessor)

    continuous_inputs = {
        col: Input(shape=(1,), dtype=tf.float32, name=f"{col}_input")
        for col in preprocessor.continuous_features
    }
    discrete_inputs = {
        col: Input(shape=(1,), dtype=tf.int32, name=f"{col}_input")
        for col in preprocessor.discrete_features
    }
    categorical_inputs = {
        col: Input(shape=(1,), dtype=tf.string, name=f"{col}_input")
        for col in preprocessor.categorical_features
    }

    # Call the instance method
    processed_features = model_instance._process_features(
        continuous_inputs, discrete_inputs, categorical_inputs
    )

    # Expanded search space
    hp_units1 = hp.Int('units1', min_value=32, max_value=256, step=16)  # Larger range for units
    hp_units2 = hp.Int('units2', min_value=32, max_value=256, step=16)   # Larger range for units
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.05)  # Wider range for dropout
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')  # Wider range for learning rate

    x = Dense(hp_units1, activation='relu', kernel_initializer='he_normal')(processed_features)
    x = tf.keras.layers.Dropout(hp_dropout)(x)
    x = Dense(hp_units2, activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Dropout(hp_dropout)(x)
    output = Dense(1, activation='sigmoid')(x)

    model_inputs = list(continuous_inputs.values()) + list(discrete_inputs.values()) + list(categorical_inputs.values())
    model = Model(inputs=model_inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


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


tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective='val_accuracy',
    max_trials=200,  # Adjust the number of trials as needed
    executions_per_trial=1,
    directory='my_hyperparameter_tuning',
    project_name='credit_risk'
)


# Early stopping and learning rate reduction (remains unchanged, but monitoring 'val_accuracy' might be better for the objective)

callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'), # Changed to monitor val_accuracy
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=15, min_lr=1e-6), # Changed to monitor val_accuracy
    ModelCheckpoint('best_logistic_credit_model_tf.keras', monitor='val_accuracy', save_best_only=True) # Changed to monitor val_accuracy
]

credit_model.model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200,
    callbacks=callbacks
)

tuner.search_space_summary()
tuner.search(train_dataset, epochs=200, validation_data=val_dataset, callbacks=callbacks)

best_hp = tuner.get_best_hyperparameters(1)[0]
print(f"Best hyperparameters: {best_hp.values}")

best_model = tuner.hypermodel.build(best_hp)
best_model.fit(train_dataset, epochs=200, validation_data=val_dataset, callbacks=callbacks)
test_loss, test_acc = best_model.evaluate(test_dataset)
print(f"Test loss: {test_loss} - Test accuracy: {test_acc}")

best_model.save('best_model_with_tuner.keras')

