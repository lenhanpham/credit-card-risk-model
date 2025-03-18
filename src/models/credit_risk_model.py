import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, Concatenate

@tf.keras.utils.register_keras_serializable(package="Custom", name="LogTransform")
class LogTransform(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.log1p(inputs)
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable(package="Custom", name="Standardize")
class Standardize(tf.keras.layers.Layer):
    def call(self, inputs):
        return (inputs - tf.reduce_mean(inputs)) / tf.math.reduce_std(inputs)
    def get_config(self):
        return super().get_config()

@tf.keras.utils.register_keras_serializable(package="Custom", name="DiscreteFeatureEncoder")
class DiscreteFeatureEncoder(tf.keras.layers.Layer):
    def __init__(self, encoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
    def build(self, input_shape):
        super().build(input_shape)
    def call(self, inputs):
        return tf.cast(self.encoder(inputs), tf.float32)
    def get_config(self):
        config = super().get_config()
        config.update({'encoder_config': tf.keras.layers.serialize(self.encoder)})
        return config
    @classmethod
    def from_config(cls, config):
        encoder_config = config.pop('encoder_config')
        encoder = tf.keras.layers.deserialize(encoder_config)
        return cls(encoder, **config)

class CreditRiskModel(tf.keras.Model):
    def __init__(self, preprocessor, embedding_size=8, **kwargs):
        super().__init__(**kwargs)
        self.preprocessor = preprocessor
        self.embedding_size = embedding_size
        self.model = self.build_model()

    def call(self, inputs):
        return self.model(inputs)

    def get_config(self):
        return {"embedding_size": self.embedding_size}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def save_model(self, path):
        self.model.save(path)

    def _process_features(self, continuous_inputs, discrete_inputs, categorical_inputs):
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

    def build_model(self):
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

        processed_features = self._process_features(continuous_inputs, discrete_inputs, categorical_inputs)
        x = Dense(128, activation='relu', kernel_initializer='he_normal')(processed_features)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        output = Dense(1, activation='sigmoid')(x)

        model_inputs = list(continuous_inputs.values()) + list(discrete_inputs.values()) + list(categorical_inputs.values())
        return tf.keras.Model(inputs=model_inputs, outputs=output)