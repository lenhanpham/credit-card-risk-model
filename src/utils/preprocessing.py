import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="Custom", name="LogTransform")
class LogTransform(tf.keras.layers.Layer):
    """
    Custom layer that applies log(x + 1) transformation to inputs.
    """
    def call(self, inputs):
        return tf.math.log1p(inputs)

    def get_config(self):  # Required for serialization
        return super().get_config()

@tf.keras.utils.register_keras_serializable(package="Custom", name="Standardize")
class Standardize(tf.keras.layers.Layer):
    """
    Custom layer that standardizes inputs by subtracting mean and dividing by standard deviation.
    """
    def call(self, inputs):
        return (inputs - tf.reduce_mean(inputs)) / tf.math.reduce_std(inputs)

    def get_config(self):  # Required for serialization
        return super().get_config()

@tf.keras.utils.register_keras_serializable(package="Custom", name="DiscreteFeatureEncoder")
class DiscreteFeatureEncoder(tf.keras.layers.Layer):
    """
    Custom layer that encodes discrete features using a provided encoder.
    """
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