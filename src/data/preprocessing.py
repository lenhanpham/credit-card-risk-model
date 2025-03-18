import tensorflow as tf
from sklearn.datasets import fetch_openml

class CreditDataPreprocessor:
    def __init__(self, discrete_features, categorical_features, continuous_features):
        self.discrete_features = discrete_features
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features

        self.ordinal_encoders = {
            col: tf.keras.layers.IntegerLookup(output_mode='int', num_oov_indices=1)
            for col in discrete_features
        }
        self.categorical_encoders = {
            col: tf.keras.layers.StringLookup(output_mode='int', num_oov_indices=1)
            for col in categorical_features
        }

    def adapt(self, dataset):
        for batch in dataset:
            features, _ = batch
            for col in self.discrete_features:
                self.ordinal_encoders[col].adapt(features[col])
            for col in self.categorical_features:
                self.categorical_encoders[col].adapt(features[col])

    def preprocess_batch(self, features, labels):
        inputs = {
            **{f"{col}_input": features[col] for col in self.continuous_features},
            **{f"{col}_input": tf.cast(features[col], tf.int32) for col in self.discrete_features},
            **{f"{col}_input": features[col] for col in self.categorical_features}
        }
        return inputs, labels

    def prepare_dataset(self, dataset):
        return dataset.map(self.preprocess_batch).cache()

def create_tf_datasets(X, y, train_size, val_size, batch_size, seed):
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

def load_credit_data():
    credit_data = fetch_openml(name='credit-g', version=1, as_frame=True)
    X = credit_data.data
    y = credit_data.target.map({'good': 1, 'bad': 0}).values
    return X, y