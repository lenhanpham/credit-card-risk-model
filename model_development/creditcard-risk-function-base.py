import tensorflow as tf
!pip install silence_tensorflow
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import numpy as np
from sklearn.datasets import fetch_openml
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, Concatenate, BatchNormalization, IntegerLookup, StringLookup
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Load dataset
credit_data = fetch_openml(name='credit-g', version=1, as_frame=True)
X = credit_data.data
y = credit_data.target.map({'good': 1, 'bad': 0}).values

# Define feature columns
discrete_features = ['installment_commitment', 'residence_since', 'num_dependents', 'existing_credits']
categorical_features = X.select_dtypes(exclude='number').columns.tolist()
continuous_features = ['duration', 'credit_amount']

# Create TensorFlow datasets with 80/10/10 split
def create_tf_datasets(X, y, train_size=0.8, val_size=0.1, batch_size=128, seed=None):
    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    dataset = dataset.shuffle(buffer_size=len(X), seed=seed)  # Configurable seed
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

# Try different seeds (None for unseeded, or specific values)
seed = 2025  # Change to None for randomness, or test 0, 1, 42, etc.
train_dataset_raw, val_dataset_raw, test_dataset_raw = create_tf_datasets(X, y, seed=seed)

# Adapt lookup layers
def adapt_preprocessing_layers(dataset):
    ordinal_encoders = {col: IntegerLookup(output_mode='int', num_oov_indices=1) for col in discrete_features}
    categorical_encoders = {col: StringLookup(output_mode='int', num_oov_indices=1) for col in categorical_features}

    for batch in dataset:
        features, _ = batch
        for col in discrete_features:
            ordinal_encoders[col].adapt(features[col])
        for col in categorical_features:
            categorical_encoders[col].adapt(features[col])

    return ordinal_encoders, categorical_encoders

ordinal_encoders, categorical_encoders = adapt_preprocessing_layers(train_dataset_raw)

def log1p_with_shape(x):
    return tf.math.log1p(x)

def cast_to_float_with_shape(x):
    return tf.cast(x, tf.float32)


# Build model with preprocessing
def build_preprocessing_model():
    continuous_inputs = {col: Input(shape=(1,), dtype=tf.float32, name=f"{col}_input") for col in continuous_features}
    discrete_inputs = {col: Input(shape=(1,), dtype=tf.int32, name=f"{col}_input") for col in discrete_features}
    categorical_inputs = {col: Input(shape=(1,), dtype=tf.string, name=f"{col}_input") for col in categorical_features}


    processed_continuous = [
        tf.keras.layers.Lambda(
            lambda x: (x - tf.reduce_mean(x)) / tf.math.reduce_std(x),
            output_shape=(1,),
            name=f'standardize_lambda_{col}'
        )(
            tf.keras.layers.Lambda(
                lambda x: log1p_with_shape(x),
                output_shape=(1,),
                name=f'log1p_lambda_{col}'
            )
        (continuous_inputs[col])) for col in continuous_features
    ]


    #processed_continuous = [
    #    BatchNormalization(momentum=0.1, epsilon=1e-5)(
    #        tf.keras.layers.Lambda(
    #            log1p_with_shape,
    #            output_shape=(1,),
    #            name=f'log1p_lambda_{col}'
    #        )(continuous_inputs[col])
    #    ) for col in continuous_features
    #]


    processed_discrete = [
        tf.keras.layers.Lambda(
            lambda x: cast_to_float_with_shape(ordinal_encoders[col](x)),
            output_shape=(1,),
            name=f'cast_lambda_{col}'
        )(discrete_inputs[col])
        for col in discrete_features
    ]

    embedding_size = 8
    embedded_features = [
        Flatten()(Embedding(input_dim=categorical_encoders[col].vocabulary_size(), output_dim=embedding_size)(
            categorical_encoders[col](categorical_inputs[col])
        )) for col in categorical_features
    ]

    all_features = Concatenate()(processed_continuous + processed_discrete + embedded_features)
    return continuous_inputs, discrete_inputs, categorical_inputs, all_features

# Build full model
continuous_inputs, discrete_inputs, categorical_inputs, processed_features = build_preprocessing_model()
x = Dense(128, activation='relu', kernel_initializer='he_normal')(processed_features)
x = tf.keras.layers.Dropout(0.1)(x)
x = Dense(64, activation='relu', kernel_initializer='he_normal')(x)
x = tf.keras.layers.Dropout(0.1)(x)
output = Dense(1, activation='sigmoid')(x)

model_inputs = list(continuous_inputs.values()) + list(discrete_inputs.values()) + list(categorical_inputs.values())
model = Model(inputs=model_inputs, outputs=output)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
#model.summary()

# Preprocess datasets
def preprocess_batch(features, labels):
    inputs = {
        **{f"{col}_input": features[col] for col in continuous_features},
        **{f"{col}_input": tf.cast(features[col], tf.int32) for col in discrete_features},
        **{f"{col}_input": features[col] for col in categorical_features}
    }
    return inputs, labels

train_dataset = train_dataset_raw.map(preprocess_batch).cache()
val_dataset = val_dataset_raw.map(preprocess_batch).cache()
test_dataset = test_dataset_raw.map(preprocess_batch).cache()

# Callbacks
callbacks = [EarlyStopping(patience=15,
                           restore_best_weights=True,
                           monitor='val_loss'),
            ReduceLROnPlateau(monitor='val_loss',
                              factor=0.5,
                              patience=15,
                              min_lr=1e-6),
            ModelCheckpoint('best_logistic_credit_model_tf.keras',
                            monitor='val_loss',
                            save_best_only=True)
            ]

# Train
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=callbacks
)

# Evaluate
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test loss: {test_loss} - Test accuracy: {test_acc}")

# Save model
custom_objects = {
    'log1p_with_shape': log1p_with_shape,
    'cast_to_float_with_shape': cast_to_float_with_shape
}

model.save("logistic_credit_model_tf.keras")