import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt
import mlflow
from mlflow.models import infer_signature
import mlflow.tensorflow
from datetime import datetime
from src.models.credit_risk_model import CreditRiskModel  # Added missing import

def build_tuner_model(hp, preprocessor, embedding_size):
    continuous_inputs = {
        col: tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name=f"{col}_input")
        for col in preprocessor.continuous_features
    }
    discrete_inputs = {
        col: tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name=f"{col}_input")
        for col in preprocessor.discrete_features
    }
    categorical_inputs = {
        col: tf.keras.layers.Input(shape=(1,), dtype=tf.string, name=f"{col}_input")
        for col in preprocessor.categorical_features
    }

    model_instance = CreditRiskModel(preprocessor, embedding_size)
    processed_features = model_instance._process_features(continuous_inputs, discrete_inputs, categorical_inputs)

    hp_units1 = hp.Int('units1', min_value=32, max_value=512, step=16)
    hp_units2 = hp.Int('units2', min_value=32, max_value=256, step=16)
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.05)
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')

    x = tf.keras.layers.Dense(hp_units1, activation='relu', kernel_initializer='he_normal')(processed_features)
    x = tf.keras.layers.Dropout(hp_dropout)(x)
    x = tf.keras.layers.Dense(hp_units2, activation='relu', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.Dropout(hp_dropout)(x)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model_inputs = list(continuous_inputs.values()) + list(discrete_inputs.values()) + list(categorical_inputs.values())
    model = tf.keras.Model(inputs=model_inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

class CreditRiskTrainer:
    def __init__(self, config, preprocessor):
        self.config = config
        self.preprocessor = preprocessor
        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=config["TRAINING_CONFIG"]["patience"],
                                           restore_best_weights=True,
                                           monitor='val_accuracy'),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                               factor=config["TRAINING_CONFIG"]["reduce_lr_factor"],
                                               patience=config["TRAINING_CONFIG"]["patience"],
                                               min_lr=config["TRAINING_CONFIG"]["min_lr"]),
            tf.keras.callbacks.ModelCheckpoint(config["PATHS_CONFIG"]["best_model_checkpoint"],
                                             monitor='val_accuracy',
                                             save_best_only=True),
            mlflow.tensorflow.MLflowCallback()
        ]
        self.tuner = kt.RandomSearch(
            lambda hp: build_tuner_model(hp, self.preprocessor, self.config["MODEL_CONFIG"]["embedding_size"]),
            objective='val_accuracy',
            max_trials=config["TRAINING_CONFIG"]["max_trials"],
            executions_per_trial=config["TRAINING_CONFIG"]["executions_per_trial"],
            directory=config["PATHS_CONFIG"]["tuner_directory"],
            project_name=config["PATHS_CONFIG"]["project_name"]
        )

    def train(self, train_dataset, val_dataset, test_dataset):
        mlflow.set_tracking_uri(self.config["PATHS_CONFIG"]["mlflow_tracking_uri"])
        mlflow.set_experiment(self.config["PATHS_CONFIG"]["mlflow_experiment_name"])

        with mlflow.start_run(run_name=f"credit_risk_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params(self.config["DATA_CONFIG"])
            mlflow.log_params(self.config["FEATURES_CONFIG"])
            
            # Perform hyperparameter search
            self.tuner.search(train_dataset, 
                            epochs=self.config["MODEL_CONFIG"]["epochs"],
                            validation_data=val_dataset,
                            callbacks=self.callbacks)

            # Get best hyperparameters and build final model
            best_hp = self.tuner.get_best_hyperparameters(1)[0]
            mlflow.log_params(best_hp.values)
            best_model = self.tuner.hypermodel.build(best_hp)

            # Train final model
            history = best_model.fit(
                train_dataset,
                epochs=self.config["MODEL_CONFIG"]["epochs"],
                validation_data=val_dataset,
                callbacks=self.callbacks
            )

            # Evaluate and log metrics
            test_loss, test_acc = best_model.evaluate(test_dataset)
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_accuracy", test_acc)

            # Save and log model
            best_model.save(self.config["PATHS_CONFIG"]["model_save_path"])
            mlflow.tensorflow.log_model(best_model, "model", registered_model_name="credit_risk_model")
            mlflow.log_artifact(self.config["PATHS_CONFIG"]["model_save_path"])

            return best_model, history