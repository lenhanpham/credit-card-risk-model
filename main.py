from config.config import DATA_CONFIG, FEATURES_CONFIG, MODEL_CONFIG, TRAINING_CONFIG, PATHS_CONFIG
from src.data.preprocessing import CreditDataPreprocessor, create_tf_datasets, load_credit_data
from src.training.trainer import CreditRiskTrainer
from src.utils.visualization import plot_evaluation_metrics
import numpy as np
import os

def main():
    # Load configuration
    config = {
        "DATA_CONFIG": DATA_CONFIG,
        "FEATURES_CONFIG": FEATURES_CONFIG,
        "MODEL_CONFIG": MODEL_CONFIG,
        "TRAINING_CONFIG": TRAINING_CONFIG,
        "PATHS_CONFIG": PATHS_CONFIG
    }

    # Override epochs if set in environment (for CI)
    epochs = int(os.getenv("EPOCHS_OVERRIDE", config["MODEL_CONFIG"]["epochs"]))
    config["MODEL_CONFIG"]["epochs"] = epochs
    
    # Load and prepare data
    X, y = load_credit_data()
    train_dataset_raw, val_dataset_raw, test_dataset_raw = create_tf_datasets(
        X, y,
        config["DATA_CONFIG"]["train_size"],
        config["DATA_CONFIG"]["val_size"],
        config["DATA_CONFIG"]["batch_size"],
        config["DATA_CONFIG"]["seed"]
    )

    # Initialize and adapt preprocessor
    preprocessor = CreditDataPreprocessor(
        config["FEATURES_CONFIG"]["discrete_features"],
        config["FEATURES_CONFIG"]["categorical_features"],
        config["FEATURES_CONFIG"]["continuous_features"]
    )
    preprocessor.adapt(train_dataset_raw)

    # Prepare datasets
    train_dataset = preprocessor.prepare_dataset(train_dataset_raw)
    val_dataset = preprocessor.prepare_dataset(val_dataset_raw)
    test_dataset = preprocessor.prepare_dataset(test_dataset_raw)

    # Train model
    trainer = CreditRiskTrainer(config, preprocessor)
    best_model, history = trainer.train(train_dataset, val_dataset, test_dataset)

    # Evaluate model
    y_true = []
    y_pred_prob = []
    for batch in test_dataset:
        X_test, y_batch = batch
        y_pred_batch = best_model.predict(X_test)
        y_true.extend(y_batch.numpy())
        y_pred_prob.extend(y_pred_batch)

    y_pred = (np.array(y_pred_prob) > 0.5).astype(int).flatten()
    os.makedirs(config["PATHS_CONFIG"]["metrics_directory"], exist_ok=True)
    plot_evaluation_metrics(config,y_true, y_pred_prob, y_pred)

if __name__ == "__main__":
    main()