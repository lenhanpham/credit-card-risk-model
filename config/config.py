# Configuration module containing all parameters

# Data parameters
DATA_CONFIG = {
    "dataset_name": "credit-g",
    "dataset_version": 1,
    "train_size": 0.7,
    "val_size": 0.15,
    "batch_size": 128,
    "seed": 2025
}

# Feature definitions
FEATURES_CONFIG = {
    "discrete_features": ["installment_commitment", "residence_since", "num_dependents", "existing_credits"],
    "continuous_features": ["duration", "credit_amount"],
    "categorical_features": [
        "checking_status", "credit_history", "purpose", "savings_status", "employment",
        "personal_status", "other_parties", "property_magnitude", "other_payment_plans",
        "housing", "job", "own_telephone", "foreign_worker"
    ]
}

# Model hyperparameters
MODEL_CONFIG = {
    "embedding_size": 8,
    "learning_rate": 0.001,
    "epochs": 200,
    "dropout_min": 0.0,
    "dropout_max": 0.5,
    "units1_min": 32,
    "units1_max": 512,
    "units2_min": 32,
    "units2_max": 256,
    "learning_rate_min": 1e-5,
    "learning_rate_max": 1e-2
}

# Training parameters
TRAINING_CONFIG = {
    "max_trials": 10,
    "executions_per_trial": 1,
    "patience": 15,
    "min_lr": 1e-6,
    "reduce_lr_factor": 0.5
}

# Paths
PATHS_CONFIG = {
    "tuner_directory": "hyperparameter_tuning",
    "project_name": "credit_risk",
    "model_save_path": "models/best_model.keras",
    "best_model_checkpoint": "models/best_checkpoint.keras",
    "mlflow_tracking_uri": "file:./mlruns",
    "mlflow_experiment_name": "credit_risk_prediction"
}