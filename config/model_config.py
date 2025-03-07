"""
Configuration file for the credit risk model.

This file contains all parameters for data processing, model architecture,
training, and hyperparameter tuning.
"""

# Feature definitions
DISCRETE_FEATURES = ['installment_commitment', 'residence_since', 'num_dependents', 'existing_credits']
CATEGORICAL_FEATURES = None  # Will be populated dynamically from data
CONTINUOUS_FEATURES = ['duration', 'credit_amount']

# Training parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
MAX_EPOCHS = 200
EMBEDDING_SIZE = 8
RANDOM_SEED = 2025

# Dataset splitting
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15  # Calculated as 1 - TRAIN_SIZE - VAL_SIZE

# Hyperparameter tuning
HP_MAX_TRIALS = 200
HP_EXECUTIONS_PER_TRIAL = 1
HP_TUNING_DIR = 'models/hyperparameter_tuning'
HP_PROJECT_NAME = 'credit_risk'

# Model architecture search space
HP_UNITS1_MIN = 32
HP_UNITS1_MAX = 256
HP_UNITS1_STEP = 16

HP_UNITS2_MIN = 32
HP_UNITS2_MAX = 256
HP_UNITS2_STEP = 16

HP_DROPOUT_MIN = 0.0
HP_DROPOUT_MAX = 0.5
HP_DROPOUT_STEP = 0.05

HP_LR_MIN = 1e-5
HP_LR_MAX = 1e-2

# Callbacks
EARLY_STOPPING_PATIENCE = 15
LR_REDUCE_PATIENCE = 15
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_MIN = 1e-6
CHECKPOINT_PATH = 'models/best_credit_risk_model.keras'

# Model paths
DEFAULT_MODEL_PATH = 'models/credit_risk_model.keras'
BEST_TUNED_MODEL_PATH = 'models/best_tuned_credit_risk_model.keras'

# Preprocessing
PREPROCESSOR_CONFIG_PATH = 'models/preprocessor_config.json'

# Evaluation
EVALUATION_METRICS_PATH = 'models/evaluation_metrics.json'

# Visualization
FIGURES_DIR = 'reports/figures'
CONFUSION_MATRIX_PATH = 'reports/figures/confusion_matrix.png'
ROC_CURVE_PATH = 'reports/figures/roc_curve.png'
PRECISION_RECALL_CURVE_PATH = 'reports/figures/precision_recall_curve.png'
TRAINING_HISTORY_PATH = 'reports/figures/training_history.png'

# MLFlow configuration
MLFLOW_TRACKING_URI = "mlruns"  # Local directory for MLFlow tracking
MLFLOW_EXPERIMENT_NAME = "credit_risk_model"
MLFLOW_TRACKING_ENABLED = True

# MLFlow tags
MLFLOW_TAGS = {
    "project": "credit_risk_assessment",
    "framework": "tensorflow",
    "model_type": "neural_network"
} 