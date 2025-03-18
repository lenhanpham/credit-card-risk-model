# Credit Risk Prediction

This project implements a credit risk prediction model using TensorFlow, with hyperparameter tuning via Keras Tuner and experiment tracking using MLflow. The codebase is structured following a modular, cookiecutter-like template for better organization and maintainability.

## Project Overview

The model is trained on the credit-g dataset from OpenML to predict whether a credit applicant is a good or bad risk. It uses a combination of continuous, discrete, and categorical features, processed through custom layers and embeddings, followed by a neural network with hyperparameter-tuned architecture.

## Project Structure

```
credit_risk_prediction/
├── .github/
│   └── workflows/
│       └── model_training.yml  # GitHub Actions workflow for CI/CD
├── config/
│   └── config.py               # Configuration parameters
├── src/
│   ├── __init__.py
│   ├── model_development/
│   │   ├── __init__.py
│   │   ├── development.ipynb
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py      # Data loading and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   └── credit_risk_model.py  # Model definition and custom layers
│   ├── utils/
│   │   ├── __init__.py
│   │   └── visualization.py  # Evaluation metrics and plots
│   └── training/
│       ├── __init__.py
│       └── trainer.py       # Training logic and hyperparameter tuning
├── tests/
│   └── test_preprocessing.py  # Unit tests
├── main.py                  # Main script to run the pipeline
├── requirements.txt         # Project dependencies
└── README.md                # This file
```

## Features

- **Data Preprocessing**: Handles continuous, discrete, and categorical features with custom TensorFlow layers.
- **Model Architecture**: Neural network with embeddings for categorical features and dense layers with dropout.
- **Hyperparameter Tuning**: Uses Keras Tuner for optimizing model architecture and training parameters.
- **Experiment Tracking**: Integrates MLflow for logging parameters, metrics, and models.
- **Evaluation**: Generates ROC curves, Precision-Recall curves, confusion matrices, and classification reports.
- **CI/CD**: Automated linting, testing, and execution via GitHub Actions.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- GitHub account (for CI/CD)

## Installation

1. **Clone the Repository**:

```
git clone <repository-url>
cd credit_risk_prediction
```

2. **Create a Virtual Environment** (optional but recommended):

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**:

```
pip install -r requirements.txt
```

## Usage

1. **Run the Training Pipeline**: Execute the main script to train the model, tune hyperparameters, and evaluate performance:

```
python main.py
```

2. **Output**:

- The trained model is saved to models/best_model.keras.
- Hyperparameter tuning results are stored in hyperparameter_tuning/credit_risk/.
- MLflow experiment logs are saved to mlruns/.
- Evaluation plots (ROC, PR curves, confusion matrix) and classification report are displayed and printed.

3. **Configuration**: Modify parameters in config/config.py to adjust:

- Dataset splits (DATA_CONFIG)
- Feature definitions (FEATURES_CONFIG)
- Model hyperparameters (MODEL_CONFIG)
- Training settings (TRAINING_CONFIG)
- File paths (PATHS_CONFIG)

## CI/CD with GitHub Actions

This project uses GitHub Actions for continuous integration:

- **Trigger**: Runs on every push or pull request to the main branch.
- Jobs:
  - Lints the code with flake8.
  - Runs unit tests with pytest.
  - Executes the main script with limited epochs (2) for validation.
- **Workflow File**: Located at .github/workflows/ci.yml.

## Configuration Details

All parameters are defined in config/config.py. Key sections include:

- **DATA_CONFIG**: Dataset name, version, split sizes, batch size, and random seed.
- **FEATURES_CONFIG**: Lists of discrete, continuous, and categorical features.
- **MODEL_CONFIG**: Embedding size, learning rate, epochs, and hyperparameter search ranges.
- **TRAINING_CONFIG**: Tuning trials, early stopping patience, and learning rate reduction settings.
- **PATHS_CONFIG**: Directory paths for model saving and MLflow tracking.

## Dependencies

Listed in requirements.txt:

- tensorflow>=2.10
- silence_tensorflow
- scikit-learn
- keras-tuner
- mlflow
- matplotlib
- seaborn
- numpy
- flake8 (for linting)
- pytest (for testing)

## Project Modules

- **src/data/preprocessing.py**: Loads the credit dataset and preprocesses it into TensorFlow datasets.
- **src/models/credit_risk_model.py**: Defines the CreditRiskModel class and custom layers.
- **src/training/trainer.py**: Implements the CreditRiskTrainer class for model training and tuning.
- **src/utils/visualization.py**: Provides functions to plot evaluation metrics.
- **main.py**: Orchestrates the full pipeline from data loading to evaluation.
- **tests/test_preprocessing.py**: Basic unit tests for data preprocessing.

## Extending the Project

- **Add New Features**: Update FEATURES_CONFIG and ensure dataset compatibility.
- **Modify Model**: Adjust architecture in credit_risk_model.py or trainer.py.
- **Custom Evaluation**: Extend visualization.py.
- **Add Tests**: Expand tests/ with more test cases.

## Troubleshooting

- **Missing Dependencies**: Ensure all packages in requirements.txt are installed.
- **TensorFlow Errors**: Verify TensorFlow version compatibility (2.10+).
- **MLflow Issues**: Check write permissions for mlruns/.
- **CI Failures**: Review GitHub Actions logs for specific errors.

## License

This project is unlicensed—feel free to use and modify it as needed.

## Contact

For questions or contributions, please open an issue or pull request on the repository.