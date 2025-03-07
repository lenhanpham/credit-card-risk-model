# Credit Risk Model

This project implements a machine learning model for credit risk assessment using TensorFlow. The model predicts whether a credit applicant is a good or bad credit risk based on various features. The project is developed for large scale dataset with very high accuracy. The project uses MLFlow for experiment tracking and model management, and includes a robust CI/CD pipeline for automated training, testing, and deployment.

The project is underdevelopment and will be updated regularly. Since the project is developed using Tensorflow, it is better to train the model using GPU. Errors might be still in the code. 

## Project Structure

The project follows the Cookiecutter Data Science template:

```
credit_risk_model/
│
├── config/               # Configuration files
│   ├── model_config.py   # Model configuration parameters
│
├── data/
│   ├── raw/                # Original data
│   ├── processed/          # Processed data
│
├── model-development/      # Jupyter notebooks for exploration and analysis
│
├── src/                    # Source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── make_dataset.py # Scripts to load and preprocess data
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py        # Model definition
│   │   ├── train_model.py  # Scripts to train the model
│   │   ├── predict_model.py# Scripts to make predictions
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── visualize.py    # Scripts for visualizations
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── preprocessing.py# Utility functions for preprocessing
│   │   ├── mlflow_utils.py # MLFlow tracking utilities
│   │
│   ├── main.py             # Main script to run the project
│
├── tests/                  # Unit tests
│
├── models/                 # Saved models
│
├── reports/                # Generated reports
│   ├── figures/            # Generated graphics and figures
│
├── mlruns/                 # MLFlow tracking directory
│
├── environment.yml         # Conda environment configuration
│
├── requirements.txt        # Python dependencies
│
├── setup.py                # Setup script for package installation
│
└── README.md               # Project documentation
```

## Features

- Data preprocessing for categorical, discrete, and continuous features
- TensorFlow-based neural network model for credit risk prediction
- Hyperparameter tuning with Keras Tuner
- Model evaluation and visualization
- Command-line interface for training, prediction, and visualization
- Centralized configuration management
- Experiment tracking with MLFlow

## Installation and Dependencies

This project uses multiple dependency management files for different purposes. Choose the appropriate installation method based on your needs:

### 1. Production Installation
For production deployment, use `requirements.txt`:
```bash
pip install -r requirements.txt
```
This installs:
- Core ML dependencies
- API and serving components
- Monitoring tools
- Security features

### 2. Development Installation
For development work, use `requirements-dev.txt`:
```bash
pip install -r requirements-dev.txt
```
This includes everything in production plus:
- Testing tools (pytest, coverage)
- Code quality tools (black, flake8)
- Debugging tools
- Documentation tools
- Type checking support

### 3. Conda Environment (Recommended for GPU Support)
For conda users or if you need GPU support:
```bash
conda env create -f environment.yml
conda activate credit_risk_model
```
This provides:
- System-level dependencies (CUDA, cuDNN)
- Core scientific packages
- Development tools
- GPU support

### 4. Custom Installation
You can install specific features using pip extras:
```bash
# Basic installation
pip install -e .

# Install with API support
pip install -e .[api]

# Install with visualization support
pip install -e .[viz]

# Install with development tools
pip install -e .[dev]

# Install with documentation tools
pip install -e .[docs]

# Install all features
pip install -e .[api,viz,dev,docs]
```

### Available Extras
- **api**: Flask API, monitoring, and serving tools
- **viz**: Visualization libraries (matplotlib, seaborn, plotly)
- **dev**: Development tools (pytest, black, flake8)
- **docs**: Documentation tools (sphinx)

### Dependency Files Overview

1. **setup.py**:
   - Core package definition
   - Base dependencies
   - Optional feature groups (extras)

2. **requirements.txt**:
   - Production dependencies
   - API and serving requirements
   - Monitoring tools

3. **requirements-dev.txt**:
   - Development tools
   - Testing utilities
   - Documentation generators
   - Type checking

4. **environment.yml**:
   - Conda environment setup
   - GPU support
   - System-level dependencies

### Version Compatibility

- Python: >= 3.9
- TensorFlow: 2.15.0
- CUDA: 11.8 (for GPU support)
- cuDNN: 8.9.2 (for GPU support)

### GPU Support

To enable GPU support:

1. Using conda (recommended):
```bash
conda env create -f environment.yml
```

2. Manual installation:
```bash
# Uncomment GPU packages in requirements.txt
pip install -r requirements.txt
```

Required NVIDIA components:
- CUDA Toolkit 11.8
- cuDNN 8.9.2
- GPU drivers (latest version recommended)

## Usage

### Training a Model with MLFlow Tracking

```bash
# Train with MLFlow tracking
python src/main.py --mode train --data_path data/raw/credit-g.csv --track_with_mlflow

# Train with hyperparameter tuning and MLFlow tracking
python src/main.py --mode train --data_path data/raw/credit-g.csv --use_tuner --track_with_mlflow
```

### Making Predictions with MLFlow Tracking

```bash
python src/main.py --mode predict --model_path models/20230101_120000/credit_model.keras --input_data data/raw/new_data.csv --track_with_mlflow
```

### Generating Visualizations with MLFlow Tracking

```bash
python src/main.py --mode visualize --model_path models/20230101_120000/credit_model.keras --output_path models/20230101_120000/predictions.csv --track_with_mlflow
```

### Running the Full Pipeline with MLFlow Tracking

```bash
python src/main.py --mode full --data_path data/raw/credit-g.csv --track_with_mlflow
```

### Using the Best Model from MLFlow

```bash
# Get the best model based on validation accuracy
python src/main.py --mode mlflow_best --mlflow_metric val_accuracy

# Get the best model and make predictions
python src/main.py --mode mlflow_best --mlflow_metric val_accuracy --input_data data/raw/new_data.csv
```

## MLFlow Experiment Tracking

The project uses MLFlow for comprehensive experiment tracking. MLFlow tracks:

### Metrics
- Training and validation metrics (accuracy, loss)
- Test metrics (accuracy, F1 score, precision, recall)
- ROC AUC score
- Confusion matrix metrics

### Parameters
- Model architecture parameters
- Training parameters (learning rate, batch size, epochs)
- Hyperparameter tuning settings
- Feature preprocessing configurations

### Artifacts
- Trained models
- Preprocessor configurations
- Confusion matrix plots
- ROC curves
- Training history plots
- Precision-recall curves
- Feature statistics
- Data validation reports
- Requirements file

### Model Registry

The project integrates with MLFlow Model Registry for model versioning and deployment:

```bash
# Register a model to the registry
python src/main.py --mode train --register_model --model_stage Staging

# Promote a model to production
python src/main.py --mode promote_model --model_stage Production

# Use a model from the registry for prediction
python src/main.py --mode predict --model_stage Production --input_data data/raw/new_data.csv
```

### Model Versioning

Models are versioned in the MLFlow registry with the following stages:
- **None**: Initial stage for newly registered models
- **Staging**: Models that are being evaluated
- **Production**: Models that are deployed to production
- **Archived**: Models that are no longer in use

### Model Comparison

Compare multiple model runs to select the best model:

```bash
# Compare the top 5 models based on validation accuracy
python src/main.py --mode compare_models --mlflow_metric val_accuracy --compare_runs_count 5
```

### Model Signatures

Models are logged with input and output signatures, enabling:
- Automatic validation of input data
- Documentation of expected input formats
- Consistent model serving

### Data Validation

Input data is validated and logged with statistics:
- Feature distributions
- Missing value counts
- Data shape and types
- Feature correlations

### Viewing MLFlow UI

To view the MLFlow tracking UI:

1. Start the MLFlow UI server:
```bash
mlflow ui
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

### MLFlow Configuration

MLFlow settings can be configured in `config/model_config.py`:

```python
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
```

## Model Architecture

The model is a neural network with the following components:

1. Input layers for different feature types (continuous, discrete, categorical)
2. Feature processing layers (log transform, standardization, embedding)
3. Dense layers with dropout for regularization
4. Sigmoid output layer for binary classification

## Data

The model uses the German Credit dataset (credit-g) from OpenML, which contains information about credit applicants and their credit risk classification.

## Configuration

The project uses a centralized configuration file (`config/model_config.py`) to manage all parameters:

- Feature definitions
- Training parameters
- Hyperparameter tuning settings
- Model architecture search space
- Callback parameters
- File paths
- MLFlow configuration

This separation of configuration from code makes the project more maintainable and easier to modify.

## License

[MIT License](LICENSE)

## CI/CD Pipeline

The project includes a comprehensive CI/CD pipeline for automated model training, testing, and deployment:

### Automated Training (`model-training.yml`)
- Weekly automated training runs (Sundays at midnight)
- Manual trigger option
- Dependency caching for faster builds
- Code quality checks (black, flake8)
- Test coverage reporting
- Model performance validation
- Artifact upload

### Automated Deployment (`model-deployment.yml`)
- Triggered after successful training or manually
- Canary deployment with gradual traffic shifting
- Health monitoring and automatic rollback
- MLFlow model registry integration
- Slack notifications for deployment status

### Monitoring and Metrics
- Prometheus metrics integration
- Model performance monitoring
- System resource tracking
- Feature drift detection
- Error rate and latency monitoring

### Development Environment

The project uses a containerized development environment with:

1. GPU Support:
```bash
# Start development environment
devfile run install
```

2. Available Services:
- MLFlow UI (port 5000)
- Jupyter Lab (port 8888)
- TensorBoard (port 6006)
- Prometheus Metrics (port 8000)

3. Development Tools:
- Code formatting (black)
- Linting (flake8)
- Test coverage (pytest-cov)
- API testing tools

### Model Serving

The project includes a production-ready model serving API with:

1. Features:
- A/B testing support
- Canary deployments
- Automatic rollbacks
- Health monitoring
- Rate limiting
- Authentication

2. API Endpoints:
```
GET  /api/v1/health           - Basic health check
GET  /api/v1/health/detailed  - Detailed health status
POST /api/v1/predict         - Single prediction
POST /api/v1/batch/predict   - Batch prediction
POST /api/v1/model/deploy    - Deploy model (canary/immediate)
GET  /api/v1/model/deployment/status - Get deployment status
POST /api/v1/model/rollback  - Rollback deployment
GET  /api/v1/model/info      - Model information
GET  /metrics               - Prometheus metrics
```

3. Security:
- JWT authentication
- Role-based access control
- Rate limiting
- CORS support

### Environment Setup

1. Development Environment:
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate credit_risk_model

# Install development dependencies
pip install -r requirements-dev.txt
```

2. MLFlow Setup:
```bash
# Start MLFlow server
mlflow server --host 0.0.0.0

# View UI at http://localhost:5000
```

3. Model Serving:
```bash
# Start API server
python -m credit_risk_model.src.api.serve_model

# View Prometheus metrics at http://localhost:8000/metrics
```

### Continuous Integration

The project uses GitHub Actions for CI/CD:

1. On Pull Request:
- Code quality checks
- Unit tests
- Test coverage reporting
- Model validation

2. On Merge to Main:
- Model training
- Performance evaluation
- Artifact generation
- Canary deployment

3. Monitoring:
- Model performance metrics
- API health checks
- System resource usage
- Deployment status 

## Troubleshooting

### Common Installation Issues

1. **GPU Support**:
   - Ensure NVIDIA drivers are installed
   - Check CUDA version compatibility
   - Verify TensorFlow can see your GPU:
     ```python
     import tensorflow as tf
     print(tf.config.list_physical_devices('GPU'))
     ```

2. **Dependency Conflicts**:
   - Use virtual environments
   - Install dependencies in order (base → extras)
   - Check package version compatibility

3. **Development Tools**:
   - Ensure nodejs is installed for JupyterLab
   - Configure git hooks for pre-commit
   - Set up IDE integration for black/flake8

### Getting Help

- Check the [Issues](https://github.com/lenhanpham/credit-card-risk-model/issues) page
- Review error logs in `logs/` directory
- Contact the maintainer: Le Nhan Pham 