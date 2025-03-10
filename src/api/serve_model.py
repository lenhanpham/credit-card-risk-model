import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from datetime import datetime
from sklearn.metrics import confusion_matrix
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge
import threading
import time
from functools import wraps
import requests
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union
import jwt
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import hashlib
import secrets
from dataclasses import dataclass
from enum import Enum
import psutil

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.data.make_dataset import CreditDataPreprocessor
from src.models.predict_model import load_model, predict
from config.model_config import (
    DEFAULT_MODEL_PATH, PREPROCESSOR_CONFIG_PATH,
    MLFLOW_TRACKING_ENABLED, MLFLOW_TRACKING_URI
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Prometheus metrics
PREDICTION_REQUEST_COUNT = Counter('prediction_requests_total', 'Total prediction requests',
                                     ['model_type', 'status'])
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency',
                                 ['model_type'])
MODEL_PREDICTION_DISTRIBUTION = Histogram('prediction_distribution', 'Distribution of predictions',
                                           ['model_type'])
MODEL_ERROR_COUNT = Counter('prediction_errors_total', 'Total prediction errors',
                              ['error_type'])
MODEL_LOAD_TIME = Gauge('model_load_time_seconds', 'Time taken to load model',
                          ['model_type'])
FEATURE_VALUE_DISTRIBUTION = Histogram('feature_value_distribution', 'Distribution of feature values',
                                     labelnames=['feature_name'])

# Security Configuration
class UserRole(Enum):
    ADMIN = "admin"
    MODEL_DEVELOPER = "model_developer"
    CLIENT = "client"

@dataclass
class SecurityConfig:
    JWT_SECRET: str = os.getenv('JWT_SECRET', secrets.token_hex(32))
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    CORS_ORIGINS: List[str] = field(default_factory=lambda: ["http://localhost:3000"])
    API_KEY_SALT: str = os.getenv('API_KEY_SALT', secrets.token_hex(16))
    REQUIRED_ROLES: Dict[str, List[UserRole]] = field(default_factory=lambda: {
        "promote_model": [UserRole.ADMIN, UserRole.MODEL_DEVELOPER],
        "rollback_model": [UserRole.ADMIN, UserRole.MODEL_DEVELOPER],
        "reload_model": [UserRole.ADMIN, UserRole.MODEL_DEVELOPER],
        "predict": [UserRole.CLIENT, UserRole.MODEL_DEVELOPER, UserRole.ADMIN],
        "batch_predict": [UserRole.CLIENT, UserRole.MODEL_DEVELOPER, UserRole.ADMIN]
    })

# Initialize security config
security_config = SecurityConfig()

# Initialize Flask app with CORS
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {"origins": security_config.CORS_ORIGINS}
})

# Global variables to store models and preprocessors for A/B testing
models = {
    'production': {'model': None, 'weight': 0.8},
    'candidate': {'model': None, 'weight': 0.2}
}
preprocessor = None
model_metadata = {
    'production': {
        "last_loaded": None,
        "current_stage": None,
        "version": None,
        "path": None
    },
    'candidate': {
        "last_loaded": None,
        "current_stage": None,
        "version": None,
        "path": None
    }
}

# Add API version
API_VERSION = 'v1'

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# TensorFlow Serving configuration
TF_SERVING_URL = os.getenv('TF_SERVING_URL', 'http://localhost:8501')

# Enhanced Input Validation Models
class PredictionFeatures(BaseModel):
    feature_name: str
    value: Union[float, int, str]
    
    @validator('value')
    def validate_value(cls, v):
        if isinstance(v, str) and not v.strip():
            raise ValueError("Value cannot be empty string")
        return v

class PredictionRequest(BaseModel):
    data: List[Dict[str, Union[float, int, str]]]
    labels: Optional[List[int]] = None
    batch_size: Optional[int] = Field(default=1000, gt=0, le=10000)
    
    @validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError("Data cannot be empty")
        if len(v) > 10000:
            raise ValueError("Maximum batch size exceeded (10000)")
        return v

class ModelPromotionRequest(BaseModel):
    strategy: str = Field(default="canary", regex="^(canary|immediate)$")
    initial_weight: float = Field(default=0.1, gt=0, lt=1)
    validation_metrics: Optional[Dict[str, float]] = None

class DeploymentConfig:
    """Configuration for model deployment."""
    CANARY_INITIAL_WEIGHT = 0.1
    CANARY_STEP_SIZE = 0.1
    CANARY_INTERVAL_HOURS = 1
    PERFORMANCE_THRESHOLD = 0.95
    DRIFT_THRESHOLD = 0.1
    ERROR_THRESHOLD = 0.01
    LATENCY_THRESHOLD = 0.5

class DeploymentMetrics:
    """Track deployment metrics."""
    def __init__(self):
        self.accuracy = 0.0
        self.error_rate = 0.0
        self.latency = 0.0
        self.drift_score = 0.0
        self.start_time = None
        self.prediction_count = 0
        
    def update(self, accuracy=None, error=None, latency=None, drift=None):
        if accuracy is not None:
            self.accuracy = accuracy
        if error is not None:
            self.error_rate = error
        if latency is not None:
            self.latency = latency
        if drift is not None:
            self.drift_score = drift
        self.prediction_count += 1

class DeploymentManager:
    """Manage model deployments and rollbacks."""
    def __init__(self):
        self.config = DeploymentConfig()
        self.metrics = {
            'production': DeploymentMetrics(),
            'candidate': DeploymentMetrics()
        }
        self.deployment_status = None
        self.rollback_triggered = False
    
    def start_canary_deployment(self, initial_weight=None):
        """Start canary deployment."""
        weight = initial_weight or self.config.CANARY_INITIAL_WEIGHT
        models['candidate']['weight'] = weight
        models['production']['weight'] = 1 - weight
        self.deployment_status = 'canary'
        self.metrics['candidate'].start_time = datetime.now()
        
        # Start monitoring thread
        threading.Thread(target=self._monitor_canary, daemon=True).start()
    
    def _monitor_canary(self):
        """Monitor canary deployment metrics."""
        while self.deployment_status == 'canary' and not self.rollback_triggered:
            try:
                # Check if metrics meet thresholds
                candidate_metrics = self.metrics['candidate']
                if candidate_metrics.prediction_count > 1000:  # Minimum sample size
                    if (candidate_metrics.accuracy < self.config.PERFORMANCE_THRESHOLD or
                        candidate_metrics.error_rate > self.config.ERROR_THRESHOLD or
                        candidate_metrics.drift_score > self.config.DRIFT_THRESHOLD or
                        candidate_metrics.latency > self.config.LATENCY_THRESHOLD):
                        self.trigger_rollback("Metrics below threshold")
                        break
                    
                    # Increase traffic if metrics are good
                    current_weight = models['candidate']['weight']
                    if current_weight < 1.0:
                        new_weight = min(current_weight + self.config.CANARY_STEP_SIZE, 1.0)
                        models['candidate']['weight'] = new_weight
                        models['production']['weight'] = 1 - new_weight
                        
                        logger.info(
                            f"Increased canary traffic to {new_weight:.1%}. "
                            f"Metrics: accuracy={candidate_metrics.accuracy:.3f}, "
                            f"error_rate={candidate_metrics.error_rate:.3f}, "
                            f"drift_score={candidate_metrics.drift_score:.3f}"
                        )
                
                time.sleep(self.config.CANARY_INTERVAL_HOURS * 3600)
            
            except Exception as e:
                logger.error(f"Error in canary monitoring: {str(e)}")
                time.sleep(60)
    
    def trigger_rollback(self, reason):
        """Trigger model rollback."""
        logger.warning(f"Triggering rollback: {reason}")
        self.rollback_triggered = True
        
        try:
            # Restore traffic to production
            models['production']['weight'] = 1.0
            models['candidate']['weight'] = 0.0
            
            # Swap back models if needed
            if self.deployment_status == 'promoted':
                swap_production_candidate()
            
            self.deployment_status = 'rolled_back'
            
            # Notify stakeholders
            self._send_rollback_notification(reason)
        
        except Exception as e:
            logger.error(f"Error during rollback: {str(e)}")
    
    def _send_rollback_notification(self, reason):
        """Send rollback notification to stakeholders."""
        message = {
            "type": "rollback",
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "candidate": vars(self.metrics['candidate']),
                "production": vars(self.metrics['production'])
            }
        }
        
        try:
            # Send to notification service (implement as needed)
            logger.info(f"Rollback notification: {message}")
        except Exception as e:
            logger.error(f"Failed to send rollback notification: {str(e)}")

# Initialize deployment manager
deployment_manager = DeploymentManager()

def require_auth_with_roles(allowed_roles: List[UserRole]):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            if not auth_header:
                return jsonify({"status": "error", "message": "Missing authorization header"}), 401
            
            try:
                token = auth_header.split(" ")[1]
                payload = jwt.decode(token, security_config.JWT_SECRET, 
                                   algorithms=[security_config.JWT_ALGORITHM])
                
                user_role = UserRole(payload.get('role'))
                if user_role not in allowed_roles:
                    return jsonify({"status": "error", "message": "Insufficient permissions"}), 403
                
            except jwt.ExpiredSignatureError:
                return jsonify({"status": "error", "message": "Token has expired"}), 401
            except (jwt.InvalidTokenError, ValueError) as e:
                return jsonify({"status": "error", "message": "Invalid token"}), 401
            
            return f(*args, **kwargs)
        return decorated
    return decorator

# Initialize TF Serving client
tf_serving_client = TFServingClient(TF_SERVING_URL)

def monitor_model_health():
    """Background thread to monitor model health."""
    while True:
        try:
            # Check model prediction distribution
            if 'production' in models and models['production']['model'] is not None:
                # Log model metadata
                logger.info(f"Production model metadata: {model_metadata['production']}")
                
                # Log memory usage
                memory_usage = tf.config.experimental.get_memory_info('GPU:0')['current'] if tf.config.list_physical_devices('GPU') else 0
                logger.info(f"Current memory usage: {memory_usage}")
            
            time.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Error in model health monitoring: {str(e)}")

# Start monitoring thread
monitoring_thread = threading.Thread(target=monitor_model_health, daemon=True)
monitoring_thread.start()

def setup_mlflow():
    """Initialize MLFlow configuration."""
    if MLFLOW_TRACKING_ENABLED:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            logger.info(f"MLFlow tracking URI set to: {MLFLOW_TRACKING_URI}")
        except Exception as e:
            logger.error(f"Failed to set MLFlow tracking URI: {str(e)}")
            raise

def load_model_and_preprocessor(model_path=None, preprocessor_config_path=None, stage='Production', model_type='production'):
    """
    Load the model and preprocessor, with support for MLFlow model registry and A/B testing.
    
    Args:
        model_path: Path to the model file or MLFlow model URI
        preprocessor_config_path: Path to the preprocessor configuration file
        stage: MLFlow model stage to load ('Production', 'Staging', 'None')
        model_type: Type of model to load ('production' or 'candidate')
    """
    global models, preprocessor, model_metadata
    
    start_time = time.time()
    try:
        setup_mlflow()
        
        if MLFLOW_TRACKING_ENABLED:
            if model_path and model_path.startswith('models:/'):
                logger.info(f"Loading {model_type} model from MLFlow registry: {model_path}")
                models[model_type]['model'] = mlflow.tensorflow.load_model(model_path)
                model_metadata[model_type]["path"] = model_path
            else:
                registry_path = f"models:/credit_risk_model/{stage}"
                logger.info(f"Loading {model_type} model from MLFlow registry: {registry_path}")
                models[model_type]['model'] = mlflow.tensorflow.load_model(registry_path)
                model_metadata[model_type]["path"] = registry_path
            
            # Update model metadata
            model_info = mlflow.tensorflow.get_model_info("credit_risk_model")
            model_metadata[model_type].update({
                "last_loaded": datetime.now().isoformat(),
                "current_stage": stage,
                "version": model_info.version
            })
        else:
            if model_path is None:
                model_path = DEFAULT_MODEL_PATH
            logger.info(f"Loading {model_type} model from local path: {model_path}")
            models[model_type]['model'] = load_model(model_path)
            model_metadata[model_type].update({
                "last_loaded": datetime.now().isoformat(),
                "path": model_path
            })
    except Exception as e:
        logger.error(f"Failed to load {model_type} model: {str(e)}")
        raise RuntimeError(f"Failed to load {model_type} model: {str(e)}")
    
    # Load preprocessor only if not already loaded
    if preprocessor is None:
        try:
            if preprocessor_config_path is None:
                preprocessor_config_path = PREPROCESSOR_CONFIG_PATH
            
            logger.info(f"Loading preprocessor config from: {preprocessor_config_path}")
            with open(preprocessor_config_path, 'r') as f:
                config = json.load(f)
            
            preprocessor = CreditDataPreprocessor(
                discrete_features=config['discrete_features'],
                categorical_features=config['categorical_features'],
                continuous_features=config['continuous_features']
            )
            logger.info("Preprocessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to load preprocessor config: {str(e)}")
            raise RuntimeError(f"Failed to load preprocessor config: {str(e)}")
    
    load_time = time.time() - start_time
    MODEL_LOAD_TIME.labels(model_type=model_type).set(load_time)
    logger.info(f"Model {model_type} loaded in {load_time:.2f} seconds")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint with detailed status."""
    status = {
        "status": "error",
        "components": {
            "production_model": "not_loaded",
            "candidate_model": "not_loaded",
            "preprocessor": "not_loaded"
        },
        "metadata": {
            "production": model_metadata["production"],
            "candidate": model_metadata["candidate"]
        }
    }
    
    if models['production']['model'] is not None:
        status["components"]["production_model"] = "loaded"
    if models['candidate']['model'] is not None:
        status["components"]["candidate_model"] = "loaded"
    if preprocessor is not None:
        status["components"]["preprocessor"] = "loaded"
    
    if models['production']['model'] is not None and preprocessor is not None:
        status["status"] = "ok"
        return jsonify(status), 200
    return jsonify(status), 503

@app.route(f'/api/{API_VERSION}/predict', methods=['POST'])
@limiter.limit("10/minute")
@require_auth_with_roles(security_config.REQUIRED_ROLES["predict"])
def predict_endpoint():
    """Prediction endpoint with A/B testing, monitoring, and TF Serving support."""
    PREDICTION_REQUEST_COUNT.labels(model_type='production', status='success').inc()
    
    if models['production']['model'] is None or preprocessor is None:
        logger.error("Model or preprocessor not loaded")
        MODEL_ERROR_COUNT.labels(error_type='model_not_loaded').inc()
        return jsonify({"status": "error", "message": "Model or preprocessor not loaded"}), 503
    
    with PREDICTION_LATENCY.labels(model_type='production').time():
        try:
            # Validate request
            try:
                request_data = PredictionRequest(**request.get_json())
            except ValueError as e:
                logger.warning(f"Invalid request format: {str(e)}")
                MODEL_ERROR_COUNT.labels(error_type='invalid_request_format').inc()
                return jsonify({"status": "error", "message": str(e)}), 400
            
            data = pd.DataFrame(request_data.data)
            logger.info(f"Received prediction request for {len(data)} samples")
            
            # Input validation and preprocessing
            try:
                processed_data = preprocess_input(data)
            except ValueError as e:
                logger.error(f"Input preprocessing failed: {str(e)}")
                MODEL_ERROR_COUNT.labels(error_type='input_preprocessing_failed').inc()
                return jsonify({"status": "error", "message": str(e)}), 400
            
            # A/B testing logic
            selected_model = select_model_for_request()
            
            # Make predictions using TF Serving
            try:
                predictions = tf_serving_client.predict(processed_data.tolist())
                logger.info(f"Generated predictions using {selected_model} model for {len(predictions)} samples")
            except Exception as e:
                # Fallback to local model if TF Serving fails
                logger.warning(f"TF Serving failed, falling back to local model: {str(e)}")
                predictions = predict(models[selected_model]['model'], data, preprocessor)
            
            # Enhanced monitoring and metrics
            log_prediction_metrics(predictions, data, selected_model)
            
            # Format and validate results
            results = format_predictions(predictions, selected_model)
            
            # Log to MLFlow with enhanced metadata
            if MLFLOW_TRACKING_ENABLED:
                log_mlflow_predictions(predictions, request_data, selected_model)
            
            response = {
                "status": "success",
                "predictions": results,
                "metadata": get_prediction_metadata(selected_model)
            }
            
            return jsonify(response), 200
        
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            MODEL_ERROR_COUNT.labels(error_type='prediction_error').inc()
            return jsonify({"status": "error", "message": str(e)}), 500

def preprocess_input(data: pd.DataFrame) -> np.ndarray:
    """Preprocess input data with validation."""
    required_features = set(preprocessor.discrete_features + 
                          preprocessor.categorical_features + 
                          preprocessor.continuous_features)
    
    missing_features = required_features - set(data.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {', '.join(missing_features)}")
    
    # Log feature distributions
    for feature in data.columns:
        if feature in preprocessor.continuous_features:
            FEATURE_VALUE_DISTRIBUTION.labels(feature_name=feature).observe(data[feature].mean())
    
    return preprocessor.transform(data)

def select_model_for_request() -> str:
    """Select model based on A/B testing rules."""
    if models['candidate']['model'] is not None and np.random.random() < models['candidate']['weight']:
        return 'candidate'
    return 'production'

def log_prediction_metrics(predictions: np.ndarray, data: pd.DataFrame, model_type: str):
    """Log comprehensive prediction metrics."""
    # Basic prediction metrics
    for pred in predictions.flatten():
        MODEL_PREDICTION_DISTRIBUTION.labels(model_type=model_type).observe(pred)
    
    # Prediction statistics
    prediction_mean = np.mean(predictions)
    prediction_std = np.std(predictions)
    good_ratio = np.mean(predictions > 0.5)
    
    # Update recent predictions for drift detection
    if not hasattr(preprocessor, 'recent_predictions'):
        preprocessor.recent_predictions = []
    preprocessor.recent_predictions = (preprocessor.recent_predictions[-9900:] + 
                                    predictions.flatten().tolist())
    
    # Log feature distributions
    for feature in data.columns:
        if feature in preprocessor.continuous_features:
            current_mean = data[feature].mean()
            FEATURE_VALUE_DISTRIBUTION.labels(feature_name=feature).observe(current_mean)
            
            # Check for feature drift
            if hasattr(preprocessor, f'{feature}_reference_mean'):
                reference_mean = getattr(preprocessor, f'{feature}_reference_mean')
                reference_std = getattr(preprocessor, f'{feature}_reference_std')
                drift_score = abs(current_mean - reference_mean) / reference_std
                FEATURE_DRIFT_SCORE.labels(feature_name=feature).set(drift_score)
    
    logger.info(
        f"Prediction stats for {model_type} - "
        f"Mean: {prediction_mean:.3f}, Std: {prediction_std:.3f}, "
        f"Good ratio: {good_ratio:.3f}"
    )

def format_predictions(predictions: np.ndarray, model_type: str) -> List[Dict]:
    """Format predictions with confidence scores."""
    return [
        {
            "prediction": float(pred),
            "credit_risk": "good" if pred > 0.5 else "bad",
            "confidence": float(abs(pred - 0.5) * 2),
            "model_version": model_metadata[model_type].get("version", "unknown")
        }
        for pred in predictions.flatten()
    ]

def get_prediction_metadata(model_type: str) -> Dict:
    """Get comprehensive prediction metadata."""
    return {
        "model_type": model_type,
        "model_version": model_metadata[model_type].get("version"),
        "model_stage": model_metadata[model_type].get("current_stage"),
        "timestamp": datetime.now().isoformat(),
        "api_version": API_VERSION
    }

@app.route(f'/api/{API_VERSION}/batch/predict', methods=['POST'])
@require_auth_with_roles(security_config.REQUIRED_ROLES["batch_predict"])
def batch_predict():
    """Batch prediction endpoint for large datasets."""
    try:
        request_data = request.get_json()
        batch_size = request_data.get('batch_size', 1000)
        data = pd.DataFrame(request_data['data'])
        
        results = []
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            predictions = predict_endpoint(batch)
            results.extend(predictions)
        
        return jsonify({
            "status": "success",
            "predictions": results,
            "metadata": {
                "total_records": len(data),
                "batch_size": batch_size,
                "num_batches": len(results) // batch_size + 1
            }
        })
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route(f'/api/{API_VERSION}/model/deploy', methods=['POST'])
@require_auth_with_roles(security_config.REQUIRED_ROLES["promote_model"])
def deploy_model():
    """Deploy model with canary deployment."""
    try:
        request_data = request.get_json()
        deployment_type = request_data.get('type', 'canary')
        initial_weight = request_data.get('initial_weight', None)
        
        if deployment_type == 'canary':
            deployment_manager.start_canary_deployment(initial_weight)
            return jsonify({
                "status": "success",
                "message": "Canary deployment started",
                "initial_weight": initial_weight or deployment_manager.config.CANARY_INITIAL_WEIGHT
            })
        elif deployment_type == 'immediate':
            swap_production_candidate()
            return jsonify({
                "status": "success",
                "message": "Immediate deployment completed"
            })
        else:
            return jsonify({
                "status": "error",
                "message": f"Unknown deployment type: {deployment_type}"
            }), 400
    
    except Exception as e:
        logger.error(f"Error in model deployment: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route(f'/api/{API_VERSION}/model/deployment/status', methods=['GET'])
@require_auth_with_roles(security_config.REQUIRED_ROLES["promote_model"])
def get_deployment_status():
    """Get current deployment status."""
    try:
        status = {
            "deployment_status": deployment_manager.deployment_status,
            "rollback_triggered": deployment_manager.rollback_triggered,
            "metrics": {
                "candidate": vars(deployment_manager.metrics['candidate']),
                "production": vars(deployment_manager.metrics['production'])
            },
            "weights": {
                "candidate": models['candidate']['weight'],
                "production": models['production']['weight']
            }
        }
        return jsonify({"status": "success", "deployment_status": status})
    
    except Exception as e:
        logger.error(f"Error getting deployment status: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus metrics endpoint."""
    return prometheus_client.generate_latest()

@app.route('/model/reload', methods=['POST'])
@require_auth_with_roles(security_config.REQUIRED_ROLES["reload_model"])
def reload_model():
    """
    Reload models with A/B testing support.
    
    Expects a JSON with the following format (all fields optional):
    {
        "production_model": {
            "path": "path/to/model.keras" or "models:/model_name/stage",
            "weight": 0.8
        },
        "candidate_model": {
            "path": "path/to/model.keras" or "models:/model_name/stage",
            "weight": 0.2
        },
        "preprocessor_config_path": "path/to/preprocessor_config.json"
    }
    """
    try:
        request_data = request.get_json() or {}
        
        if 'production_model' in request_data:
            prod_config = request_data['production_model']
            load_model_and_preprocessor(
                model_path=prod_config.get('path'),
                preprocessor_config_path=request_data.get('preprocessor_config_path'),
                stage='Production',
                model_type='production'
            )
            if 'weight' in prod_config:
                models['production']['weight'] = float(prod_config['weight'])
        
        if 'candidate_model' in request_data:
            cand_config = request_data['candidate_model']
            load_model_and_preprocessor(
                model_path=cand_config.get('path'),
                preprocessor_config_path=request_data.get('preprocessor_config_path'),
                stage='Staging',
                model_type='candidate'
            )
            if 'weight' in cand_config:
                models['candidate']['weight'] = float(cand_config['weight'])
        
        return jsonify({
            "status": "success",
            "message": "Models reloaded successfully",
            "weights": {
                "production": models['production']['weight'],
                "candidate": models['candidate']['weight']
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Error during model reload: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get detailed information about loaded models."""
    if models['production']['model'] is None and models['candidate']['model'] is None:
        return jsonify({"status": "error", "message": "No models loaded"}), 503
    
    try:
        info = {
            "models": {
                "production": {
                    "loaded": models['production']['model'] is not None,
                    "weight": models['production']['weight'],
                    "metadata": model_metadata['production']
                },
                "candidate": {
                    "loaded": models['candidate']['model'] is not None,
                    "weight": models['candidate']['weight'],
                    "metadata": model_metadata['candidate']
                }
            },
            "preprocessor": {
                "loaded": preprocessor is not None,
                "features": {
                    "discrete": preprocessor.discrete_features if preprocessor else None,
                    "categorical": preprocessor.categorical_features if preprocessor else None,
                    "continuous": preprocessor.continuous_features if preprocessor else None
                }
            }
        }
        
        # Add MLFlow information if available
        if MLFLOW_TRACKING_ENABLED:
            try:
                for model_type in ['production', 'candidate']:
                    if models[model_type]['model'] is not None:
                        model_info = mlflow.tensorflow.get_model_info("credit_risk_model")
                        info["models"][model_type]["mlflow"] = {
                            "model_name": "credit_risk_model",
                            "current_stage": model_info.current_stage,
                            "version": model_info.version,
                            "creation_timestamp": model_info.creation_timestamp
                        }
            except Exception as e:
                info["mlflow_status"] = f"Error fetching MLFlow info: {str(e)}"
        
        return jsonify({"status": "success", "info": info}), 200
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

def setup_monitoring():
    """Initialize monitoring metrics."""
    global PREDICTION_REQUEST_COUNT, PREDICTION_LATENCY, MODEL_PREDICTION_DISTRIBUTION
    global MODEL_ERROR_COUNT, MODEL_LOAD_TIME, FEATURE_VALUE_DISTRIBUTION
    
    # Basic metrics
    PREDICTION_REQUEST_COUNT = Counter('prediction_requests_total', 'Total prediction requests',
                                     ['model_type', 'status'])
    PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency',
                                 ['model_type'])
    MODEL_PREDICTION_DISTRIBUTION = Histogram('prediction_distribution', 'Distribution of predictions',
                                           ['model_type'])
    MODEL_ERROR_COUNT = Counter('prediction_errors_total', 'Total prediction errors',
                              ['error_type'])
    MODEL_LOAD_TIME = Gauge('model_load_time_seconds', 'Time taken to load model',
                          ['model_type'])
    
    # Advanced metrics
    global MODEL_MEMORY_USAGE, MODEL_PREDICTION_ACCURACY, MODEL_DRIFT_SCORE
    global FEATURE_DRIFT_SCORE, SYSTEM_MEMORY_USAGE, API_RESPONSE_TIME
    
    MODEL_MEMORY_USAGE = Gauge('model_memory_usage_bytes', 'Memory usage by model',
                             ['model_type'])
    MODEL_PREDICTION_ACCURACY = Gauge('model_prediction_accuracy', 'Model prediction accuracy',
                                   ['model_type'])
    MODEL_DRIFT_SCORE = Gauge('model_drift_score', 'Model drift score',
                            ['model_type', 'metric'])
    FEATURE_DRIFT_SCORE = Gauge('feature_drift_score', 'Feature drift score',
                              ['feature_name'])
    SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'System memory usage')
    API_RESPONSE_TIME = Histogram('api_response_time_seconds', 'API response time',
                                ['endpoint'])

def monitor_system_health():
    """Monitor system health metrics."""
    while True:
        try:
            # Monitor system resources
            memory_info = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.set(memory_info.used)
            
            # Monitor model memory usage
            for model_type in ['production', 'candidate']:
                if models[model_type]['model'] is not None:
                    try:
                        memory_usage = tf.config.experimental.get_memory_info('GPU:0')['current'] \
                            if tf.config.list_physical_devices('GPU') else 0
                        MODEL_MEMORY_USAGE.labels(model_type=model_type).set(memory_usage)
                    except Exception as e:
                        logger.warning(f"Failed to get memory usage for {model_type} model: {str(e)}")
            
            # Monitor model drift
            if hasattr(preprocessor, 'reference_distribution'):
                current_distribution = get_current_distribution()
                drift_score = calculate_drift_score(preprocessor.reference_distribution, 
                                                 current_distribution)
                MODEL_DRIFT_SCORE.labels(model_type='production', 
                                      metric='distribution').set(drift_score)
            
            time.sleep(60)  # Check every minute
            
        except Exception as e:
            logger.error(f"Error in system health monitoring: {str(e)}")
            time.sleep(60)  # Wait before retrying

def get_current_distribution():
    """Get current feature distribution statistics."""
    if not hasattr(preprocessor, 'recent_predictions'):
        return None
    return {
        'mean': np.mean(preprocessor.recent_predictions),
        'std': np.std(preprocessor.recent_predictions),
        'quantiles': np.percentile(preprocessor.recent_predictions, [25, 50, 75])
    }

def calculate_drift_score(reference_dist, current_dist):
    """Calculate drift score between reference and current distributions."""
    if not reference_dist or not current_dist:
        return 0.0
    
    # Calculate KL divergence or other drift metrics
    drift_score = abs(reference_dist['mean'] - current_dist['mean']) / reference_dist['std']
    return float(drift_score)

@app.route('/health/detailed', methods=['GET'])
def detailed_health_check():
    """Detailed health check endpoint with system metrics."""
    try:
        status = {
            "status": "error",
            "components": {
                "production_model": {
                    "status": "not_loaded",
                    "memory_usage": None,
                    "drift_score": None,
                    "last_prediction_time": None
                },
                "candidate_model": {
                    "status": "not_loaded",
                    "memory_usage": None,
                    "drift_score": None,
                    "last_prediction_time": None
                },
                "preprocessor": {
                    "status": "not_loaded",
                    "feature_drift_scores": {}
                }
            },
            "system": {
                "memory_usage": psutil.virtual_memory().percent,
                "cpu_usage": psutil.cpu_percent(),
                "disk_usage": psutil.disk_usage('/').percent
            }
        }
        
        # Check models
        for model_type in ['production', 'candidate']:
            if models[model_type]['model'] is not None:
                model_status = status["components"][model_type]
                model_status["status"] = "loaded"
                model_status["memory_usage"] = tf.config.experimental.get_memory_info('GPU:0')['current'] \
                    if tf.config.list_physical_devices('GPU') else 0
                model_status["drift_score"] = calculate_drift_score(
                    getattr(preprocessor, 'reference_distribution', None),
                    get_current_distribution()
                )
                model_status["last_prediction_time"] = model_metadata[model_type].get("last_prediction_time")
        
        # Check preprocessor
        if preprocessor is not None:
            status["components"]["preprocessor"]["status"] = "loaded"
            for feature in preprocessor.continuous_features:
                if hasattr(preprocessor, f'{feature}_reference_mean'):
                    current_mean = getattr(preprocessor, f'{feature}_current_mean', 0)
                    reference_mean = getattr(preprocessor, f'{feature}_reference_mean')
                    reference_std = getattr(preprocessor, f'{feature}_reference_std')
                    drift_score = abs(current_mean - reference_mean) / reference_std
                    status["components"]["preprocessor"]["feature_drift_scores"][feature] = drift_score
        
        # Set overall status
        if models['production']['model'] is not None and preprocessor is not None:
            status["status"] = "ok"
            
        return jsonify(status), 200
    
    except Exception as e:
        logger.error(f"Error in detailed health check: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Failed to get detailed health status",
            "error": str(e)
        }), 500

def main():
    """Main function to run the Flask app with monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the model serving API')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to run the API on')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to run the API on')
    parser.add_argument('--production_model', type=str, default=None,
                       help='Path to the production model file')
    parser.add_argument('--candidate_model', type=str, default=None,
                       help='Path to the candidate model file')
    parser.add_argument('--preprocessor_config', type=str, default=None,
                       help='Path to the preprocessor configuration file')
    parser.add_argument('--production_weight', type=float, default=0.8,
                       help='Weight for the production model in A/B testing')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    
    args = parser.parse_args()
    
    # Load models
    try:
        load_model_and_preprocessor(args.production_model, args.preprocessor_config, 'Production', 'production')
        models['production']['weight'] = args.production_weight
        
        if args.candidate_model:
            load_model_and_preprocessor(args.candidate_model, args.preprocessor_config, 'Staging', 'candidate')
            models['candidate']['weight'] = 1.0 - args.production_weight
        
        logger.info("Models and preprocessor loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models and preprocessor: {str(e)}")
        sys.exit(1)
    
    # Start Prometheus metrics server
    prometheus_client.start_http_server(8000)
    logger.info("Prometheus metrics server started on port 8000")
    
    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()