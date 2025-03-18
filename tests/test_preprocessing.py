import pytest
from src.data.preprocessing import CreditDataPreprocessor, create_tf_datasets, load_credit_data
from config.config import FEATURES_CONFIG, DATA_CONFIG

def test_load_credit_data():
    X, y = load_credit_data()
    assert X is not None, "X should not be None"
    assert y is not None, "y should not be None"
    assert len(X) == len(y), "X and y should have the same length"
    assert len(X) > 0, "Dataset should not be empty"

def test_preprocessor_init():
    preprocessor = CreditDataPreprocessor(
        FEATURES_CONFIG["discrete_features"],
        FEATURES_CONFIG["categorical_features"],
        FEATURES_CONFIG["continuous_features"]
    )
    assert preprocessor.discrete_features == FEATURES_CONFIG["discrete_features"]
    assert preprocessor.categorical_features == FEATURES_CONFIG["categorical_features"]
    assert preprocessor.continuous_features == FEATURES_CONFIG["continuous_features"]
    assert len(preprocessor.ordinal_encoders) == len(FEATURES_CONFIG["discrete_features"])
    assert len(preprocessor.categorical_encoders) == len(FEATURES_CONFIG["categorical_features"])

def test_create_tf_datasets():
    X, y = load_credit_data()
    train_ds, val_ds, test_ds = create_tf_datasets(
        X, y,
        DATA_CONFIG["train_size"],
        DATA_CONFIG["val_size"],
        DATA_CONFIG["batch_size"],
        DATA_CONFIG["seed"]
    )
    assert train_ds is not None, "Train dataset should not be None"
    assert val_ds is not None, "Validation dataset should not be None"
    assert test_ds is not None, "Test dataset should not be None"