"""Central configuration: paths, artifact names and constants.

Directories can be overridden with the environment variables
``WAFERWISE_DATA_DIR`` and ``WAFERWISE_MODELS_DIR`` (used by the tests and
useful for Docker volumes). All defaults are relative to the repository root.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RANDOM_SEED = 42
FAIL_LABEL = "FAIL"
PASS_LABEL = "PASS"

# Artifact file names written by src/train.py
FINAL_PIPELINE = "final_pipeline.joblib"
METADATA_JSON = "metadata.json"
METRICS_JSON = "metrics.json"
FEATURE_STATS_CSV = "feature_stats.csv"
FEATURE_IMPORTANCE_CSV = "feature_importance.csv"
SHAP_CACHE = "shap_cache.joblib"
MODEL_FILES = {
    "logistic_regression": "logistic_regression.joblib",
    "random_forest": "random_forest.joblib",
    "xgboost": "xgboost.joblib",
}
MODEL_DISPLAY = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}

# Column names that are treated as labels / identifiers when found in a CSV.
TARGET_NAME_CANDIDATES = (
    "pass/fail", "pass_fail", "passfail", "pass-fail", "target", "label",
    "class", "y", "fail", "status", "result", "outcome", "defect",
)
ID_NAME_CANDIDATES = ("id", "index", "sample_id", "unnamed: 0", "row_id")
TIME_NAME_CANDIDATES = ("time", "timestamp", "date", "datetime", "date_time")

MAX_UPLOAD_MB = 50


def get_data_dir() -> Path:
    return Path(os.environ.get("WAFERWISE_DATA_DIR", ROOT / "data"))


def get_models_dir() -> Path:
    return Path(os.environ.get("WAFERWISE_MODELS_DIR", ROOT / "models"))
