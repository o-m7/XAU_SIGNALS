"""
Configuration module for the XAUUSD Signal Engine.

Dr. Chen Style Implementation:
==============================
Configuration for the TWO-STAGE signal system:
1. Environment classification (binary: good/bad)
2. Direction determination (from microstructure)
"""

from pathlib import Path

# =============================================================================
# HORIZON CONFIGURATION
# =============================================================================

HORIZONS = {
    "5m": {"minutes": 5, "k1": 1.5, "k2": 2.0},
    "15m": {"minutes": 15, "k1": 1.5, "k2": 2.5},
    "30m": {"minutes": 30, "k1": 1.5, "k2": 3.0},
}

# =============================================================================
# ENVIRONMENT LABELING
# =============================================================================

# Minimum movement/cost ratio to label as "good environment"
MIN_EDGE_MULTIPLIER = 1.5

# Micro-slippage estimate (additional cost beyond spread)
MICRO_SLIPPAGE_PCT = 0.0001

# Minimum volatility for tradeable environment
MIN_SIGMA_THRESHOLD = 0.00003

# =============================================================================
# VOLATILITY
# =============================================================================

VOL_LOOKBACKS = [5, 15, 30, 60]
PRIMARY_VOL_LOOKBACK = 60

# =============================================================================
# SIGNAL GENERATION (Two-Stage)
# =============================================================================

# Stage 1: Environment threshold
ENV_THRESHOLD = 0.5  # P(good_env) must exceed this

# Stage 2: Direction thresholds
IMBALANCE_THRESHOLD = 0.3  # |imbalance| must exceed for direction
SIGMA_SLOPE_THRESHOLD = 0.0  # Volatility must be >= this

# =============================================================================
# PRE-TRADE FILTERS
# =============================================================================

FILTER_PARAMS = {
    "min_sigma": 0.00003,
    "max_sigma": 0.003,
    "max_spread_pct": 0.001,
    "min_session_quality": 1,
}

# =============================================================================
# MODEL TRAINING
# =============================================================================

# Walk-forward dates
TRAIN_END = "2023-12-31"
VAL_END = "2024-06-30"

# XGBoost parameters for binary classification
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_weight": 20,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "gamma": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
    "early_stopping_rounds": 50,
}

# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT.parent / "Data"
MINUTE_DATA_DIR = DATA_DIR / "ohlcv_minute"
QUOTES_DATA_DIR = DATA_DIR / "quotes"
MODEL_DIR = PROJECT_ROOT / "models"
