MODEL_NAME = "model6_orderflow"
MODEL_DESCRIPTION = "Order Flow / Microstructure-Based Strategy"
HORIZON_MINUTES = 30
TP_MULT = 1.5
SL_MULT = 1.0
THRESHOLD_LONG = 0.70
THRESHOLD_SHORT = 0.30
IMBALANCE_THRESHOLD = 0.30
MOMENTUM_THRESHOLD = 0.0005

from sklearn.ensemble import HistGradientBoostingClassifier

DEFAULT_PARAMS = {
    'max_depth':3,
    'learning_rate': 0.05,
    'max_iter': 200,
    'min_samples_leaf': 300,
    'l2_regularization': 0.2,
    'early_stopping': True,
    'validation_fraction': 0.15,
    'random_state': 42,
    'verbose': 0,
}

SUCCESS_CRITERIA = {
    'min_win_rate': 0.60,
    'max_drawdown': 0.05,
}
