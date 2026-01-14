"""
Model 5: Statistical Mean Reversion (XAUUSD 15M)

Pure statistical approach - no TA indicators, no narratives.
Only mathematically defined features with testable hypotheses.
"""

from .config import Model5Config
from .features import build_all_features, get_feature_columns
from .signals import Model5SignalEngine, Signal, SignalResult
from .validation import run_all_validations
from .labels import add_labels
from .backtest import run_backtest, BacktestResult, Trade

__all__ = [
    'Model5Config',
    'build_all_features',
    'get_feature_columns',
    'Model5SignalEngine',
    'Signal',
    'SignalResult',
    'run_all_validations',
    'add_labels',
    'run_backtest',
    'BacktestResult',
    'Trade',
]

