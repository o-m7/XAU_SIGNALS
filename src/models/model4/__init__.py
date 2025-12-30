"""Model 4: Trend-Following with Entry Timing"""

from .config import Model4Config
from .features import build_model4_features, get_model4_feature_columns
from .labels import add_directional_labels, add_trend_aligned_labels
from .train import run_training_pipeline
from .signal_engine import Model4SignalEngine, Signal, SignalResult

__all__ = [
    'Model4Config',
    'build_model4_features',
    'get_model4_feature_columns',
    'add_directional_labels',
    'add_trend_aligned_labels',
    'run_training_pipeline',
    'Model4SignalEngine',
    'Signal',
    'SignalResult',
]
