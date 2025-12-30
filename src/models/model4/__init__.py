"""
Model 4: VWAP Mean Reversion

Strategy: Trade mean reversion to session VWAP when price is stretched.
ML predicts: "Given price is stretched, will it revert before hitting stop?"
"""

from .config import Model4Config
from .vwap import calculate_session_vwap, calculate_vwap_zscore
from .regime import classify_regime, calculate_adx, calculate_atr
from .features import build_model4_features, get_model4_feature_columns
from .labels import add_reversion_labels, analyze_label_distribution
from .train import run_training_pipeline
from .signal_engine import Model4SignalEngine, Signal, SignalResult

__all__ = [
    # Config
    'Model4Config',

    # VWAP
    'calculate_session_vwap',
    'calculate_vwap_zscore',

    # Regime
    'classify_regime',
    'calculate_adx',
    'calculate_atr',

    # Features
    'build_model4_features',
    'get_model4_feature_columns',

    # Labels
    'add_reversion_labels',
    'analyze_label_distribution',

    # Training
    'run_training_pipeline',

    # Signal Engine
    'Model4SignalEngine',
    'Signal',
    'SignalResult',
]
