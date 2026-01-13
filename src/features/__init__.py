#!/usr/bin/env python3
"""
Feature Engineering Package - Single Source of Truth

This package provides unified feature engineering for XAUUSD trading models.
All features are computed identically for training and live trading.

Main Entry Point:
    from src.features import build_features

Submodules:
    - unified_features: Master orchestrator (USE THIS!)
    - microstructure: Order flow and spread dynamics (7 features)
    - price_action: Returns, volatility, candlesticks, MTF, time (35 features)
    - technical: CMF, MACD, RSI, Bollinger Bands (25 features)
    - regime: ADX, efficiency ratio, volatility regime (6 features)
    - model_feature_sets: Feature registry (what each model needs)

Critical: ALWAYS use build_features() for consistency!
"""

# Main entry point
from .unified_features import (
    build_features,
    validate_features,
    get_feature_summary,
    merge_quotes_to_bars
)

# Feature set registry
from .model_feature_sets import (
    get_features_for_model,
    get_all_feature_names,
    get_feature_count,
    MODEL1_MICROSTRUCTURE_FEATURES,
    MODEL3_TECHNICAL_FEATURES,
    MODEL_RF_FEATURES
)

# Individual feature modules (for advanced use)
from .microstructure import compute_microstructure_features
from .price_action import compute_price_action_features
from .technical import compute_technical_features
from .regime import compute_regime_features

__all__ = [
    # Main entry point
    'build_features',
    'validate_features',
    'get_feature_summary',
    'merge_quotes_to_bars',

    # Feature sets
    'get_features_for_model',
    'get_all_feature_names',
    'get_feature_count',
    'MODEL1_MICROSTRUCTURE_FEATURES',
    'MODEL3_TECHNICAL_FEATURES',
    'MODEL_RF_FEATURES',

    # Individual modules
    'compute_microstructure_features',
    'compute_price_action_features',
    'compute_technical_features',
    'compute_regime_features',
]

__version__ = '1.0.0'
__author__ = 'XAUUSD Trading System'
__description__ = 'Unified feature engineering with training/live parity'
