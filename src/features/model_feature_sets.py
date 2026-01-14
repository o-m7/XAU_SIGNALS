#!/usr/bin/env python3
"""
Model Feature Sets - Single Source of Truth

Defines which features each model uses. This ensures training and live
use identical feature sets.

Models:
- Model 1 (HGB): 37 microstructure features
- Model 3 (CMF/MACD): 31 technical indicator features
- Model RF: 37 microstructure features (same as Model 1)
"""

from typing import List, Set

# =============================================================================
# MODEL 1 & RF: MICROSTRUCTURE FEATURES (37 features)
# =============================================================================

MODEL1_MICROSTRUCTURE_FEATURES = [
    # Microstructure (7) - Order flow and spread dynamics
    'micro_velocity',          # Quote update rate (ticks/minute)
    'micro_entropy',           # Shannon entropy of tick directions
    'effort_ratio',            # Price range per velocity unit
    'spread_zscore',           # Normalized bid-ask spread
    'synthetic_order_flow',    # Net tick pressure (FIXED - no volume multiplication)
    'flow_cvd_60',             # Cumulative Volume Delta over 60 minutes
    'flow_divergence',         # Price vs flow divergence

    # Returns (6)
    'ret_1',                   # 1-bar return
    'ret_3',                   # 3-bar return (for 5min model)
    'ret_5',                   # 5-bar return
    'ret_10',                  # 10-bar return
    'log_ret_1',               # Log return
    'ret_mean_5',              # 5-bar mean return

    # Volatility (5)
    'vol_10',                  # 10-bar volatility
    'vol_60',                  # 60-bar volatility
    'hl_range',                # High-low range
    'norm_range',              # Normalized range
    'ATR_14',                  # Average True Range (14-period)

    # Microstructure prices (4)
    'mid',                     # Bid-ask midpoint
    'spread',                  # Bid-ask spread
    'spread_pct',              # Spread as percentage
    'close_mid_diff',          # Close - mid difference

    # Candlestick structure (3)
    'body_pct',                # Body as % of range
    'upper_wick_pct',          # Upper wick as % of range
    'lower_wick_pct',          # Lower wick as % of range

    # Regime detection (6)
    'adx',                     # Average Directional Index
    'efficiency_ratio',        # Kaufman efficiency ratio
    'atr_percentile',          # ATR percentile rank
    'regime_trending',         # Is trending (boolean)
    'regime_ranging',          # Is ranging (boolean)
    'regime_volatile',         # Is volatile (boolean)

    # Additional momentum (2)
    'momentum_5',              # 5-bar momentum
    'momentum_10',             # 10-bar momentum
]

# =============================================================================
# MODEL 3: CMF/MACD TECHNICAL FEATURES (31 features)
# =============================================================================

MODEL3_TECHNICAL_FEATURES = [
    # Chaikin Money Flow (4)
    'cmf',                     # Base CMF (20-period)
    'cmf_momentum',            # Rate of change of CMF
    'cmf_zscore',              # Normalized CMF
    'cmf_trend',               # OLS slope of CMF

    # MACD (10)
    'macd',                    # Fast EMA - Slow EMA
    'macd_signal',             # Signal line (EMA of MACD)
    'macd_histogram',          # MACD - Signal
    'macd_momentum',           # Rate of change of MACD
    'macd_signal_momentum',    # Rate of change of signal
    'macd_above_signal',       # Binary flag
    'macd_cross_up',           # Bullish crossover
    'macd_cross_down',         # Bearish crossover
    'macd_hist_momentum',      # Histogram momentum
    'macd_zscore',             # Normalized MACD

    # RSI & Bollinger Bands (6)
    'rsi',                     # Relative Strength Index (14)
    'bb_middle',               # BB middle band
    'bb_upper',                # BB upper band
    'bb_lower',                # BB lower band
    'bb_width',                # BB width
    'bb_position',             # Price position in BB

    # Volume & Price Momentum (5)
    'volume_ratio',            # Volume / 20-bar SMA
    'price_momentum_5',        # 5-bar price momentum
    'price_momentum_20',       # 20-bar price momentum
    'price_vs_sma20',          # Price vs 20-bar SMA
    'price_vs_sma50',          # Price vs 50-bar SMA

    # Base features (6)
    'ATR_14',                  # Average True Range
    'ret_1',                   # 1-bar return
    'ret_5',                   # 5-bar return
    'vol_10',                  # 10-bar volatility
    'close',                   # Close price
    'volume',                  # Volume
]

# =============================================================================
# MODEL RF: RANDOM FOREST ENSEMBLE (same as Model 1)
# =============================================================================

MODEL_RF_FEATURES = MODEL1_MICROSTRUCTURE_FEATURES.copy()

# =============================================================================
# FEATURE SET MAPPINGS
# =============================================================================

def get_features_for_model(model_name: str) -> List[str]:
    """
    Get feature list for a specific model.

    Args:
        model_name: One of 'model1', 'model1_rebuilt', 'model3', 'model3_rebuilt',
                   'model_rf', 'model_rf_rebuilt', 'microstructure', 'cmf_macd'

    Returns:
        List of feature names required by the model

    Examples:
        >>> get_features_for_model('model1_rebuilt')
        ['micro_velocity', 'micro_entropy', ...]  # 37 features

        >>> get_features_for_model('model3_rebuilt')
        ['cmf', 'cmf_momentum', ...]  # 31 features
    """
    mapping = {
        # Model 1 variations
        'model1': MODEL1_MICROSTRUCTURE_FEATURES,
        'model1_rebuilt': MODEL1_MICROSTRUCTURE_FEATURES,
        'model1_hgb': MODEL1_MICROSTRUCTURE_FEATURES,
        'microstructure': MODEL1_MICROSTRUCTURE_FEATURES,

        # Model 3 variations
        'model3': MODEL3_TECHNICAL_FEATURES,
        'model3_rebuilt': MODEL3_TECHNICAL_FEATURES,
        'model3_cmf_macd': MODEL3_TECHNICAL_FEATURES,
        'cmf_macd': MODEL3_TECHNICAL_FEATURES,

        # Model RF variations
        'model_rf': MODEL_RF_FEATURES,
        'model_rf_rebuilt': MODEL_RF_FEATURES,
        'modelrf': MODEL_RF_FEATURES,
    }

    if model_name not in mapping:
        raise ValueError(
            f"Unknown model '{model_name}'. Valid options: {list(mapping.keys())}"
        )

    return mapping[model_name]


def get_all_feature_names() -> List[str]:
    """
    Get union of all features used by any model.

    Useful for:
    - Feature buffer initialization (must compute all features)
    - Feature engineering validation
    - Data quality checks

    Returns:
        Sorted list of unique feature names (no duplicates)
    """
    all_features: Set[str] = set()
    all_features.update(MODEL1_MICROSTRUCTURE_FEATURES)
    all_features.update(MODEL3_TECHNICAL_FEATURES)
    all_features.update(MODEL_RF_FEATURES)

    # Remove duplicates and sort
    return sorted(list(all_features))


def get_feature_count(model_name: str) -> int:
    """
    Get number of features for a model.

    Args:
        model_name: Model identifier

    Returns:
        Feature count
    """
    return len(get_features_for_model(model_name))


def validate_features(df, model_name: str, raise_on_missing: bool = True) -> List[str]:
    """
    Validate that DataFrame contains all required features for a model.

    Args:
        df: DataFrame with features
        model_name: Model identifier
        raise_on_missing: If True, raise ValueError on missing features

    Returns:
        List of missing features (empty if all present)

    Raises:
        ValueError: If features are missing and raise_on_missing=True
    """
    required_features = get_features_for_model(model_name)
    available_features = set(df.columns)
    missing_features = [f for f in required_features if f not in available_features]

    if missing_features and raise_on_missing:
        raise ValueError(
            f"Model '{model_name}' missing {len(missing_features)} features: "
            f"{missing_features[:10]}{'...' if len(missing_features) > 10 else ''}"
        )

    return missing_features


# =============================================================================
# FEATURE SET SUMMARIES
# =============================================================================

def print_feature_summary():
    """Print summary of all feature sets."""
    print("=" * 80)
    print("FEATURE SET SUMMARY")
    print("=" * 80)
    print()

    print(f"Model 1 (Microstructure):  {len(MODEL1_MICROSTRUCTURE_FEATURES)} features")
    print(f"Model 3 (CMF/MACD):        {len(MODEL3_TECHNICAL_FEATURES)} features")
    print(f"Model RF (Ensemble):       {len(MODEL_RF_FEATURES)} features")
    print()

    all_features = get_all_feature_names()
    print(f"Total unique features:     {len(all_features)} features")
    print()

    # Feature overlap
    m1_set = set(MODEL1_MICROSTRUCTURE_FEATURES)
    m3_set = set(MODEL3_TECHNICAL_FEATURES)
    overlap = m1_set & m3_set

    print(f"Feature overlap (M1 & M3): {len(overlap)} features")
    if overlap:
        print(f"  Shared: {sorted(overlap)[:5]}...")
    print()

    print("=" * 80)


if __name__ == "__main__":
    # Print feature summary
    print_feature_summary()

    # Test feature retrieval
    print("\nTesting feature retrieval:")
    print("-" * 80)
    for model in ['model1_rebuilt', 'model3_rebuilt', 'model_rf_rebuilt']:
        features = get_features_for_model(model)
        print(f"{model:20s}: {len(features)} features")

    print("\nAll features:", len(get_all_feature_names()))
