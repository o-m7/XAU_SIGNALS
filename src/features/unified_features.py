#!/usr/bin/env python3
"""
Unified Feature Engineering - Single Source of Truth

This module is the MASTER entry point for feature engineering.
Used by BOTH training and live trading to ensure perfect parity.

CRITICAL: This module must be used identically in both environments!
- Training: rebuild_from_scratch.py → calls build_features()
- Live: feature_buffer.py → calls build_features()

Features generated (~60+ features):
- Microstructure (7): Order flow, spread dynamics
- Price Action (35): Returns, volatility, candlesticks, multi-timeframe, time
- Technical (25): CMF, MACD, RSI, Bollinger Bands
- Regime (6): ADX, efficiency ratio, volatility regime

All features use NO lookahead bias (only past/current data).
"""

import pandas as pd
import numpy as np
from typing import Optional, List
import warnings

# Import all feature modules
from .microstructure import compute_microstructure_features
from .price_action import compute_price_action_features
from .technical import compute_technical_features
from .regime import compute_regime_features
from .model_feature_sets import get_features_for_model, get_all_feature_names


# =============================================================================
# QUOTE MERGING UTILITY
# =============================================================================

def merge_quotes_to_bars(
    bars: pd.DataFrame,
    quotes: Optional[pd.DataFrame] = None,
    tolerance: str = "120s"
) -> pd.DataFrame:
    """
    Merge top-of-book quotes into bar data.

    Args:
        bars: Bar data with timestamp index, OHLCV columns
        quotes: Quote data with timestamp index, bid_price, ask_price (optional)
        tolerance: Maximum time tolerance for forward-fill (default 120s)

    Returns:
        Merged DataFrame with quote columns added to bars

    Notes:
        If quotes is None, returns bars unchanged.
    """
    if quotes is None or quotes.empty:
        return bars

    # Ensure both have datetime index
    if not isinstance(bars.index, pd.DatetimeIndex):
        if "timestamp" in bars.columns:
            bars = bars.set_index("timestamp")

    if not isinstance(quotes.index, pd.DatetimeIndex):
        if "timestamp" in quotes.columns:
            quotes = quotes.set_index("timestamp")

    # Sort both by timestamp
    bars = bars.sort_index()
    quotes = quotes.sort_index()

    # Reset index for merge_asof
    bars_reset = bars.reset_index()
    quotes_reset = quotes[["bid_price", "ask_price"]].reset_index()

    # Rename timestamp columns to match
    ts_col = bars_reset.columns[0]
    quotes_reset = quotes_reset.rename(columns={quotes_reset.columns[0]: ts_col})

    # Merge with tolerance
    merged = pd.merge_asof(
        bars_reset.sort_values(ts_col),
        quotes_reset.sort_values(ts_col),
        on=ts_col,
        direction="backward",
        tolerance=pd.Timedelta(tolerance)
    )

    # Set timestamp back as index
    merged = merged.set_index(ts_col)

    return merged


# =============================================================================
# MASTER FEATURE BUILDER
# =============================================================================

def build_features(
    bars: pd.DataFrame,
    quotes: Optional[pd.DataFrame] = None,
    feature_set: str = "all"
) -> pd.DataFrame:
    """
    Build features for training or live inference.

    This is the SINGLE SOURCE OF TRUTH for feature engineering.
    Uses SAME code for both training and live to ensure perfect parity.

    Args:
        bars: Bar data with OHLCV columns (timestamp index)
        quotes: Optional quote data with bid_price, ask_price (timestamp index)
        feature_set: Which features to return:
            - "all": All features (union of all models)
            - "model1", "model1_rebuilt": Model 1 features (37 microstructure)
            - "model3", "model3_rebuilt": Model 3 features (31 CMF/MACD)
            - "model_rf", "model_rf_rebuilt": Model RF features (37 microstructure)
            - "microstructure": Microstructure features only
            - "cmf_macd": Technical features only

    Returns:
        DataFrame with requested features

    Raises:
        ValueError: If bars missing required columns or feature_set unknown

    Notes:
        - All features use NO lookahead bias
        - Microstructure features require quotes (bid/ask)
        - If quotes missing, microstructure features will be NaN
        - Features are computed in this order:
          1. Merge quotes (if available)
          2. Microstructure (from quotes)
          3. Price action (returns, volatility, candlesticks, MTF, time)
          4. Technical indicators (CMF, MACD, RSI, BB)
          5. Regime detection (ADX, efficiency ratio)
          6. Filter to requested feature set

    Examples:
        >>> # Training: Build all features
        >>> df = build_features(bars, quotes, feature_set="all")
        >>>
        >>> # Live: Build features for specific model
        >>> df = build_features(bars, quotes, feature_set="model1_rebuilt")
    """
    # Validate inputs
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [c for c in required_cols if c not in bars.columns]
    if missing_cols:
        raise ValueError(f"Bars missing required columns: {missing_cols}")

    # Ensure datetime index
    if not isinstance(bars.index, pd.DatetimeIndex):
        if "timestamp" in bars.columns:
            bars = bars.set_index("timestamp")
        else:
            raise ValueError("Bars must have DatetimeIndex or 'timestamp' column")

    print("="*80)
    print("UNIFIED FEATURE ENGINEERING")
    print("="*80)
    print(f"Input bars: {len(bars):,} rows")
    if quotes is not None and not quotes.empty:
        print(f"Input quotes: {len(quotes):,} rows")
    else:
        print("Input quotes: None (microstructure features will be NaN)")
    print(f"Feature set: {feature_set}")
    print()

    # Copy to avoid modifying input
    df = bars.copy()

    # Step 1: Merge quotes if available
    if quotes is not None and not quotes.empty:
        print("Step 1: Merging quotes into bars...")
        df = merge_quotes_to_bars(df, quotes, tolerance="120s")
        print(f"  ✓ Quotes merged: {len(df.columns)} columns")
    else:
        print("Step 1: Skipping quote merge (no quotes provided)")

    # Step 2: Compute microstructure features (requires quotes)
    print("Step 2: Computing microstructure features...")
    if quotes is not None and not quotes.empty:
        micro_features = compute_microstructure_features(df, quotes)
        # Join microstructure features
        df = df.join(micro_features, how='left')
        print(f"  ✓ Microstructure complete: {len(micro_features.columns)} features")
    else:
        print("  ⚠️ Skipping microstructure (no quotes)")

    # Step 3: Compute price action features
    print("Step 3: Computing price action features...")
    df = compute_price_action_features(df)

    # Step 4: Compute technical indicators
    print("Step 4: Computing technical indicators...")
    df = compute_technical_features(df)

    # Step 5: Compute regime features
    print("Step 5: Computing regime features...")
    df = compute_regime_features(df)

    # Step 6: Compute interaction features
    print("Step 6: Computing interaction features...")
    df = _compute_interaction_features(df)

    # Step 7: Filter to requested feature set
    if feature_set != "all":
        print(f"Step 7: Filtering to {feature_set} features...")
        try:
            feature_cols = get_features_for_model(feature_set)
            # Keep only features that exist in df
            available_features = [f for f in feature_cols if f in df.columns]
            missing_features = [f for f in feature_cols if f not in df.columns]

            if missing_features:
                warnings.warn(
                    f"Feature set '{feature_set}' missing {len(missing_features)} features: "
                    f"{missing_features[:5]}{'...' if len(missing_features) > 5 else ''}"
                )

            # Filter DataFrame
            df = df[available_features]
            print(f"  ✓ Filtered to {len(available_features)} features")
        except ValueError as e:
            print(f"  ⚠️ Unknown feature set '{feature_set}', returning all features")
            print(f"  Error: {e}")

    # Final summary
    print()
    print("="*80)
    print(f"FEATURE ENGINEERING COMPLETE: {len(df.columns)} features, {len(df)} rows")
    print("="*80)
    print()

    return df


# =============================================================================
# INTERACTION FEATURES
# =============================================================================

def _compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute feature interactions (cross-terms).

    Interactions capture relationships between features:
    - wick_flow_interaction: Wick size * order flow
    - body_flow_interaction: Body size * order flow
    - wick_volume_interaction: Wick size * volume

    Args:
        df: DataFrame with base features

    Returns:
        DataFrame with interaction features added

    Notes:
        Only computes interactions if base features exist.
    """
    df = df.copy()

    # Wick * Flow interaction
    if 'upper_wick_pct' in df.columns and 'synthetic_order_flow' in df.columns:
        df['wick_flow_interaction'] = df['upper_wick_pct'] * df['synthetic_order_flow']
    else:
        df['wick_flow_interaction'] = 0

    # Body * Flow interaction
    if 'body_pct' in df.columns and 'synthetic_order_flow' in df.columns:
        df['body_flow_interaction'] = df['body_pct'] * df['synthetic_order_flow']
    else:
        df['body_flow_interaction'] = 0

    # Wick * Volume interaction
    if 'upper_wick_pct' in df.columns and 'volume' in df.columns:
        vol_norm = df['volume'] / (df['volume'].rolling(20, min_periods=1).mean() + 1e-8)
        df['wick_volume_interaction'] = df['upper_wick_pct'] * vol_norm
    else:
        df['wick_volume_interaction'] = 0

    # Fill NaNs
    df['wick_flow_interaction'] = df['wick_flow_interaction'].fillna(0)
    df['body_flow_interaction'] = df['body_flow_interaction'].fillna(0)
    df['wick_volume_interaction'] = df['wick_volume_interaction'].fillna(0)

    return df


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_features(
    df: pd.DataFrame,
    model_name: str,
    raise_on_missing: bool = True,
    raise_on_nan: bool = False,
    nan_threshold: float = 0.5
) -> dict:
    """
    Validate features for a specific model.

    Args:
        df: DataFrame with features
        model_name: Model identifier
        raise_on_missing: Raise error if features missing
        raise_on_nan: Raise error if NaN ratio exceeds threshold
        nan_threshold: Max allowed NaN ratio (0-1)

    Returns:
        Dict with validation results:
            - missing_features: List of missing features
            - nan_features: Dict of {feature: nan_ratio} for high NaN features
            - valid: Bool indicating if validation passed

    Raises:
        ValueError: If validation fails and raise_on_* flags set
    """
    from .model_feature_sets import validate_features as validate_model_features

    # Check for missing features
    missing_features = validate_model_features(df, model_name, raise_on_missing=False)

    # Check for high NaN ratios
    required_features = get_features_for_model(model_name)
    available_features = [f for f in required_features if f in df.columns]

    nan_features = {}
    for feat in available_features:
        nan_ratio = df[feat].isna().mean()
        if nan_ratio > nan_threshold:
            nan_features[feat] = nan_ratio

    # Validation result
    valid = (len(missing_features) == 0) and (len(nan_features) == 0)

    # Raise errors if requested
    if raise_on_missing and missing_features:
        raise ValueError(
            f"Model '{model_name}' missing {len(missing_features)} features: {missing_features}"
        )

    if raise_on_nan and nan_features:
        raise ValueError(
            f"Model '{model_name}' has {len(nan_features)} features with NaN > {nan_threshold}: "
            f"{list(nan_features.keys())}"
        )

    return {
        'missing_features': missing_features,
        'nan_features': nan_features,
        'valid': valid
    }


# =============================================================================
# UTILITY: GET FEATURE SUMMARY
# =============================================================================

def get_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics for all features.

    Args:
        df: DataFrame with features

    Returns:
        DataFrame with summary statistics (count, mean, std, min, max, nan_ratio)
    """
    summary = df.describe().T
    summary['nan_ratio'] = df.isna().mean()
    summary['nan_count'] = df.isna().sum()
    summary = summary[['count', 'mean', 'std', 'min', 'max', 'nan_ratio', 'nan_count']]
    return summary.sort_values('nan_ratio', ascending=False)


if __name__ == "__main__":
    print("=" * 80)
    print("UNIFIED FEATURE ENGINEERING MODULE")
    print("=" * 80)
    print()
    print("This is the MASTER module for feature engineering.")
    print("Used by BOTH training and live to ensure perfect parity.")
    print()
    print("Usage:")
    print("  from src.features.unified_features import build_features")
    print()
    print("  # Training")
    print("  df = build_features(bars, quotes, feature_set='all')")
    print()
    print("  # Live (Model 1)")
    print("  df = build_features(bars, quotes, feature_set='model1_rebuilt')")
    print()
    print("Available feature sets:")
    print("  - 'all': All features (~60+)")
    print("  - 'model1_rebuilt': Model 1 microstructure (37 features)")
    print("  - 'model3_rebuilt': Model 3 CMF/MACD (31 features)")
    print("  - 'model_rf_rebuilt': Model RF ensemble (37 features)")
    print()
    print("=" * 80)
