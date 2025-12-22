"""
Feature Engineering Module - Microstructure Features.

REQUIRES bid_size and ask_size for true microstructure features.
NO FALLBACKS - if sizes are missing, an error is raised.

Features computed:
- mid, spread, spread_pct
- imbalance = (bid_size - ask_size) / (bid_size + ask_size)
- microprice = (ask_price * bid_size + bid_price * ask_size) / (bid_size + ask_size)
- micro_dislocation = (microprice - mid) / mid
- sigma_short, sigma_long, sigma, sigma_slope
- spread regime features
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


# =============================================================================
# CONFIGURATION
# =============================================================================

VOL_LOOKBACK_SHORT = 30   # minutes for short-term volatility
VOL_LOOKBACK_LONG = 120   # minutes for long-term volatility
SPREAD_REGIME_LOOKBACK = 60  # minutes for spread median
SPREAD_MULT_DEFAULT = 1.2

# Maximum allowed NaN ratio for critical features
MAX_NAN_RATIO = 0.05


# =============================================================================
# VALIDATION
# =============================================================================

def validate_required_columns(df: pd.DataFrame, required: List[str]) -> None:
    """Validate that required columns exist and are not all NaN."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    all_nan = [c for c in required if df[c].isna().all()]
    if all_nan:
        raise ValueError(f"Columns are all NaN: {all_nan}")


def validate_feature_nan_ratio(
    df: pd.DataFrame,
    feature_cols: List[str],
    max_ratio: float = MAX_NAN_RATIO
) -> Tuple[bool, dict]:
    """
    Validate that features don't have too many NaN values.
    
    Returns:
        (passed: bool, nan_ratios: dict)
    """
    nan_ratios = {}
    failed_cols = []
    
    for col in feature_cols:
        if col in df.columns:
            ratio = df[col].isna().mean()
            nan_ratios[col] = ratio
            if ratio > max_ratio:
                failed_cols.append(col)
    
    passed = len(failed_cols) == 0
    
    return passed, nan_ratios, failed_cols


# =============================================================================
# CORE MICROSTRUCTURE FEATURES
# =============================================================================

def add_basic_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic price features: mid, spread, spread_pct.
    
    Requires: bid_price, ask_price
    """
    df = df.copy()
    
    validate_required_columns(df, ["bid_price", "ask_price"])
    
    df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    df["spread"] = df["ask_price"] - df["bid_price"]
    df["spread_pct"] = np.where(
        df["mid"] > 0,
        df["spread"] / df["mid"],
        np.nan
    )
    
    return df


def add_imbalance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add order book imbalance and microprice features.
    
    REQUIRES: bid_size, ask_size (NO FALLBACK)
    
    Features:
    - imbalance = (bid_size - ask_size) / (bid_size + ask_size)
    - microprice = (ask_price * bid_size + bid_price * ask_size) / (bid_size + ask_size)
    - micro_dislocation = (microprice - mid) / mid
    """
    df = df.copy()
    
    # Validate required columns
    validate_required_columns(df, ["bid_size", "ask_size", "bid_price", "ask_price"])
    
    # Ensure mid exists
    if "mid" not in df.columns:
        df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    
    # Compute denominator with safety for zero/NaN
    bid_size = df["bid_size"].values
    ask_size = df["ask_size"].values
    bid_price = df["bid_price"].values
    ask_price = df["ask_price"].values
    mid = df["mid"].values
    
    denom = bid_size + ask_size
    
    # Count zero/invalid denominators
    invalid_denom = (denom <= 0) | np.isnan(denom)
    n_invalid = invalid_denom.sum()
    
    if n_invalid > 0:
        print(f"  Warning: {n_invalid:,} rows have invalid bid_size + ask_size")
    
    # Imbalance
    imbalance = np.where(
        ~invalid_denom,
        (bid_size - ask_size) / denom,
        np.nan
    )
    df["imbalance"] = imbalance
    
    # Microprice (size-weighted fair price)
    microprice = np.where(
        ~invalid_denom,
        (ask_price * bid_size + bid_price * ask_size) / denom,
        np.nan
    )
    df["microprice"] = microprice
    
    # Micro dislocation
    df["micro_dislocation"] = np.where(
        (mid > 0) & (~np.isnan(microprice)),
        (microprice - mid) / mid,
        np.nan
    )
    
    return df


def add_volatility_features(
    df: pd.DataFrame,
    short_lookback: int = VOL_LOOKBACK_SHORT,
    long_lookback: int = VOL_LOOKBACK_LONG
) -> pd.DataFrame:
    """
    Add volatility features: sigma_short, sigma_long, sigma, sigma_slope.
    
    Uses LAGGED data only (shift(1) before rolling) to prevent look-ahead.
    """
    df = df.copy()
    
    if "mid" not in df.columns:
        validate_required_columns(df, ["bid_price", "ask_price"])
        df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    
    # Log returns
    df["log_ret"] = np.log(df["mid"] / df["mid"].shift(1))
    
    # LAGGED returns for volatility (no look-ahead)
    past_ret = df["log_ret"].shift(1)
    
    # Short-term volatility
    df["sigma_short"] = past_ret.rolling(
        window=short_lookback,
        min_periods=short_lookback // 2
    ).std()
    
    # Long-term volatility
    df["sigma_long"] = past_ret.rolling(
        window=long_lookback,
        min_periods=long_lookback // 2
    ).std()
    
    # Primary sigma
    df["sigma"] = df["sigma_short"]
    
    # Sigma slope: ratio of short to long - 1
    # Positive = volatility expanding, Negative = volatility contracting
    df["sigma_slope"] = np.where(
        (df["sigma_long"] > 0) & (~np.isnan(df["sigma_long"])),
        (df["sigma_short"] / df["sigma_long"]) - 1,
        np.nan
    )
    
    return df


def add_spread_regime_features(
    df: pd.DataFrame,
    lookback: int = SPREAD_REGIME_LOOKBACK,
    spread_mult: float = SPREAD_MULT_DEFAULT
) -> pd.DataFrame:
    """Add spread regime features."""
    df = df.copy()
    
    if "spread_pct" not in df.columns:
        if "mid" in df.columns and "spread" in df.columns:
            df["spread_pct"] = df["spread"] / df["mid"]
        else:
            raise ValueError("spread_pct not found and cannot be computed")
    
    # Rolling median of spread (LAGGED)
    past_spread = df["spread_pct"].shift(1)
    df["spread_med_60"] = past_spread.rolling(
        window=lookback,
        min_periods=lookback // 2
    ).median()
    
    # Spread regime OK flag
    df["spread_regime_ok"] = (
        df["spread_pct"] < (spread_mult * df["spread_med_60"])
    ).astype(int)
    
    return df


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trading session features."""
    df = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    hour = df.index.hour
    
    df["is_london"] = ((hour >= 8) & (hour < 16)).astype(int)
    df["is_ny"] = ((hour >= 13) & (hour < 22)).astype(int)
    df["is_overlap"] = ((hour >= 13) & (hour < 16)).astype(int)
    df["session_quality"] = df["is_overlap"] * 2 + df["is_london"] + df["is_ny"]
    
    df["hour_of_day"] = hour
    df["day_of_week"] = df.index.dayofweek
    
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum features (lagged to prevent look-ahead)."""
    df = df.copy()
    
    if "mid" not in df.columns:
        df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    
    past_mid = df["mid"].shift(1)
    
    df["momentum_5"] = past_mid.pct_change(5)
    df["momentum_15"] = past_mid.pct_change(15)
    df["momentum_30"] = past_mid.pct_change(30)
    
    return df


# =============================================================================
# MASTER BUILD FUNCTION
# =============================================================================

def build_feature_matrix(
    df: pd.DataFrame,
    require_sizes: bool = True,
    vol_short: int = VOL_LOOKBACK_SHORT,
    vol_long: int = VOL_LOOKBACK_LONG,
    spread_lookback: int = SPREAD_REGIME_LOOKBACK,
    spread_mult: float = SPREAD_MULT_DEFAULT,
    max_nan_ratio: float = MAX_NAN_RATIO
) -> pd.DataFrame:
    """
    Build complete feature matrix.
    
    REQUIRES bid_size and ask_size if require_sizes=True.
    NO FALLBACKS - raises error if sizes missing.
    
    Args:
        df: DataFrame with OHLCV + quotes data
        require_sizes: If True, require bid_size/ask_size
        vol_short: Short volatility lookback
        vol_long: Long volatility lookback
        spread_lookback: Spread regime lookback
        spread_mult: Spread regime multiplier
        max_nan_ratio: Maximum allowed NaN ratio for critical features
        
    Returns:
        DataFrame with all features
        
    Raises:
        ValueError: If required columns missing or too many NaN
    """
    print("Building feature matrix...")
    
    # Step 1: Basic price features
    print("  Adding price features...")
    df = add_basic_price_features(df)
    
    # Step 2: Imbalance and microprice (REQUIRED if require_sizes)
    if require_sizes:
        print("  Adding imbalance/microprice (requires bid_size, ask_size)...")
        df = add_imbalance_features(df)
    else:
        # Still try to add if columns exist
        if "bid_size" in df.columns and "ask_size" in df.columns:
            print("  Adding imbalance/microprice (sizes available)...")
            df = add_imbalance_features(df)
        else:
            print("  Skipping imbalance/microprice (no size data)")
            df["imbalance"] = np.nan
            df["microprice"] = np.nan
            df["micro_dislocation"] = np.nan
    
    # Step 3: Volatility features
    print("  Adding volatility features...")
    df = add_volatility_features(df, vol_short, vol_long)
    
    # Step 4: Spread regime
    print("  Adding spread regime features...")
    df = add_spread_regime_features(df, spread_lookback, spread_mult)
    
    # Step 5: Session features
    print("  Adding session features...")
    df = add_session_features(df)
    
    # Step 6: Momentum features
    print("  Adding momentum features...")
    df = add_momentum_features(df)
    
    # Step 7: Validate NaN ratios for critical features
    critical_features = ["mid", "spread_pct", "sigma_slope"]
    if require_sizes:
        critical_features.extend(["imbalance", "microprice"])
    
    print("  Validating feature NaN ratios...")
    passed, nan_ratios, failed_cols = validate_feature_nan_ratio(
        df, critical_features, max_nan_ratio
    )
    
    for col, ratio in nan_ratios.items():
        status = "✓" if ratio <= max_nan_ratio else "❌"
        print(f"    {status} {col}: {ratio*100:.1f}% NaN")
    
    if not passed:
        raise ValueError(
            f"Feature NaN ratio too high (> {max_nan_ratio*100:.0f}%) for: {failed_cols}. "
            f"Check data quality or adjust max_nan_ratio."
        )
    
    print(f"  ✓ Feature matrix complete: {len(df):,} rows, {len(df.columns)} columns")
    
    return df


# =============================================================================
# FEATURE LISTS
# =============================================================================

def get_microstructure_features() -> List[str]:
    """Get list of microstructure features for model training."""
    return [
        "sigma_short", "sigma_long", "sigma", "sigma_slope",
        "spread_pct", "spread_med_60", "spread_regime_ok",
        "imbalance", "micro_dislocation",
        "momentum_5", "momentum_15", "momentum_30",
        "is_london", "is_ny", "is_overlap", "session_quality",
        "hour_of_day", "day_of_week"
    ]


def get_feature_info() -> dict:
    """Get information about features."""
    return {
        "price": ["mid", "spread", "spread_pct"],
        "imbalance": ["imbalance", "microprice", "micro_dislocation"],
        "volatility": ["sigma_short", "sigma_long", "sigma", "sigma_slope"],
        "spread_regime": ["spread_med_60", "spread_regime_ok"],
        "session": ["is_london", "is_ny", "is_overlap", "session_quality"],
        "momentum": ["momentum_5", "momentum_15", "momentum_30"],
    }
