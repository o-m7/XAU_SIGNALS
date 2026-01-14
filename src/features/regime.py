#!/usr/bin/env python3
"""
Regime Detection Features - ADX, Efficiency Ratio, Volatility

Generates features to identify market regime (trending vs ranging vs volatile).

Features generated (6):
- adx: Average Directional Index (trend strength)
- adx_slope: Rate of change in ADX (regime transitions)
- efficiency_ratio: Kaufman Efficiency Ratio (trend quality)
- atr_percentile: ATR percentile rank (volatility regime)
- regime_trending: Binary flag (ADX > 25, efficient trends)
- regime_ranging: Binary flag (ADX <= 25, low volatility)
- regime_volatile: Binary flag (ATR in top 25%)

All features use NO lookahead bias (only past/current data).
"""

import pandas as pd
import numpy as np
from typing import Optional


# =============================================================================
# ADX (AVERAGE DIRECTIONAL INDEX)
# =============================================================================

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute Average Directional Index (ADX).

    ADX measures trend strength (0-100):
    - ADX < 20: Weak trend (ranging)
    - ADX 20-25: Developing trend
    - ADX > 25: Strong trend
    - ADX > 40: Very strong trend

    Formula:
        1. True Range (TR) = max(high-low, |high-prev_close|, |low-prev_close|)
        2. Directional Movement: +DM = high - prev_high, -DM = prev_low - low
        3. +DI = 100 * EMA(+DM) / EMA(TR)
        4. -DI = 100 * EMA(-DM) / EMA(TR)
        5. DX = 100 * |+DI - -DI| / (+DI + -DI)
        6. ADX = EMA(DX)

    Args:
        df: DataFrame with 'high', 'low', 'close'
        period: Smoothing period (default 14)

    Returns:
        DataFrame with 'adx' column

    Notes:
        Uses EMA (exponential moving average) for smoother results.
    """
    df = df.copy()

    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    # Only count directional move if it's greater than opposite move
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed indicators (EMA)
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / (atr + 1e-8)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / (atr + 1e-8)

    # Directional Index (DX)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)

    # ADX (smoothed DX)
    df['adx'] = dx.ewm(span=period, adjust=False).mean()

    # Fill NaNs
    df['adx'] = df['adx'].fillna(0)

    return df


# =============================================================================
# ADX SLOPE (REGIME TRANSITIONS)
# =============================================================================

def compute_adx_slope(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Compute ADX slope (rate of change).

    ADX slope detects regime transitions:
    - Positive slope: Trend strengthening
    - Negative slope: Trend weakening (potential reversal)
    - Large positive spike: Breakout
    - Large negative spike: Trend exhaustion

    Args:
        df: DataFrame with 'adx' column
        lookback: Bars to measure slope over (default 5)

    Returns:
        DataFrame with 'adx_slope' column
    """
    df = df.copy()

    if 'adx' not in df.columns:
        df = compute_adx(df)

    # Slope = change over lookback period
    df['adx_slope'] = df['adx'].diff(lookback)
    df['adx_slope'] = df['adx_slope'].fillna(0)

    return df


# =============================================================================
# EFFICIENCY RATIO (KAUFMAN)
# =============================================================================

def compute_efficiency_ratio(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Compute Kaufman Efficiency Ratio.

    Efficiency Ratio measures trend quality (0-1):
    - ER close to 1: Strong, efficient trend (price moves in one direction)
    - ER close to 0: Noisy, choppy market (price oscillates)
    - ER > 0.3: Trending market
    - ER < 0.3: Ranging market

    Formula:
        ER = |Net Change| / Sum(Absolute Changes)
        Net Change = |close[t] - close[t-period]|
        Sum = sum(|close[i] - close[i-1]|) for i in period

    Args:
        df: DataFrame with 'close'
        period: Lookback period (default 20)

    Returns:
        DataFrame with 'efficiency_ratio' column

    Notes:
        ER = 1.0 means price moved perfectly in one direction.
        ER = 0.0 means price oscillated without net progress.
    """
    df = df.copy()

    close = df['close']

    # Net change over period (how far price actually moved)
    net_change = (close - close.shift(period)).abs()

    # Sum of absolute changes (total distance traveled)
    sum_changes = close.diff().abs().rolling(period, min_periods=1).sum()

    # Efficiency = net / total
    df['efficiency_ratio'] = net_change / (sum_changes + 1e-8)
    df['efficiency_ratio'] = df['efficiency_ratio'].fillna(0)

    return df


# =============================================================================
# ATR PERCENTILE (VOLATILITY REGIME)
# =============================================================================

def compute_atr_percentile(df: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
    """
    Compute ATR percentile rank (volatility regime).

    ATR Percentile identifies volatility regime:
    - Percentile < 0.25: Low volatility (ranging)
    - Percentile 0.25-0.75: Normal volatility
    - Percentile > 0.75: High volatility (volatile regime)

    Args:
        df: DataFrame with 'ATR_14' column (or will compute if missing)
        lookback: Bars for percentile calculation (default 100)

    Returns:
        DataFrame with 'atr_percentile' column

    Notes:
        Uses rolling percentile rank (0-1 scale).
        High percentile = ATR is high relative to recent history.
    """
    df = df.copy()

    # Ensure ATR_14 exists
    if 'ATR_14' not in df.columns:
        print("    ⚠️ ATR_14 not found, computing...")
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['ATR_14'] = true_range.rolling(14, min_periods=7).mean()

    # Percentile rank (rolling)
    # NOTE: When raw=False, x is already a Series, so no need to wrap in pd.Series()
    df['atr_percentile'] = df['ATR_14'].rolling(lookback, min_periods=10).apply(
        lambda x: x.rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
        raw=False
    )

    # Fill NaNs with median (0.5 = normal volatility)
    df['atr_percentile'] = df['atr_percentile'].fillna(0.5)

    return df


# =============================================================================
# REGIME CLASSIFICATION (BINARY FLAGS)
# =============================================================================

def compute_regime_flags(
    df: pd.DataFrame,
    adx_trending_threshold: float = 25.0,
    efficiency_threshold: float = 0.3,
    atr_volatile_threshold: float = 0.75
) -> pd.DataFrame:
    """
    Compute regime classification flags.

    Regimes:
    - Trending: ADX > 25 AND efficiency > 0.3 (strong, efficient trend)
    - Ranging: ADX <= 25 AND ATR percentile <= 0.75 (sideways, normal vol)
    - Volatile: ATR percentile > 0.75 (high volatility, risky)

    Args:
        df: DataFrame with 'adx', 'efficiency_ratio', 'atr_percentile'
        adx_trending_threshold: ADX level for trending (default 25)
        efficiency_threshold: Efficiency for trending (default 0.3)
        atr_volatile_threshold: ATR percentile for volatile (default 0.75)

    Returns:
        DataFrame with 'regime_trending', 'regime_ranging', 'regime_volatile' columns

    Notes:
        Regimes can overlap (e.g., both trending and volatile).
    """
    df = df.copy()

    # Ensure all required features exist
    if 'adx' not in df.columns:
        df = compute_adx(df)
    if 'efficiency_ratio' not in df.columns:
        df = compute_efficiency_ratio(df)
    if 'atr_percentile' not in df.columns:
        df = compute_atr_percentile(df)

    # Regime flags
    df['regime_trending'] = (
        (df['adx'] > adx_trending_threshold) &
        (df['efficiency_ratio'] > efficiency_threshold)
    ).astype(int)

    df['regime_ranging'] = (
        (df['adx'] <= adx_trending_threshold) &
        (df['atr_percentile'] <= atr_volatile_threshold)
    ).astype(int)

    df['regime_volatile'] = (
        df['atr_percentile'] > atr_volatile_threshold
    ).astype(int)

    return df


# =============================================================================
# MASTER REGIME FEATURE BUILDER
# =============================================================================

def compute_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all regime detection features.

    Features (6):
    - adx: Average Directional Index (trend strength)
    - adx_slope: ADX rate of change (regime transitions)
    - efficiency_ratio: Kaufman Efficiency Ratio (trend quality)
    - atr_percentile: ATR percentile (volatility regime)
    - regime_trending: Binary flag for trending
    - regime_ranging: Binary flag for ranging
    - regime_volatile: Binary flag for volatile

    Args:
        df: DataFrame with OHLC data

    Returns:
        DataFrame with all regime features added

    Notes:
        - All features use NO lookahead bias
        - Regimes help filter trades (e.g., trend-following in trending regime)
    """
    print("  Computing regime detection features...")

    # ADX
    df = compute_adx(df, period=14)

    # ADX Slope
    df = compute_adx_slope(df, lookback=5)

    # Efficiency Ratio
    df = compute_efficiency_ratio(df, period=20)

    # ATR Percentile
    df = compute_atr_percentile(df, lookback=100)

    # Regime Flags
    df = compute_regime_flags(
        df,
        adx_trending_threshold=25.0,
        efficiency_threshold=0.3,
        atr_volatile_threshold=0.75
    )

    print(f"    ✓ Regime features complete")
    return df


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    DEPRECATED: Use compute_regime_features() instead.

    Legacy function name for backward compatibility.
    """
    import warnings
    warnings.warn(
        "add_regime_features() is deprecated. Use compute_regime_features() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return compute_regime_features(df)


if __name__ == "__main__":
    print("=" * 80)
    print("REGIME DETECTION MODULE")
    print("=" * 80)
    print()
    print("Features generated (6):")
    print("  1. adx:              Average Directional Index (trend strength)")
    print("  2. adx_slope:        ADX rate of change (regime transitions)")
    print("  3. efficiency_ratio: Kaufman Efficiency Ratio (trend quality)")
    print("  4. atr_percentile:   ATR percentile (volatility regime)")
    print("  5. regime_trending:  Binary flag (ADX > 25, efficient)")
    print("  6. regime_ranging:   Binary flag (ADX <= 25, normal vol)")
    print("  7. regime_volatile:  Binary flag (ATR in top 25%)")
    print()
    print("Regime classification:")
    print("  - Trending: ADX > 25 AND efficiency > 0.3")
    print("  - Ranging:  ADX <= 25 AND ATR percentile <= 0.75")
    print("  - Volatile: ATR percentile > 0.75")
    print()
    print("Use cases:")
    print("  - Filter trend-following trades to trending regime")
    print("  - Filter mean-reversion trades to ranging regime")
    print("  - Reduce position size in volatile regime")
    print("=" * 80)
