#!/usr/bin/env python3
"""
Price Action Features

Generates features related to price movement, volatility, volume, candlestick structure,
and multi-timeframe context.

Features generated (~35 features):
- Returns (8): ret_1, ret_3, ret_5, ret_10, log_ret_1, ret_mean_5/20/60
- Volatility (5): vol_10, vol_60, hl_range, norm_range, ATR_14
- Microstructure (8): mid, spread, spread_pct, mid_ret_1, mid_vol_20, mid_slope_10,
                      close_mid_diff, close_mid_spread_ratio
- Volume (4): vol_change, vol_rel_20, vol_zscore_20, dollar_vol
- Candlestick (8): body, range, upper_wick, lower_wick, body_pct, upper_wick_pct,
                   lower_wick_pct, is_bull
- Multi-Timeframe (4): ma_fast_15, ma_slow_60, ma_ratio, ma_slope_60
- Time (6): minute_sin, minute_cos, day_of_week, is_asia, is_europe, is_us

All features use NO lookahead bias (only past/current data).
"""

import pandas as pd
import numpy as np
from typing import Optional


# =============================================================================
# RETURN FEATURES
# =============================================================================

def compute_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute price return features.

    Features (8):
    - ret_1, ret_3, ret_5, ret_10: Simple returns
    - log_ret_1: Log return (better for statistical properties)
    - ret_mean_5, ret_mean_20, ret_mean_60: Rolling mean returns

    Args:
        df: DataFrame with 'close' column

    Returns:
        DataFrame with return features added
    """
    df = df.copy()

    # Simple returns at various lags
    for n in [1, 3, 5, 10]:
        df[f"ret_{n}"] = df["close"].pct_change(n)

    # Log return (more stable for statistics)
    df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))

    # Rolling mean returns (smoothed momentum)
    for n in [5, 20, 60]:
        df[f"ret_mean_{n}"] = df["log_ret_1"].rolling(n, min_periods=n//2).mean()

    return df


# =============================================================================
# VOLATILITY FEATURES
# =============================================================================

def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volatility and range features.

    Features (5):
    - vol_10, vol_60: Rolling std of log returns
    - hl_range: High - Low (bar range)
    - norm_range: (High - Low) / Close (normalized)
    - ATR_14: Average True Range (14-period)

    Args:
        df: DataFrame with 'high', 'low', 'close' columns

    Returns:
        DataFrame with volatility features added
    """
    df = df.copy()

    # Ensure log_ret_1 exists
    if "log_ret_1" not in df.columns:
        df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))

    # Rolling volatility
    df["vol_10"] = df["log_ret_1"].rolling(10, min_periods=5).std()
    df["vol_60"] = df["log_ret_1"].rolling(60, min_periods=30).std()

    # High-Low range
    df["hl_range"] = df["high"] - df["low"]
    df["norm_range"] = df["hl_range"] / df["close"]

    # True Range for ATR
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR_14"] = true_range.rolling(14, min_periods=7).mean()

    return df


# =============================================================================
# MICROSTRUCTURE FEATURES (BID/ASK)
# =============================================================================

def compute_quote_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute quote microstructure features from bid/ask prices.

    Features (8):
    - mid: Bid-ask midpoint
    - spread: Bid-ask spread
    - spread_pct: Spread as percentage of mid
    - mid_ret_1: Mid price return
    - mid_vol_20: Mid price volatility
    - mid_slope_10: Rolling OLS slope of mid
    - close_mid_diff: Close - Mid (where close traded)
    - close_mid_spread_ratio: (Close - Mid) / Spread (normalized position)

    Args:
        df: DataFrame with 'bid_price', 'ask_price', 'close' columns

    Returns:
        DataFrame with microstructure features added

    Notes:
        If bid_price/ask_price not available, returns df unchanged with warning.
    """
    df = df.copy()

    # Check for required columns
    if "bid_price" not in df.columns or "ask_price" not in df.columns:
        print("    ⚠️ bid_price/ask_price not found, skipping quote microstructure features")
        return df

    # Mid price and spread
    df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    df["spread"] = df["ask_price"] - df["bid_price"]
    df["spread_pct"] = df["spread"] / df["mid"]

    # Mid returns and volatility
    df["mid_ret_1"] = np.log(df["mid"] / df["mid"].shift(1))
    df["mid_vol_20"] = df["mid_ret_1"].rolling(20, min_periods=10).std()

    # Rolling OLS slope of mid over last 10 bars
    # Using vectorized approach: slope = cov(x, y) / var(x)
    window = 10
    x = np.arange(window)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def rolling_slope(series: pd.Series, w: int) -> pd.Series:
        """Compute rolling OLS slope."""
        y = series.values
        n = len(y)
        slopes = np.full(n, np.nan)

        for i in range(w - 1, n):
            y_window = y[i - w + 1: i + 1]
            if np.any(np.isnan(y_window)):
                continue
            y_mean = y_window.mean()
            cov_xy = ((x - x_mean) * (y_window - y_mean)).sum()
            slopes[i] = cov_xy / x_var

        return pd.Series(slopes, index=series.index)

    df["mid_slope_10"] = rolling_slope(df["mid"], 10)

    # Close vs Mid (where did close trade relative to mid?)
    # Positive: Close above mid (aggressive buying)
    # Negative: Close below mid (aggressive selling)
    df["close_mid_diff"] = df["close"] - df["mid"]
    df["close_mid_spread_ratio"] = np.where(
        df["spread"] > 0,
        (df["close"] - df["mid"]) / df["spread"],
        0
    )

    return df


# =============================================================================
# VOLUME FEATURES
# =============================================================================

def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volume and liquidity features.

    Features (4):
    - vol_change: Volume change ratio (current / previous)
    - vol_rel_20: Volume relative to 20-bar median
    - vol_zscore_20: Volume z-score (normalized)
    - dollar_vol: Dollar volume (close * volume)

    Args:
        df: DataFrame with 'volume', 'close' columns

    Returns:
        DataFrame with volume features added

    Notes:
        If volume column not found, returns df unchanged with warning.
    """
    df = df.copy()

    if "volume" not in df.columns:
        print("    ⚠️ volume column not found, skipping volume features")
        return df

    # Volume change ratio
    df["vol_change"] = df["volume"] / df["volume"].shift(1) - 1

    # Volume relative to median (robust to outliers)
    vol_median_20 = df["volume"].rolling(20, min_periods=10).median()
    df["vol_rel_20"] = df["volume"] / vol_median_20

    # Volume z-score (standardized)
    vol_mean_20 = df["volume"].rolling(20, min_periods=10).mean()
    vol_std_20 = df["volume"].rolling(20, min_periods=10).std()
    df["vol_zscore_20"] = np.where(
        vol_std_20 > 0,
        (df["volume"] - vol_mean_20) / vol_std_20,
        0
    )

    # Dollar volume (liquidity proxy)
    df["dollar_vol"] = df["close"] * df["volume"]

    return df


# =============================================================================
# CANDLESTICK STRUCTURE FEATURES
# =============================================================================

def compute_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute candlestick structure features.

    Features (8):
    - body: Absolute body size |close - open|
    - range: High - Low
    - upper_wick: Distance from top of body to high
    - lower_wick: Distance from bottom of body to low
    - body_pct: Body as % of range
    - upper_wick_pct: Upper wick as % of range
    - lower_wick_pct: Lower wick as % of range
    - is_bull: 1 if bullish (close > open), 0 if bearish

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns

    Returns:
        DataFrame with candlestick features added
    """
    df = df.copy()

    # Body and range
    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = df["high"] - df["low"]

    # Wicks (distance from body to high/low)
    df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
    df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

    # Percentages (avoid division by zero)
    df["body_pct"] = np.where(df["range"] > 0, df["body"] / df["range"], 0)
    df["upper_wick_pct"] = np.where(df["range"] > 0, df["upper_wick"] / df["range"], 0)
    df["lower_wick_pct"] = np.where(df["range"] > 0, df["lower_wick"] / df["range"], 0)

    # Bullish/Bearish
    df["is_bull"] = (df["close"] > df["open"]).astype(int)

    return df


# =============================================================================
# MULTI-TIMEFRAME (MTF) CONTEXT FEATURES
# =============================================================================

def compute_mtf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute multi-timeframe context features.

    Features (4):
    - ma_fast_15: 15-bar moving average (fast)
    - ma_slow_60: 60-bar moving average (slow)
    - ma_ratio: Ratio of fast to slow MA (trend direction)
    - ma_slope_60: Slope of slow MA over last 5 bars (trend strength)

    Args:
        df: DataFrame with 'close' column

    Returns:
        DataFrame with MTF features added
    """
    df = df.copy()

    # Moving averages
    df["ma_fast_15"] = df["close"].rolling(15, min_periods=8).mean()
    df["ma_slow_60"] = df["close"].rolling(60, min_periods=30).mean()

    # MA ratio (fast / slow - 1)
    # Positive: Fast above slow (bullish)
    # Negative: Fast below slow (bearish)
    df["ma_ratio"] = np.where(
        df["ma_slow_60"] > 0,
        df["ma_fast_15"] / df["ma_slow_60"] - 1,
        0
    )

    # MA slope (average change over last 5 bars)
    # Positive: Uptrend, Negative: Downtrend
    df["ma_slope_60"] = df["ma_slow_60"].diff(5) / 5

    return df


# =============================================================================
# TIME FEATURES
# =============================================================================

def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute time-based features.

    Features (6):
    - minute_sin, minute_cos: Cyclical minute of day encoding
    - day_of_week: 0-6 (Monday-Sunday)
    - is_asia: 1 if Asian session (00:00-08:00 UTC)
    - is_europe: 1 if European session (07:00-16:00 UTC)
    - is_us: 1 if US session (13:00-22:00 UTC)

    Args:
        df: DataFrame with DatetimeIndex (UTC timezone)

    Returns:
        DataFrame with time features added

    Notes:
        Assumes timestamps are in UTC.
    """
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex for time features")

    # Minute of day (0-1439)
    minute_of_day = df.index.hour * 60 + df.index.minute

    # Cyclical encoding (handles 23:59 → 00:00 continuity)
    df["minute_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    df["minute_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)

    # Day of week (0=Monday, 6=Sunday)
    df["day_of_week"] = df.index.dayofweek

    # Trading sessions (UTC)
    hour = df.index.hour

    # Asia: 00:00 - 08:00 UTC (Tokyo 09:00 - 17:00 JST)
    df["is_asia"] = ((hour >= 0) & (hour < 8)).astype(int)

    # Europe: 07:00 - 16:00 UTC (London 07:00 - 16:00 GMT)
    df["is_europe"] = ((hour >= 7) & (hour < 16)).astype(int)

    # US: 13:00 - 22:00 UTC (NY 08:00 - 17:00 EST)
    df["is_us"] = ((hour >= 13) & (hour < 22)).astype(int)

    return df


# =============================================================================
# MASTER PRICE ACTION BUILDER
# =============================================================================

def compute_price_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all price action features.

    Combines:
    - Returns (8 features)
    - Volatility (5 features)
    - Microstructure (8 features, if bid/ask available)
    - Volume (4 features, if volume available)
    - Candlestick (8 features)
    - Multi-Timeframe (4 features)
    - Time (6 features)

    Total: ~35-43 features depending on data availability

    Args:
        df: DataFrame with OHLCV data (and optionally bid_price, ask_price)

    Returns:
        DataFrame with all price action features added

    Notes:
        - All features use NO lookahead bias
        - Optional features (microstructure, volume) skipped if data unavailable
    """
    print("  Computing price action features...")

    # Returns
    df = compute_return_features(df)

    # Volatility
    df = compute_volatility_features(df)

    # Microstructure (if bid/ask available)
    df = compute_quote_microstructure_features(df)

    # Volume
    df = compute_volume_features(df)

    # Candlestick structure
    df = compute_candlestick_features(df)

    # Multi-timeframe context
    df = compute_mtf_features(df)

    # Time features
    df = compute_time_features(df)

    print(f"    ✓ Price action features complete")
    return df


if __name__ == "__main__":
    print("=" * 80)
    print("PRICE ACTION FEATURES MODULE")
    print("=" * 80)
    print()
    print("Feature categories:")
    print("  1. Returns (8):         ret_1/3/5/10, log_ret_1, ret_mean_5/20/60")
    print("  2. Volatility (5):      vol_10/60, hl_range, norm_range, ATR_14")
    print("  3. Microstructure (8):  mid, spread, spread_pct, mid_ret_1, ...")
    print("  4. Volume (4):          vol_change, vol_rel_20, vol_zscore_20, dollar_vol")
    print("  5. Candlestick (8):     body, range, wicks, body_pct, ...")
    print("  6. Multi-Timeframe (4): ma_fast_15, ma_slow_60, ma_ratio, ma_slope_60")
    print("  7. Time (6):            minute_sin/cos, day_of_week, session flags")
    print()
    print("Total: ~35-43 features (depending on data availability)")
    print("=" * 80)
