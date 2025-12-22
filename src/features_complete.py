"""
Complete Feature Engineering for XAUUSD.

This module provides comprehensive feature engineering including:
- Price & Returns
- Volatility & Range
- Quote Microstructure
- Volume & Liquidity
- Candlestick Structure
- Multi-Timeframe Context
- Time Features
- Fixed-Horizon Labels
- Triple-Barrier Labels

All computations are vectorized where possible.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


# =============================================================================
# MERGE QUOTES INTO BARS
# =============================================================================

def merge_quotes_to_bars(
    bars: pd.DataFrame,
    quotes: pd.DataFrame,
    tolerance: str = "120s"  # 2 minutes tolerance for minute bars
) -> pd.DataFrame:
    """
    Merge top-of-book quotes into bar data using forward-fill within tolerance.
    
    Args:
        bars: Bar data with timestamp index, OHLCV columns
        quotes: Quote data with timestamp index, bid_price, ask_price
        tolerance: Maximum time tolerance for forward-fill
        
    Returns:
        Merged DataFrame with quote columns added to bars
    """
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
    ts_col = bars_reset.columns[0]  # First column is the timestamp
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
# PRICE & RETURNS FEATURES
# =============================================================================

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price return features.
    
    Features:
    - ret_1, ret_3, ret_5, ret_10: Simple returns
    - log_ret_1: Log return
    - ret_mean_5, ret_mean_20, ret_mean_60: Rolling mean returns
    """
    df = df.copy()
    
    # Simple returns at various lags
    for n in [1, 3, 5, 10]:
        df[f"ret_{n}"] = df["close"].pct_change(n)
    
    # Log return
    df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))
    
    # Rolling mean returns
    for n in [5, 20, 60]:
        df[f"ret_mean_{n}"] = df["log_ret_1"].rolling(n, min_periods=n//2).mean()
    
    return df


# =============================================================================
# VOLATILITY & RANGE FEATURES
# =============================================================================

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility and range features.
    
    Features:
    - vol_10, vol_60: Rolling std of log returns
    - hl_range: High - Low
    - norm_range: (High - Low) / Close
    - ATR_14: Average True Range
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
# QUOTE MICROSTRUCTURE FEATURES
# =============================================================================

def add_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add quote microstructure features.
    
    Requires: bid_price, ask_price
    
    Features:
    - mid: Mid price
    - spread: Bid-ask spread
    - spread_pct: Spread as percentage
    - mid_ret_1: Mid price return
    - mid_vol_20: Mid price volatility
    - mid_slope_10: Rolling OLS slope of mid
    - close_mid_diff: Close - Mid
    - close_mid_spread_ratio: (Close - Mid) / Spread
    """
    df = df.copy()
    
    # Check for required columns
    if "bid_price" not in df.columns or "ask_price" not in df.columns:
        print("Warning: bid_price/ask_price not found, skipping microstructure features")
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
    
    # Close vs Mid
    df["close_mid_diff"] = df["close"] - df["mid"]
    df["close_mid_spread_ratio"] = np.where(
        df["spread"] > 0,
        (df["close"] - df["mid"]) / df["spread"],
        0
    )
    
    return df


# =============================================================================
# VOLUME & LIQUIDITY FEATURES
# =============================================================================

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume and liquidity features.
    
    Features:
    - vol_change: Volume change ratio
    - vol_rel_20: Volume relative to 20-bar median
    - vol_zscore_20: Volume z-score
    - dollar_vol: Dollar volume (close * volume)
    """
    df = df.copy()
    
    if "volume" not in df.columns:
        print("Warning: volume column not found, skipping volume features")
        return df
    
    # Volume change
    df["vol_change"] = df["volume"] / df["volume"].shift(1) - 1
    
    # Volume relative to median
    vol_median_20 = df["volume"].rolling(20, min_periods=10).median()
    df["vol_rel_20"] = df["volume"] / vol_median_20
    
    # Volume z-score
    vol_mean_20 = df["volume"].rolling(20, min_periods=10).mean()
    vol_std_20 = df["volume"].rolling(20, min_periods=10).std()
    df["vol_zscore_20"] = np.where(
        vol_std_20 > 0,
        (df["volume"] - vol_mean_20) / vol_std_20,
        0
    )
    
    # Dollar volume
    df["dollar_vol"] = df["close"] * df["volume"]
    
    return df


# =============================================================================
# CANDLESTICK STRUCTURE FEATURES
# =============================================================================

def add_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add candlestick structure features.
    
    Features:
    - body: Absolute body size
    - range: High - Low
    - upper_wick, lower_wick: Wick sizes
    - body_pct, upper_wick_pct, lower_wick_pct: Percentages of range
    - is_bull: 1 if bullish, 0 if bearish
    """
    df = df.copy()
    
    # Body and range
    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = df["high"] - df["low"]
    
    # Wicks
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

def add_mtf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add multi-timeframe context features.
    
    Features:
    - ma_fast_15: 15-bar moving average
    - ma_slow_60: 60-bar moving average
    - ma_ratio: Ratio of fast to slow MA
    - ma_slope_60: Slope of slow MA over last 5 bars
    """
    df = df.copy()
    
    # Moving averages
    df["ma_fast_15"] = df["close"].rolling(15, min_periods=8).mean()
    df["ma_slow_60"] = df["close"].rolling(60, min_periods=30).mean()
    
    # MA ratio
    df["ma_ratio"] = np.where(
        df["ma_slow_60"] > 0,
        df["ma_fast_15"] / df["ma_slow_60"] - 1,
        0
    )
    
    # MA slope (average change over last 5 bars)
    df["ma_slope_60"] = df["ma_slow_60"].diff(5) / 5
    
    return df


# =============================================================================
# TIME FEATURES
# =============================================================================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features.
    
    Features:
    - minute_sin, minute_cos: Cyclical minute of day
    - day_of_week: 0-6
    - is_asia, is_europe, is_us: Session flags
    
    Assumes timestamps are in UTC.
    """
    df = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    
    # Minute of day (0-1439)
    minute_of_day = df.index.hour * 60 + df.index.minute
    
    # Cyclical encoding
    df["minute_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    df["minute_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    
    # Day of week
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
# FIXED-HORIZON DIRECTIONAL LABELS
# =============================================================================

def add_fixed_horizon_labels(
    df: pd.DataFrame,
    horizons: list = [5, 15, 60],
    epsilon: float = 0.0005  # 0.05% threshold
) -> pd.DataFrame:
    """
    Add fixed-horizon directional labels.
    
    For each horizon H:
    - y_ret_H: Log return over H bars
    - y_dir_H: Direction (+1, 0, -1) based on epsilon threshold
    
    Args:
        df: DataFrame with close prices
        horizons: List of horizons in bars
        epsilon: Threshold for directional labels
    """
    df = df.copy()
    
    for h in horizons:
        # Future log return
        df[f"y_ret_{h}"] = np.log(df["close"].shift(-h) / df["close"])
        
        # Directional label
        df[f"y_dir_{h}"] = 0
        df.loc[df[f"y_ret_{h}"] > epsilon, f"y_dir_{h}"] = 1
        df.loc[df[f"y_ret_{h}"] < -epsilon, f"y_dir_{h}"] = -1
    
    return df


# =============================================================================
# TRIPLE-BARRIER LABELS
# =============================================================================

def add_triple_barrier_labels(
    df: pd.DataFrame,
    h_max: int = 60,
    tp_mult: float = 1.0,
    sl_mult: float = 1.0
) -> pd.DataFrame:
    """
    Add triple-barrier labels using ATR-scaled barriers.
    
    For each bar t:
    - Upper barrier: close_t * (1 + TP_mult * ATR_14_t / close_t)
    - Lower barrier: close_t * (1 - SL_mult * ATR_14_t / close_t)
    - Walk forward until barrier hit or h_max reached
    - y_tb_60 = +1 (upper hit first), -1 (lower hit first), 0 (neither)
    
    Args:
        df: DataFrame with close and ATR_14
        h_max: Maximum horizon
        tp_mult: Take-profit multiplier
        sl_mult: Stop-loss multiplier
    """
    df = df.copy()
    
    # Ensure ATR_14 exists
    if "ATR_14" not in df.columns:
        df = add_volatility_features(df)
    
    close = df["close"].values
    atr = df["ATR_14"].values
    n = len(df)
    
    labels = np.zeros(n)
    
    for t in range(n - 1):
        if np.isnan(close[t]) or np.isnan(atr[t]) or atr[t] <= 0:
            labels[t] = np.nan
            continue
        
        # Compute barriers
        upper = close[t] * (1 + tp_mult * atr[t] / close[t])
        lower = close[t] * (1 - sl_mult * atr[t] / close[t])
        
        # Walk forward
        label = 0
        for h in range(1, min(h_max + 1, n - t)):
            future_close = close[t + h]
            
            if np.isnan(future_close):
                continue
            
            # Check barriers
            if future_close >= upper:
                label = 1  # Upper barrier hit first
                break
            elif future_close <= lower:
                label = -1  # Lower barrier hit first
                break
        
        labels[t] = label
    
    # Last h_max bars don't have enough future data
    labels[n - h_max:] = np.nan
    
    df[f"y_tb_{h_max}"] = labels
    
    return df


# =============================================================================
# MASTER BUILD FUNCTION
# =============================================================================

def build_complete_features(
    bars: pd.DataFrame,
    quotes: Optional[pd.DataFrame] = None,
    horizons: list = [5, 15, 60],
    epsilon: float = 0.0005,
    tb_h_max: int = 60,
    tb_tp_mult: float = 1.0,
    tb_sl_mult: float = 1.0,
    drop_na: bool = True
) -> pd.DataFrame:
    """
    Build complete feature matrix with all features and labels.
    
    Args:
        bars: Bar data (OHLCV)
        quotes: Optional quote data (bid_price, ask_price)
        horizons: Horizons for fixed-horizon labels
        epsilon: Threshold for directional labels
        tb_h_max: Max horizon for triple-barrier
        tb_tp_mult: Take-profit multiplier for triple-barrier
        tb_sl_mult: Stop-loss multiplier for triple-barrier
        drop_na: Whether to drop rows with NaN labels
        
    Returns:
        Complete DataFrame with features and labels
    """
    print("Building complete feature matrix...")
    
    # Merge quotes if provided
    if quotes is not None:
        print("  Merging quotes into bars...")
        df = merge_quotes_to_bars(bars, quotes)
    else:
        df = bars.copy()
    
    # Add all features
    print("  Adding return features...")
    df = add_return_features(df)
    
    print("  Adding volatility features...")
    df = add_volatility_features(df)
    
    print("  Adding microstructure features...")
    df = add_microstructure_features(df)
    
    print("  Adding volume features...")
    df = add_volume_features(df)
    
    print("  Adding candlestick features...")
    df = add_candlestick_features(df)
    
    print("  Adding MTF features...")
    df = add_mtf_features(df)
    
    print("  Adding time features...")
    df = add_time_features(df)
    
    # Add labels
    print("  Adding fixed-horizon labels...")
    df = add_fixed_horizon_labels(df, horizons=horizons, epsilon=epsilon)
    
    print("  Adding triple-barrier labels...")
    df = add_triple_barrier_labels(df, h_max=tb_h_max, tp_mult=tb_tp_mult, sl_mult=tb_sl_mult)
    
    # Drop rows with NaN labels if requested
    if drop_na:
        label_cols = [c for c in df.columns if c.startswith("y_")]
        initial_rows = len(df)
        df = df.dropna(subset=label_cols)
        dropped = initial_rows - len(df)
        print(f"  Dropped {dropped:,} rows with NaN labels")
    
    print(f"  ✓ Complete: {len(df):,} rows, {len(df.columns)} columns")
    
    return df


# =============================================================================
# FEATURE LISTS
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature columns (excluding labels)."""
    label_prefixes = ["y_ret_", "y_dir_", "y_tb_"]
    raw_cols = ["open", "high", "low", "close", "volume", "bid_price", "ask_price"]
    
    features = []
    for col in df.columns:
        # Skip labels
        if any(col.startswith(p) for p in label_prefixes):
            continue
        # Skip raw OHLCV/quote columns
        if col in raw_cols:
            continue
        features.append(col)
    
    return features


def get_label_columns(df: pd.DataFrame) -> list:
    """Get list of label columns."""
    return [c for c in df.columns if c.startswith("y_")]


# =============================================================================
# MAIN - Demo usage
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add project to path
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT.parent / "Data"
    
    # Load sample data
    print("Loading sample data...")
    
    bars_path = DATA_DIR / "ohlcv_minute" / "XAUUSD_minute_2024.parquet"
    quotes_path = DATA_DIR / "quotes" / "XAUUSD_quotes_2024.parquet"
    
    if not bars_path.exists():
        print(f"Bars file not found: {bars_path}")
        sys.exit(1)
    
    bars = pd.read_parquet(bars_path)
    if "timestamp" in bars.columns:
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        bars = bars.set_index("timestamp")
    
    quotes = None
    if quotes_path.exists():
        quotes = pd.read_parquet(quotes_path)
        if "timestamp" in quotes.columns:
            quotes["timestamp"] = pd.to_datetime(quotes["timestamp"], utc=True)
            quotes = quotes.set_index("timestamp")
    
    print(f"Bars: {len(bars):,} rows")
    print(f"Quotes: {len(quotes):,} rows" if quotes is not None else "No quotes")
    
    # Build features
    df = build_complete_features(bars, quotes)
    
    # Summary
    print("\n" + "=" * 60)
    print("FEATURE SUMMARY")
    print("=" * 60)
    
    features = get_feature_columns(df)
    labels = get_label_columns(df)
    
    print(f"\nFeatures ({len(features)}):")
    for f in features:
        nan_pct = df[f].isna().mean() * 100
        print(f"  {f}: {nan_pct:.1f}% NaN")
    
    print(f"\nLabels ({len(labels)}):")
    for l in labels:
        if l.startswith("y_dir_") or l.startswith("y_tb_"):
            dist = df[l].value_counts(normalize=True)
            print(f"  {l}: +1={dist.get(1, 0)*100:.1f}%, 0={dist.get(0, 0)*100:.1f}%, -1={dist.get(-1, 0)*100:.1f}%")
        else:
            print(f"  {l}: mean={df[l].mean():.6f}, std={df[l].std():.6f}")

