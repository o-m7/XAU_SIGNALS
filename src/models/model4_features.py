#!/usr/bin/env python3
"""
Model #4 Feature Engineering (5m News-Gated Long-Only).

Fast feature set (5 features only) for low-latency inference:
1. RSI 14 (mean-reversion on 5m)
2. MACD histogram sign (momentum confirmation)
3. Bollinger Band width ratio (volatility regime)
4. ATR 14 (position sizing + stop placement)
5. Volume delta (tick imbalance over last 5 bars)

All features computed on 5-minute bars.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("Model4Features")


def compute_model4_features(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    """
    Compute Model #4 features on 5-minute XAUUSD data.

    Args:
        df: DataFrame with OHLCV columns (5min bars)
        inplace: If True, modify df in place. If False, return new DataFrame.

    Returns:
        DataFrame with 5 additional feature columns:
            - rsi_14
            - macd_hist_sign
            - bb_width_ratio
            - atr_14
            - volume_delta_5
    """
    if not inplace:
        df = df.copy()

    logger.info(f"Computing Model4 features on {len(df)} bars...")

    # Ensure we have required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # =============================================================================
    # FEATURE 1: RSI 14 (Mean-Reversion Indicator)
    # =============================================================================
    df['rsi_14'] = compute_rsi(df['close'], period=14)

    # =============================================================================
    # FEATURE 2: MACD Histogram Sign (Momentum Confirmation)
    # =============================================================================
    macd_line, signal_line, macd_hist = compute_macd(df['close'])
    df['macd_hist_sign'] = np.sign(macd_hist)  # -1, 0, or +1

    # =============================================================================
    # FEATURE 3: Bollinger Band Width Ratio (Volatility Regime)
    # =============================================================================
    df['bb_width_ratio'] = compute_bb_width_ratio(df['close'], period=20, num_std=2.0)

    # =============================================================================
    # FEATURE 4: ATR 14 (For position sizing + stop placement)
    # =============================================================================
    df['atr_14'] = compute_atr(df['high'], df['low'], df['close'], period=14)

    # =============================================================================
    # FEATURE 5: Volume Delta (Tick Imbalance over Last 5 Bars)
    # =============================================================================
    df['volume_delta_5'] = compute_volume_delta(df['volume'], window=5)

    logger.info("✓ Model4 features computed")

    return df


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI (Relative Strength Index).

    Args:
        series: Price series (typically close)
        period: RSI period (default: 14)

    Returns:
        RSI values [0, 100]
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> tuple:
    """
    Compute MACD (Moving Average Convergence Divergence).

    Args:
        series: Price series (typically close)
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)

    Returns:
        (macd_line, signal_line, histogram)
    """
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def compute_bb_width_ratio(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0
) -> pd.Series:
    """
    Compute Bollinger Band width ratio (volatility measure).

    BB Width Ratio = (upper_band - lower_band) / middle_band

    Args:
        series: Price series (typically close)
        period: SMA period (default: 20)
        num_std: Number of standard deviations (default: 2.0)

    Returns:
        BB width ratio (higher = more volatile)
    """
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()

    upper_band = sma + (num_std * std)
    lower_band = sma - (num_std * std)

    width_ratio = (upper_band - lower_band) / sma

    return width_ratio


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Compute ATR (Average True Range).

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period (default: 14)

    Returns:
        ATR values
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()

    return atr


def compute_volume_delta(volume: pd.Series, window: int = 5) -> pd.Series:
    """
    Compute volume delta (tick imbalance over rolling window).

    Simple version: (current_volume - rolling_mean) / rolling_std

    Args:
        volume: Volume series
        window: Rolling window (default: 5)

    Returns:
        Volume delta (z-score normalized)
    """
    vol_mean = volume.rolling(window=window).mean()
    vol_std = volume.rolling(window=window).std()

    volume_delta = (volume - vol_mean) / (vol_std + 1e-6)

    return volume_delta


# =============================================================================
# Feature List (for model training/inference)
# =============================================================================

MODEL4_FEATURES = [
    'rsi_14',
    'macd_hist_sign',
    'bb_width_ratio',
    'atr_14',
    'volume_delta_5',
]


# =============================================================================
# Standalone Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load 5min data
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "models" / "model2_regime" / "regime_features_5min.parquet"

    if not data_path.exists():
        print(f"Data not found: {data_path}")
        sys.exit(1)

    print(f"Loading data from: {data_path}")
    df = pd.read_parquet(data_path)

    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Compute features
    df_with_features = compute_model4_features(df, inplace=False)

    print(f"\nFeatures computed:")
    print(df_with_features[MODEL4_FEATURES].describe())

    # Check for NaN
    nan_counts = df_with_features[MODEL4_FEATURES].isna().sum()
    print(f"\nNaN counts:")
    print(nan_counts)

    print(f"\n✓ Feature engineering test successful")
