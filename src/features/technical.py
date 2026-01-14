#!/usr/bin/env python3
"""
Technical Indicators - CMF, MACD, RSI, Bollinger Bands

Generates technical indicator features for momentum and mean reversion trading.

Features generated (~25 features):
- CMF (4): cmf, cmf_momentum, cmf_zscore, cmf_trend
- MACD (10): macd, signal, histogram, momentum, crossovers, zscore
- RSI & Bollinger (6): rsi, bb_middle/upper/lower, bb_width, bb_position
- Volume & Price (5): volume_ratio, price_momentum_5/20, price_vs_sma20/50

All features use NO lookahead bias (only past/current data).
"""

import pandas as pd
import numpy as np
from typing import Optional


# =============================================================================
# CHAIKIN MONEY FLOW (CMF)
# =============================================================================

def compute_cmf_features(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Compute Chaikin Money Flow (CMF) features.

    CMF measures buying/selling pressure by combining price and volume:
    - CMF > 0: Buying pressure (accumulation)
    - CMF < 0: Selling pressure (distribution)
    - CMF > 0.25: Strong buying
    - CMF < -0.25: Strong selling

    Formula:
        Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        Money Flow Volume = Multiplier * Volume
        CMF = Sum(MF Volume) / Sum(Volume) over period

    Features (4):
    - cmf: Base Chaikin Money Flow
    - cmf_momentum: Rate of change of CMF
    - cmf_zscore: Normalized CMF (z-score)
    - cmf_trend: OLS slope of CMF

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume'
        period: Lookback period (default 20)

    Returns:
        DataFrame with CMF features added
    """
    df = df.copy()

    # Money Flow Multiplier
    # Range from -1 (close at low) to +1 (close at high)
    high_low = df['high'] - df['low']
    close_low = df['close'] - df['low']
    high_close = df['high'] - df['close']

    # Avoid division by zero
    mf_multiplier = np.where(
        high_low > 0,
        (close_low - high_close) / high_low,
        0.0
    )

    # Money Flow Volume
    mf_volume = mf_multiplier * df['volume']

    # CMF = Sum(MF Volume) / Sum(Volume) over period
    df['cmf'] = (
        mf_volume.rolling(period, min_periods=1).sum() /
        df['volume'].rolling(period, min_periods=1).sum()
    )

    # CMF momentum (rate of change)
    df['cmf_momentum'] = df['cmf'].diff(period)

    # CMF z-score (normalized relative to recent values)
    cmf_mean = df['cmf'].rolling(period * 2, min_periods=period).mean()
    cmf_std = df['cmf'].rolling(period * 2, min_periods=period).std()
    df['cmf_zscore'] = (df['cmf'] - cmf_mean) / (cmf_std + 1e-8)

    # CMF trend (OLS slope over period)
    df['cmf_trend'] = df['cmf'].rolling(period, min_periods=period).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= period else 0,
        raw=False
    )

    # Fill NaNs with neutral values
    df['cmf'] = df['cmf'].fillna(0)
    df['cmf_momentum'] = df['cmf_momentum'].fillna(0)
    df['cmf_zscore'] = df['cmf_zscore'].fillna(0)
    df['cmf_trend'] = df['cmf_trend'].fillna(0)

    return df


# =============================================================================
# MACD (MOVING AVERAGE CONVERGENCE DIVERGENCE)
# =============================================================================

def compute_macd_features(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """
    Compute MACD (Moving Average Convergence Divergence) features.

    MACD is a trend-following momentum indicator:
    - MACD > 0: Bullish (fast MA above slow MA)
    - MACD < 0: Bearish (fast MA below slow MA)
    - Histogram > 0: MACD above signal (momentum increasing)
    - Crossovers: Signal line crosses (entry/exit signals)

    Formula:
        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD, signal_period)
        Histogram = MACD - Signal

    Features (10):
    - macd: MACD line
    - macd_signal: Signal line
    - macd_histogram: Histogram (MACD - Signal)
    - macd_momentum: Rate of change of MACD
    - macd_signal_momentum: Rate of change of signal
    - macd_above_signal: Binary flag (1 if MACD > signal)
    - macd_cross_up: Bullish crossover (1 if just crossed up)
    - macd_cross_down: Bearish crossover (1 if just crossed down)
    - macd_hist_momentum: Rate of change of histogram
    - macd_zscore: Normalized MACD (z-score)

    Args:
        df: DataFrame with 'close' column
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)

    Returns:
        DataFrame with MACD features added
    """
    df = df.copy()

    # Calculate EMAs
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()

    # MACD line
    df['macd'] = ema_fast - ema_slow

    # Signal line (EMA of MACD)
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()

    # Histogram (MACD - Signal)
    df['macd_histogram'] = df['macd'] - df['macd_signal']

    # MACD momentum (rate of change)
    df['macd_momentum'] = df['macd'].diff()
    df['macd_signal_momentum'] = df['macd_signal'].diff()

    # MACD crossovers (entry/exit signals)
    df['macd_above_signal'] = (df['macd'] > df['macd_signal']).astype(int)

    # Bullish crossover (MACD crosses above signal)
    df['macd_cross_up'] = (
        (df['macd'] > df['macd_signal']) &
        (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    ).astype(int)

    # Bearish crossover (MACD crosses below signal)
    df['macd_cross_down'] = (
        (df['macd'] < df['macd_signal']) &
        (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    ).astype(int)

    # MACD histogram momentum
    df['macd_hist_momentum'] = df['macd_histogram'].diff()

    # Normalized MACD (z-score over double slow period)
    macd_mean = df['macd'].rolling(slow * 2, min_periods=slow).mean()
    macd_std = df['macd'].rolling(slow * 2, min_periods=slow).std()
    df['macd_zscore'] = (df['macd'] - macd_mean) / (macd_std + 1e-8)

    # Fill NaNs with neutral values
    df['macd'] = df['macd'].fillna(0)
    df['macd_signal'] = df['macd_signal'].fillna(0)
    df['macd_histogram'] = df['macd_histogram'].fillna(0)
    df['macd_momentum'] = df['macd_momentum'].fillna(0)
    df['macd_signal_momentum'] = df['macd_signal_momentum'].fillna(0)
    df['macd_hist_momentum'] = df['macd_hist_momentum'].fillna(0)
    df['macd_zscore'] = df['macd_zscore'].fillna(0)

    return df


# =============================================================================
# RSI & BOLLINGER BANDS
# =============================================================================

def compute_rsi_bb_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute RSI and Bollinger Bands features.

    RSI (Relative Strength Index):
    - RSI > 70: Overbought
    - RSI < 30: Oversold
    - RSI ~50: Neutral

    Bollinger Bands:
    - Price at upper band: Overbought
    - Price at lower band: Oversold
    - BB width: Volatility measure (narrow = low vol, wide = high vol)
    - BB position: Where price is within bands (0 = lower, 1 = upper)

    Features (6):
    - rsi: Relative Strength Index (14-period)
    - bb_middle: Middle band (20-period SMA)
    - bb_upper: Upper band (middle + 2*std)
    - bb_lower: Lower band (middle - 2*std)
    - bb_width: Band width as % of middle
    - bb_position: Price position in bands (0-1)

    Args:
        df: DataFrame with 'high', 'low', 'close'

    Returns:
        DataFrame with RSI and BB features added
    """
    df = df.copy()

    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR (Average True Range) - used by some models
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14, min_periods=1).mean()

    # Bollinger Bands (20-period, 2 std)
    df['bb_middle'] = df['close'].rolling(20, min_periods=1).mean()
    bb_std = df['close'].rolling(20, min_periods=1).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

    # BB width (volatility measure)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    # BB position (where price is within bands)
    # 0 = at lower band, 1 = at upper band, 0.5 = at middle
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

    # Fill NaNs
    df['rsi'] = df['rsi'].fillna(50)  # Neutral RSI
    df['atr'] = df['atr'].fillna(0)
    df['bb_middle'] = df['bb_middle'].fillna(df['close'])
    df['bb_upper'] = df['bb_upper'].fillna(df['close'])
    df['bb_lower'] = df['bb_lower'].fillna(df['close'])
    df['bb_width'] = df['bb_width'].fillna(0)
    df['bb_position'] = df['bb_position'].fillna(0.5)

    return df


# =============================================================================
# VOLUME & PRICE MOMENTUM
# =============================================================================

def compute_volume_price_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volume and price momentum features.

    Features (5):
    - volume_ratio: Volume / 20-period SMA (relative volume)
    - price_momentum_5: 5-bar price change %
    - price_momentum_20: 20-bar price change %
    - price_vs_sma20: Price distance from 20-SMA (%)
    - price_vs_sma50: Price distance from 50-SMA (%)

    Args:
        df: DataFrame with 'close', 'volume'

    Returns:
        DataFrame with volume/price momentum features added
    """
    df = df.copy()

    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(20, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)

    # Price momentum
    df['price_momentum_5'] = df['close'].pct_change(5, fill_method=None)
    df['price_momentum_20'] = df['close'].pct_change(20, fill_method=None)

    # Moving averages and price distance
    df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
    df['sma_50'] = df['close'].rolling(50, min_periods=1).mean()
    df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']

    # Fill NaNs
    df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
    df['price_momentum_5'] = df['price_momentum_5'].fillna(0)
    df['price_momentum_20'] = df['price_momentum_20'].fillna(0)
    df['price_vs_sma20'] = df['price_vs_sma20'].fillna(0)
    df['price_vs_sma50'] = df['price_vs_sma50'].fillna(0)

    return df


# =============================================================================
# MASTER TECHNICAL INDICATOR BUILDER
# =============================================================================

def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicator features.

    Combines:
    - CMF (4 features): Money flow pressure
    - MACD (10 features): Trend and momentum
    - RSI & BB (6 features): Overbought/oversold, volatility
    - Volume & Momentum (5 features): Volume and price dynamics

    Total: ~25 features

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with all technical indicator features added

    Notes:
        - All features use NO lookahead bias
        - Default periods: CMF=20, MACD=12/26/9, RSI=14, BB=20
    """
    print("  Computing technical indicator features...")

    # CMF features
    df = compute_cmf_features(df, period=20)

    # MACD features
    df = compute_macd_features(df, fast=12, slow=26, signal=9)

    # RSI & Bollinger Bands
    df = compute_rsi_bb_features(df)

    # Volume & Price Momentum
    df = compute_volume_price_momentum(df)

    print(f"    ✓ Technical indicator features complete")
    return df


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def compute_chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    DEPRECATED: Use compute_cmf_features() instead.

    Legacy function name for backward compatibility.
    """
    import warnings
    warnings.warn(
        "compute_chaikin_money_flow() is deprecated. Use compute_cmf_features() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return compute_cmf_features(df, period)


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    DEPRECATED: Use compute_macd_features() instead.

    Legacy function name for backward compatibility.
    """
    import warnings
    warnings.warn(
        "compute_macd() is deprecated. Use compute_macd_features() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return compute_macd_features(df, fast, slow, signal)


def compute_additional_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    DEPRECATED: Use compute_rsi_bb_features() + compute_volume_price_momentum() instead.

    Legacy function name for backward compatibility.
    """
    import warnings
    warnings.warn(
        "compute_additional_indicators() is deprecated. Use compute_rsi_bb_features() + compute_volume_price_momentum() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    df = compute_rsi_bb_features(df)
    df = compute_volume_price_momentum(df)
    return df


def build_cmf_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    DEPRECATED: Use compute_technical_features() instead.

    Legacy function name for backward compatibility.
    """
    import warnings
    warnings.warn(
        "build_cmf_macd_features() is deprecated. Use compute_technical_features() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return compute_technical_features(df)


if __name__ == "__main__":
    print("=" * 80)
    print("TECHNICAL INDICATORS MODULE")
    print("=" * 80)
    print()
    print("Feature categories:")
    print("  1. CMF (4):             cmf, cmf_momentum, cmf_zscore, cmf_trend")
    print("  2. MACD (10):           macd, signal, histogram, momentum, crossovers, zscore")
    print("  3. RSI & BB (6):        rsi, atr, bb_middle/upper/lower, bb_width, bb_position")
    print("  4. Volume/Momentum (5): volume_ratio, price_momentum_5/20, price_vs_sma20/50")
    print()
    print("Total: ~25 features")
    print()
    print("Key indicators:")
    print("  - CMF: Money flow pressure (buying/selling)")
    print("  - MACD: Trend and momentum detection")
    print("  - RSI: Overbought/oversold conditions")
    print("  - BB: Volatility and price extremes")
    print("=" * 80)
