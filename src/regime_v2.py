#!/usr/bin/env python3
"""
Market Regime Detection for Trading Models

Identifies market regimes to filter trades:
- TRENDING_UP: Strong uptrend (good for longs)
- TRENDING_DOWN: Strong downtrend (good for shorts)
- RANGING: Sideways/choppy (good for mean reversion)
- VOLATILE: High volatility (reduce position size or skip)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX) and DI+/DI-.
    
    Args:
        df: DataFrame with high, low, close
        period: ADX period
    
    Returns:
        DataFrame with adx, di_plus, di_minus columns
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()
    
    return pd.DataFrame({
        'adx': adx,
        'di_plus': plus_di,
        'di_minus': minus_di
    }, index=df.index)


def detect_regime(
    df: pd.DataFrame,
    adx_trending_threshold: float = 25.0,
    adx_ranging_threshold: float = 20.0,
    volatility_percentile: float = 80.0,
    trend_ma_period: int = 50
) -> pd.Series:
    """
    Detect market regime for each bar.
    
    Regimes:
    - TRENDING_UP: ADX > 25, close > SMA50, DI+ > DI-
    - TRENDING_DOWN: ADX > 25, close < SMA50, DI- > DI+
    - RANGING: ADX < 20
    - VOLATILE: ATR in top 20% (high volatility)
    
    Args:
        df: DataFrame with OHLC data
        adx_trending_threshold: ADX level for trending (default 25)
        adx_ranging_threshold: ADX level for ranging (default 20)
        volatility_percentile: Percentile for volatile regime (default 80)
        trend_ma_period: MA period for trend direction (default 50)
    
    Returns:
        Series with regime labels
    """
    # Calculate indicators
    adx_data = calculate_adx(df)
    adx = adx_data['adx']
    di_plus = adx_data['di_plus']
    di_minus = adx_data['di_minus']
    
    # Trend direction
    sma = df['close'].rolling(trend_ma_period).mean()
    above_ma = df['close'] > sma
    below_ma = df['close'] < sma
    
    # Volatility (ATR percentile)
    if 'atr_14' in df.columns:
        atr = df['atr_14']
    else:
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
    
    # NOTE: With raw=False, x is already a Series
    atr_pct_rank = atr.rolling(100).apply(
        lambda x: int(x.iloc[-1] >= np.percentile(x.dropna(), volatility_percentile)) if len(x.dropna()) > 0 else 0,
        raw=False
    )
    is_volatile = atr_pct_rank == 1
    
    # Determine regime
    regime = pd.Series('UNKNOWN', index=df.index)
    
    # Ranging: Low ADX
    regime[adx < adx_ranging_threshold] = 'RANGING'
    
    # Trending up: High ADX, above MA, DI+ > DI-
    trending_up = (adx >= adx_trending_threshold) & above_ma & (di_plus > di_minus)
    regime[trending_up] = 'TRENDING_UP'
    
    # Trending down: High ADX, below MA, DI- > DI+
    trending_down = (adx >= adx_trending_threshold) & below_ma & (di_minus > di_plus)
    regime[trending_down] = 'TRENDING_DOWN'
    
    # Volatile: Override if high volatility
    regime[is_volatile] = 'VOLATILE'
    
    return regime


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all regime-related features to dataframe.
    
    Adds:
    - adx: Average Directional Index
    - di_plus: Positive Directional Indicator
    - di_minus: Negative Directional Indicator
    - regime: Market regime label
    - regime_trending_up: Boolean for trending up
    - regime_trending_down: Boolean for trending down
    - regime_ranging: Boolean for ranging
    - regime_volatile: Boolean for volatile
    """
    df = df.copy()
    
    print("  Calculating ADX...")
    adx_data = calculate_adx(df)
    df['adx'] = adx_data['adx']
    df['di_plus'] = adx_data['di_plus']
    df['di_minus'] = adx_data['di_minus']
    
    print("  Detecting market regime...")
    df['regime'] = detect_regime(df)
    
    # Boolean features for model input
    df['regime_trending_up'] = (df['regime'] == 'TRENDING_UP').astype(int)
    df['regime_trending_down'] = (df['regime'] == 'TRENDING_DOWN').astype(int)
    df['regime_ranging'] = (df['regime'] == 'RANGING').astype(int)
    df['regime_volatile'] = (df['regime'] == 'VOLATILE').astype(int)
    
    # Print distribution
    regime_dist = df['regime'].value_counts(normalize=True)
    print("  Regime distribution:")
    for regime, pct in regime_dist.items():
        print(f"    {regime}: {pct:.1%}")
    
    return df


def filter_signals_by_regime(
    signals: pd.Series,
    regime: pd.Series,
    long_regimes: list = ['TRENDING_UP'],
    short_regimes: list = ['TRENDING_DOWN'],
    allow_ranging: bool = False
) -> pd.Series:
    """
    Filter trading signals based on market regime.
    
    Args:
        signals: Series with +1 (long), -1 (short), 0 (no trade)
        regime: Series with regime labels
        long_regimes: Regimes where longs are allowed
        short_regimes: Regimes where shorts are allowed
        allow_ranging: If True, allow both directions in RANGING
    
    Returns:
        Filtered signals series
    """
    filtered = signals.copy()
    
    # Filter longs
    long_mask = (signals == 1) & (~regime.isin(long_regimes))
    if not allow_ranging:
        long_mask = long_mask | ((signals == 1) & (regime == 'RANGING'))
    filtered[long_mask] = 0
    
    # Filter shorts
    short_mask = (signals == -1) & (~regime.isin(short_regimes))
    if not allow_ranging:
        short_mask = short_mask | ((signals == -1) & (regime == 'RANGING'))
    filtered[short_mask] = 0
    
    # Always filter volatile regime (reduce risk)
    filtered[regime == 'VOLATILE'] = 0
    
    # Stats
    original_trades = (signals != 0).sum()
    filtered_trades = (filtered != 0).sum()
    print(f"  Regime filter: {original_trades:,} -> {filtered_trades:,} trades ({filtered_trades/original_trades:.1%} kept)")
    
    return filtered


if __name__ == "__main__":
    # Test with sample data
    print("Testing regime detection module...")
    
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2024-01-01', periods=n, freq='1min')
    
    # Create trending then ranging data
    close = np.concatenate([
        2000 + np.cumsum(np.random.randn(300) * 0.5 + 0.1),  # Uptrend
        2050 + np.cumsum(np.random.randn(400) * 0.3),        # Ranging
        2040 + np.cumsum(np.random.randn(300) * 0.5 - 0.1),  # Downtrend
    ])
    
    df = pd.DataFrame({
        'open': close + np.random.randn(n) * 0.1,
        'high': close + np.abs(np.random.randn(n)) * 0.5,
        'low': close - np.abs(np.random.randn(n)) * 0.5,
        'close': close,
        'volume': np.random.randint(100, 1000, n)
    }, index=dates)
    
    df = add_regime_features(df)
    print("\nRegime detection complete!")
    print(f"Shape: {df.shape}")

