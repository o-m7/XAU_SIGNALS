"""
Regime Classification for Mean Reversion

Mean reversion works best in ranging markets, not trending ones.
This module classifies market regime to filter trades.
"""
import numpy as np
import pandas as pd


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average Directional Index (ADX).

    ADX measures trend strength:
    - ADX < 20: Weak/no trend (good for mean reversion)
    - ADX 20-25: Developing trend (cautious)
    - ADX > 25: Strong trend (avoid mean reversion)
    """
    df = df.copy()

    # True Range
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    # Directional Movement
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed averages
    atr = tr.rolling(period, min_periods=1).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period, min_periods=1).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period, min_periods=1).mean() / atr

    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    df['adx'] = dx.rolling(period, min_periods=1).mean()
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    return df


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Calculate Average True Range."""
    df = df.copy()

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift(1))
    low_close = np.abs(df['low'] - df['close'].shift(1))

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f'atr_{period}'] = tr.rolling(period, min_periods=1).mean()

    return df


def classify_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify market regime for mean reversion trading.

    Regimes:
    - RANGING: ADX < 20, good for mean reversion
    - WEAK_TREND: ADX 20-30, cautious mean reversion
    - STRONG_TREND: ADX > 30, avoid mean reversion
    - LOW_VOL: ATR percentile < 20, avoid (no movement)
    - HIGH_VOL: ATR percentile > 80, avoid (erratic)

    Returns DataFrame with regime columns.
    """
    df = df.copy()

    # Calculate ADX if not present
    if 'adx' not in df.columns:
        df = calculate_adx(df, period=14)

    # Calculate ATR if not present
    if 'atr_14' not in df.columns:
        df = calculate_atr(df, period=14)

    # ATR percentile (rolling 100-bar window)
    df['atr_percentile'] = df['atr_14'].rolling(100, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
        raw=False
    )

    # Range compression (current range vs average)
    df['range'] = df['high'] - df['low']
    df['range_ma'] = df['range'].rolling(20, min_periods=1).mean()
    df['range_compression'] = df['range'] / df['range_ma'].replace(0, np.nan)

    # Regime classification
    conditions = [
        (df['adx'] < 20) & (df['atr_percentile'].between(20, 80)),  # Ideal for MR
        (df['adx'].between(20, 30)) & (df['atr_percentile'].between(20, 80)),  # Cautious MR
        (df['adx'] >= 30),  # Trending - avoid MR
        (df['atr_percentile'] < 20),  # Dead - avoid
        (df['atr_percentile'] > 80),  # Volatile - avoid
    ]
    choices = ['RANGING', 'WEAK_TREND', 'STRONG_TREND', 'LOW_VOL', 'HIGH_VOL']
    df['regime'] = np.select(conditions, choices, default='UNKNOWN')

    # Binary flag for tradeable regime
    df['regime_tradeable'] = df['regime'].isin(['RANGING', 'WEAK_TREND']).astype(int)

    return df
