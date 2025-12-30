"""
Feature Engineering for VWAP Mean Reversion Model

Features designed for mean reversion strategy:
- VWAP distance and velocity
- Regime context (ADX, ATR percentile)
- Momentum exhaustion signals
- Session/spread context
"""
import numpy as np
import pandas as pd
from typing import Optional

from .vwap import calculate_session_vwap, calculate_vwap_zscore
from .regime import classify_regime, calculate_atr


def calculate_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(length, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(length, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))


def build_model4_features(
    df_1t: pd.DataFrame,
    df_quotes: Optional[pd.DataFrame] = None,
    timeframe: str = "5T",
    session_hours: int = 8
) -> pd.DataFrame:
    """
    Build features for VWAP mean reversion model.

    Parameters:
    -----------
    df_1t : pd.DataFrame
        1-minute OHLCV data with DatetimeIndex
    df_quotes : pd.DataFrame, optional
        Quote data with bid_price, ask_price
    timeframe : str
        Target timeframe for resampling (default "5T")
    session_hours : int
        Rolling window for VWAP calculation (default 8)

    Returns:
    --------
    pd.DataFrame with all features
    """

    # Resample to target timeframe
    df = df_1t.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # ===== ATR FIRST (needed for z-score) =====
    df = calculate_atr(df, period=14)
    df = calculate_atr(df, period=5)

    # ===== VWAP FEATURES =====
    df = calculate_session_vwap(df, session_hours=session_hours)
    df = calculate_vwap_zscore(df)

    # ===== REGIME FEATURES =====
    df = classify_regime(df)

    # ===== SESSION POSITION FEATURES =====
    # Rolling session high/low
    freq_minutes = 5 if timeframe == "5T" else int(timeframe.replace("T", ""))
    session_bars = int(session_hours * 60 / freq_minutes)

    df['session_high'] = df['high'].rolling(session_bars, min_periods=1).max()
    df['session_low'] = df['low'].rolling(session_bars, min_periods=1).min()
    df['session_range'] = df['session_high'] - df['session_low']

    df['price_vs_session_high'] = (df['session_high'] - df['close']) / df['atr_14'].replace(0, np.nan)
    df['price_vs_session_low'] = (df['close'] - df['session_low']) / df['atr_14'].replace(0, np.nan)
    df['price_in_session_range'] = (df['close'] - df['session_low']) / df['session_range'].replace(0, np.nan)

    # ===== MOMENTUM EXHAUSTION FEATURES =====
    df['rsi_14'] = calculate_rsi(df['close'], length=14)
    df['rsi_7'] = calculate_rsi(df['close'], length=7)

    # RSI divergence: price making new high/low but RSI not confirming
    df['price_high_5'] = df['high'].rolling(5, min_periods=1).max()
    df['price_low_5'] = df['low'].rolling(5, min_periods=1).min()
    df['rsi_high_5'] = df['rsi_14'].rolling(5, min_periods=1).max()
    df['rsi_low_5'] = df['rsi_14'].rolling(5, min_periods=1).min()

    # Bearish divergence: price at high, RSI below recent high
    df['bearish_divergence'] = (
        (df['close'] >= df['price_high_5'] * 0.999) &
        (df['rsi_14'] < df['rsi_high_5'] - 5)
    ).astype(int)

    # Bullish divergence: price at low, RSI above recent low
    df['bullish_divergence'] = (
        (df['close'] <= df['price_low_5'] * 1.001) &
        (df['rsi_14'] > df['rsi_low_5'] + 5)
    ).astype(int)

    df['rsi_divergence'] = df['bearish_divergence'] - df['bullish_divergence']

    # Bars since price was at extreme z-score
    df['at_upper_extreme'] = (df['vwap_zscore'] > 1.5).astype(int)
    df['at_lower_extreme'] = (df['vwap_zscore'] < -1.5).astype(int)

    # Count consecutive bars at extreme
    df['bars_at_upper'] = df['at_upper_extreme'].groupby(
        (~df['at_upper_extreme'].astype(bool)).cumsum()
    ).cumsum()
    df['bars_at_lower'] = df['at_lower_extreme'].groupby(
        (~df['at_lower_extreme'].astype(bool)).cumsum()
    ).cumsum()
    df['bars_since_extreme'] = np.maximum(df['bars_at_upper'], df['bars_at_lower'])

    # ===== SPREAD FEATURES =====
    if df_quotes is not None and len(df_quotes) > 0:
        quotes_resampled = df_quotes.resample(timeframe).agg({
            'ask_price': 'mean',
            'bid_price': 'mean',
        })
        quotes_resampled['spread'] = quotes_resampled['ask_price'] - quotes_resampled['bid_price']
        quotes_resampled['mid'] = (quotes_resampled['ask_price'] + quotes_resampled['bid_price']) / 2
        quotes_resampled['spread_pct'] = quotes_resampled['spread'] / quotes_resampled['mid'].replace(0, np.nan)
        quotes_resampled['spread_zscore'] = (
            (quotes_resampled['spread_pct'] - quotes_resampled['spread_pct'].rolling(60).mean()) /
            quotes_resampled['spread_pct'].rolling(60).std().replace(0, np.nan)
        )

        quote_counts = df_quotes.resample(timeframe).size()
        quotes_resampled['quote_rate'] = quote_counts
        quotes_resampled['quote_rate_zscore'] = (
            (quotes_resampled['quote_rate'] - quotes_resampled['quote_rate'].rolling(60).mean()) /
            quotes_resampled['quote_rate'].rolling(60).std().replace(0, np.nan)
        )

        df = df.join(quotes_resampled[['spread_pct', 'spread_zscore', 'quote_rate_zscore']], how='left')
    else:
        df['spread_pct'] = 0.0001
        df['spread_zscore'] = 0.0
        df['quote_rate_zscore'] = 0.0

    # ===== TIME FEATURES =====
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Minutes since session open (London = 7:00 UTC)
    df['minutes_since_london'] = (df['hour'] - 7) * 60 + df.index.minute
    df['minutes_since_london'] = df['minutes_since_london'].clip(lower=0)

    df['is_london'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
    df['is_ny'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
    df['is_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)

    # ===== CLEANUP =====
    df = df.replace([np.inf, -np.inf], np.nan).ffill().dropna()

    return df


def get_model4_feature_columns() -> list:
    """Return feature columns for Model 4 VWAP Mean Reversion."""
    return [
        'vwap_zscore',
        'vwap_zscore_velocity',
        'price_vs_session_high',
        'price_vs_session_low',
        'adx',
        'atr_percentile',
        'range_compression',
        'rsi_14',
        'rsi_divergence',
        'bars_since_extreme',
        'spread_zscore',
        'quote_rate_zscore',
        'hour_sin',
        'hour_cos',
    ]
