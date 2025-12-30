"""
Feature Engineering for Model #3: CMF and MACD

Main features:
- Chaikin Money Flow (CMF)
- MACD (signal, histogram)
- Additional technical indicators
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def compute_chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Compute Chaikin Money Flow (CMF).
    
    CMF = Sum(Volume * Money Flow Multiplier) / Sum(Volume) over period
    Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
    
    CMF > 0: Buying pressure
    CMF < 0: Selling pressure
    CMF > 0.25: Strong buying
    CMF < -0.25: Strong selling
    
    Args:
        df: DataFrame with OHLCV data
        period: Period for CMF calculation (default 20)
        
    Returns:
        DataFrame with CMF features added
    """
    df = df.copy()
    
    # Money Flow Multiplier
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
    
    # CMF z-score (normalized)
    cmf_mean = df['cmf'].rolling(period * 2, min_periods=period).mean()
    cmf_std = df['cmf'].rolling(period * 2, min_periods=period).std()
    df['cmf_zscore'] = (df['cmf'] - cmf_mean) / (cmf_std + 1e-8)
    
    # CMF trend (slope)
    df['cmf_trend'] = df['cmf'].rolling(period, min_periods=period).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= period else 0,
        raw=False
    )
    
    # Fill NaNs
    df['cmf'] = df['cmf'].fillna(0)
    df['cmf_momentum'] = df['cmf_momentum'].fillna(0)
    df['cmf_zscore'] = df['cmf_zscore'].fillna(0)
    df['cmf_trend'] = df['cmf_trend'].fillna(0)
    
    return df


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Compute MACD (Moving Average Convergence Divergence).
    
    MACD = EMA(fast) - EMA(slow)
    Signal = EMA(MACD, signal_period)
    Histogram = MACD - Signal
    
    Args:
        df: DataFrame with close prices
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
    
    # Signal line
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    
    # Histogram
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # MACD momentum (rate of change)
    df['macd_momentum'] = df['macd'].diff()
    df['macd_signal_momentum'] = df['macd_signal'].diff()
    
    # MACD crossovers
    df['macd_above_signal'] = (df['macd'] > df['macd_signal']).astype(int)
    df['macd_cross_up'] = (
        (df['macd'] > df['macd_signal']) & 
        (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    ).astype(int)
    df['macd_cross_down'] = (
        (df['macd'] < df['macd_signal']) & 
        (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    ).astype(int)
    
    # MACD histogram momentum
    df['macd_hist_momentum'] = df['macd_histogram'].diff()
    
    # Normalized MACD (z-score)
    macd_mean = df['macd'].rolling(slow * 2, min_periods=slow).mean()
    macd_std = df['macd'].rolling(slow * 2, min_periods=slow).std()
    df['macd_zscore'] = (df['macd'] - macd_mean) / (macd_std + 1e-8)
    
    # Fill NaNs
    df['macd'] = df['macd'].fillna(0)
    df['macd_signal'] = df['macd_signal'].fillna(0)
    df['macd_histogram'] = df['macd_histogram'].fillna(0)
    df['macd_momentum'] = df['macd_momentum'].fillna(0)
    df['macd_zscore'] = df['macd_zscore'].fillna(0)
    
    return df


def compute_additional_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute additional technical indicators for confirmation.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional indicators
    """
    df = df.copy()
    
    # RSI (Relative Strength Index)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-8)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(14, min_periods=1).mean()
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(20, min_periods=1).mean()
    bb_std = df['close'].rolling(20, min_periods=1).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(20, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-8)
    
    # Price momentum
    df['price_momentum_5'] = df['close'].pct_change(5)
    df['price_momentum_20'] = df['close'].pct_change(20)
    
    # Moving averages
    df['sma_20'] = df['close'].rolling(20, min_periods=1).mean()
    df['sma_50'] = df['close'].rolling(50, min_periods=1).mean()
    df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    
    # Fill NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df


def build_cmf_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features for Model #3 (CMF and MACD based).
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with all Model #3 features
    """
    logger.info("Building Model #3 features (CMF and MACD)...")
    
    # CMF features
    df = compute_chaikin_money_flow(df, period=20)
    
    # MACD features
    df = compute_macd(df, fast=12, slow=26, signal=9)
    
    # Additional indicators
    df = compute_additional_indicators(df)
    
    logger.info(f"Built {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])} features")
    
    return df


def get_feature_columns_for_model3() -> list:
    """
    Get list of feature columns for Model #3.
    
    Returns:
        List of feature column names
    """
    return [
        # CMF features
        'cmf', 'cmf_momentum', 'cmf_zscore', 'cmf_trend',
        
        # MACD features
        'macd', 'macd_signal', 'macd_histogram',
        'macd_momentum', 'macd_signal_momentum',
        'macd_above_signal', 'macd_cross_up', 'macd_cross_down',
        'macd_hist_momentum', 'macd_zscore',
        
        # Additional indicators
        'rsi', 'atr',
        'bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
        'volume_ratio', 'price_momentum_5', 'price_momentum_20',
        'price_vs_sma20', 'price_vs_sma50',
    ]

