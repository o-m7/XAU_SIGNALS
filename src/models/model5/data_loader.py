"""
data_loader.py - Load and Resample Data

Loads minute aggregates and resamples to 15-minute bars.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def load_minute_data(file_path: str) -> pd.DataFrame:
    """
    Load minute aggregate data from parquet file.
    
    Args:
        file_path: Path to parquet file with minute OHLCV data
    
    Returns:
        DataFrame with DatetimeIndex and OHLCV columns
    """
    df = pd.read_parquet(file_path)
    
    # Ensure timestamp index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Data must have timestamp index or 'timestamp' column")
    
    # Ensure required columns
    required = ['open', 'high', 'low', 'close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Sort by index
    df = df.sort_index()
    
    return df


def resample_to_15m(df_minute: pd.DataFrame) -> pd.DataFrame:
    """
    Resample minute data to 15-minute bars.
    
    Args:
        df_minute: DataFrame with minute OHLCV data
    
    Returns:
        DataFrame with 15-minute OHLCV bars
    """
    df = df_minute.copy()
    
    # Resample OHLCV
    df_15m = pd.DataFrame()
    df_15m['open'] = df['open'].resample('15T').first()
    df_15m['high'] = df['high'].resample('15T').max()
    df_15m['low'] = df['low'].resample('15T').min()
    df_15m['close'] = df['close'].resample('15T').last()
    
    # Volume (sum)
    if 'volume' in df.columns:
        df_15m['volume'] = df['volume'].resample('15T').sum()
    
    # VWAP (volume-weighted average)
    if 'vwap' in df.columns and 'volume' in df.columns:
        df['pv'] = df['vwap'] * df['volume']
        pv_sum = df['pv'].resample('15T').sum()
        vol_sum = df['volume'].resample('15T').sum()
        df_15m['vwap'] = pv_sum / vol_sum.replace(0, np.nan)
    
    # Transactions (sum)
    if 'transactions' in df.columns:
        df_15m['transactions'] = df['transactions'].resample('15T').sum()
    
    # Drop rows with missing OHLC
    df_15m = df_15m.dropna(subset=['open', 'high', 'low', 'close'])
    
    return df_15m


def load_quote_data(file_path: str) -> pd.DataFrame:
    """
    Load quote data from parquet file.
    
    Args:
        file_path: Path to parquet file with quote data
    
    Returns:
        DataFrame with quote data
    """
    df = pd.read_parquet(file_path)
    
    # Handle timestamp
    if 'participant_timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['participant_timestamp'], unit='ns')
        df = df.set_index('timestamp')
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Quote data must have timestamp")
    
    # Ensure required columns
    required = ['bid_price', 'ask_price']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    df = df.sort_index()
    
    return df


def load_second_data(file_path: str) -> pd.DataFrame:
    """
    Load second aggregate data from parquet file.
    
    Args:
        file_path: Path to parquet file with second OHLCV data
    
    Returns:
        DataFrame with second OHLCV data
    """
    return load_minute_data(file_path)  # Same structure


def load_all_data(
    minute_path: str,
    quote_path: Optional[str] = None,
    second_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Load all data sources.
    
    Args:
        minute_path: Path to minute aggregate parquet
        quote_path: Optional path to quote data
        second_path: Optional path to second aggregate data
    
    Returns:
        Tuple of (df_15m, df_quotes, df_seconds)
    """
    # Load and resample minute data
    df_minute = load_minute_data(minute_path)
    df_15m = resample_to_15m(df_minute)
    
    # Load quotes if provided
    df_quotes = None
    if quote_path and Path(quote_path).exists():
        df_quotes = load_quote_data(quote_path)
    
    # Load second data if provided
    df_seconds = None
    if second_path and Path(second_path).exists():
        df_seconds = load_second_data(second_path)
    
    return df_15m, df_quotes, df_seconds

