"""
Time utility functions for the XAUUSD Signal Engine.

Provides helpers for timestamp handling, timezone conversion,
and time-based calculations.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union
from datetime import datetime, timedelta


def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame index is a UTC-aware DatetimeIndex.
    
    Args:
        df: DataFrame with a timestamp index or column
        
    Returns:
        DataFrame with UTC-aware DatetimeIndex
        
    Raises:
        ValueError: If no valid timestamp found
    """
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        elif str(df.index.tz) != "UTC":
            df.index = df.index.tz_convert("UTC")
    elif "timestamp" in df.columns:
        df = df.set_index("timestamp")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        elif str(df.index.tz) != "UTC":
            df.index = df.index.tz_convert("UTC")
    else:
        raise ValueError("DataFrame must have a DatetimeIndex or 'timestamp' column")
    
    return df


def get_minute_of_day(timestamp: pd.Timestamp) -> int:
    """
    Get the minute of day (0-1439) for a timestamp.
    
    Args:
        timestamp: A pandas Timestamp
        
    Returns:
        Integer in range [0, 1439]
    """
    return timestamp.hour * 60 + timestamp.minute


def get_trading_session(timestamp: pd.Timestamp) -> str:
    """
    Determine the trading session for a given timestamp.
    
    Sessions (UTC):
        - Asian: 00:00 - 08:00
        - European: 08:00 - 16:00
        - American: 16:00 - 24:00
    
    Args:
        timestamp: A pandas Timestamp
        
    Returns:
        Session name: "asian", "european", or "american"
    """
    hour = timestamp.hour
    
    if 0 <= hour < 8:
        return "asian"
    elif 8 <= hour < 16:
        return "european"
    else:
        return "american"


def is_weekend(timestamp: pd.Timestamp) -> bool:
    """
    Check if timestamp falls on a weekend.
    
    Note: Forex markets close Friday ~22:00 UTC and reopen Sunday ~22:00 UTC
    
    Args:
        timestamp: A pandas Timestamp
        
    Returns:
        True if weekend, False otherwise
    """
    return timestamp.dayofweek >= 5


def get_time_to_market_close(
    timestamp: pd.Timestamp,
    friday_close_hour: int = 22
) -> Optional[int]:
    """
    Get minutes until Friday market close.
    
    Args:
        timestamp: Current timestamp
        friday_close_hour: Hour when market closes on Friday (UTC)
        
    Returns:
        Minutes to close, or None if not a Friday
    """
    if timestamp.dayofweek != 4:  # Not Friday
        return None
    
    close_time = timestamp.replace(hour=friday_close_hour, minute=0, second=0)
    
    if timestamp >= close_time:
        return 0
    
    delta = close_time - timestamp
    return int(delta.total_seconds() / 60)


def find_timestamp_position(
    df: pd.DataFrame,
    target_timestamp: pd.Timestamp,
    direction: str = "backward"
) -> Optional[int]:
    """
    Find the index position for a target timestamp.
    
    Args:
        df: DataFrame with DatetimeIndex
        target_timestamp: Timestamp to find
        direction: "backward" for last timestamp <= target,
                   "forward" for first timestamp >= target
                   
    Returns:
        Index position (iloc), or None if not found
    """
    if direction == "backward":
        mask = df.index <= target_timestamp
        if mask.any():
            return mask.sum() - 1
    else:
        mask = df.index >= target_timestamp
        if mask.any():
            return (~mask).sum()
    
    return None


def get_forward_window_slice(
    df: pd.DataFrame,
    start_idx: int,
    horizon_minutes: int
) -> slice:
    """
    Get a slice for the forward window from a starting index.
    
    Args:
        df: DataFrame with DatetimeIndex (minute frequency assumed)
        start_idx: Starting index position
        horizon_minutes: Number of minutes to look forward
        
    Returns:
        A slice object (start_idx + 1, end_idx) exclusive
        
    Note:
        Returns slice starting AFTER start_idx (exclusive of current bar)
        If horizon extends beyond data, end is clamped to len(df)
    """
    start_time = df.index[start_idx]
    end_time = start_time + pd.Timedelta(minutes=horizon_minutes)
    
    # Find end index
    end_idx = start_idx + 1
    while end_idx < len(df) and df.index[end_idx] <= end_time:
        end_idx += 1
    
    return slice(start_idx + 1, end_idx)


def resample_to_higher_timeframe(
    df: pd.DataFrame,
    target_freq: str = "5min"
) -> pd.DataFrame:
    """
    Resample minute data to higher timeframe OHLCV.
    
    Args:
        df: Minute-level DataFrame with OHLCV columns
        target_freq: Target frequency (e.g., "5min", "15min", "1H")
        
    Returns:
        Resampled DataFrame
        
    Assumptions:
        df has columns: open, high, low, close, volume
    """
    resampled = df.resample(target_freq).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    
    return resampled


def compute_time_since_reference(
    df: pd.DataFrame,
    reference_times: pd.DatetimeIndex
) -> pd.Series:
    """
    Compute minutes since the last reference time for each row.
    
    Args:
        df: DataFrame with DatetimeIndex
        reference_times: Index of reference timestamps
        
    Returns:
        Series of minutes since last reference time
    """
    result = pd.Series(index=df.index, dtype=float)
    
    for i, ts in enumerate(df.index):
        # Find most recent reference time
        past_refs = reference_times[reference_times <= ts]
        if len(past_refs) > 0:
            delta = ts - past_refs[-1]
            result.iloc[i] = delta.total_seconds() / 60
        else:
            result.iloc[i] = np.nan
    
    return result

