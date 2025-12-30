"""
Session VWAP Calculation

VWAP (Volume Weighted Average Price) represents fair value.
Mean reversion edge exists because price tends to revert to VWAP.
"""
import numpy as np
import pandas as pd


def calculate_session_vwap(
    df: pd.DataFrame,
    session_hours: int = 8,
    price_col: str = 'close',
    volume_col: str = 'volume'
) -> pd.DataFrame:
    """
    Calculate rolling session VWAP.

    VWAP = Cumulative(Price * Volume) / Cumulative(Volume)

    Parameters:
    -----------
    df : pd.DataFrame
        OHLCV data with DatetimeIndex
    session_hours : int
        Rolling window in hours

    Returns:
    --------
    DataFrame with VWAP columns added
    """
    df = df.copy()

    # Typical price (more stable than close)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

    # Calculate window size based on timeframe
    if hasattr(df.index, 'freq') and df.index.freq:
        freq_minutes = df.index.freq.delta.total_seconds() / 60
    else:
        # Infer from data
        if len(df) > 1:
            freq_minutes = (df.index[1] - df.index[0]).total_seconds() / 60
        else:
            freq_minutes = 5  # Default to 5 minutes

    window_bars = int(session_hours * 60 / freq_minutes)

    # Handle missing volume
    if volume_col not in df.columns or df[volume_col].isna().all():
        # Use equal weighting if no volume
        df['vwap'] = df['typical_price'].rolling(window_bars, min_periods=1).mean()
    else:
        # Rolling VWAP with volume weighting
        df['cum_vol_price'] = (df['typical_price'] * df[volume_col]).rolling(window_bars, min_periods=1).sum()
        df['cum_vol'] = df[volume_col].rolling(window_bars, min_periods=1).sum()
        df['vwap'] = df['cum_vol_price'] / df['cum_vol'].replace(0, np.nan)

        # Cleanup temporary columns
        df.drop(['cum_vol_price', 'cum_vol'], axis=1, inplace=True, errors='ignore')

    # Distance from VWAP
    df['vwap_distance'] = df['close'] - df['vwap']
    df['vwap_distance_pct'] = df['vwap_distance'] / df['vwap'].replace(0, np.nan)

    # Cleanup
    df.drop(['typical_price'], axis=1, inplace=True, errors='ignore')

    return df


def calculate_vwap_zscore(
    df: pd.DataFrame,
    atr_col: str = 'atr_14'
) -> pd.DataFrame:
    """
    Calculate z-score of distance from VWAP normalized by ATR.

    z_score = (close - VWAP) / ATR

    This tells us how many ATRs price is from fair value.
    Positive z-score = price above VWAP (potential short setup)
    Negative z-score = price below VWAP (potential long setup)
    """
    df = df.copy()

    # Ensure ATR column exists
    if atr_col not in df.columns:
        raise ValueError(f"ATR column '{atr_col}' not found in DataFrame")

    df['vwap_zscore'] = df['vwap_distance'] / df[atr_col].replace(0, np.nan)

    # Velocity of z-score (is it stretching further or reverting?)
    df['vwap_zscore_velocity'] = df['vwap_zscore'].diff(3)  # 3-bar change

    # Acceleration
    df['vwap_zscore_accel'] = df['vwap_zscore_velocity'].diff(2)

    return df
