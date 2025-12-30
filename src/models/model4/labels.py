"""
Model 4 Labeling
Simple directional labels - NO triple-barrier
"""
import numpy as np
import pandas as pd


def add_directional_labels(
    df: pd.DataFrame,
    horizon: int = 12,
    threshold_atr_mult: float = 0.5,
    atr_col: str = 'atr_14'
) -> pd.DataFrame:
    """
    Create directional labels based on future price movement.

    Labels:
        +1: Price moves UP more than threshold (good long entry)
        -1: Price moves DOWN more than threshold (good short entry)
         0: Neither (no clear direction - filter during training)

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC and ATR
    horizon : int
        Number of bars to look ahead
    threshold_atr_mult : float
        Required move as multiple of ATR
    atr_col : str
        Name of ATR column

    Returns:
    --------
    DataFrame with 'y_direction' column
    """
    df = df.copy()

    # Future high and low within horizon
    future_high = df['high'].rolling(horizon).max().shift(-horizon)
    future_low = df['low'].rolling(horizon).min().shift(-horizon)

    # Maximum up and down moves from current close
    max_up_move = future_high - df['close']
    max_down_move = df['close'] - future_low

    # Threshold based on ATR
    threshold = threshold_atr_mult * df[atr_col]

    # Label logic
    labels = np.zeros(len(df))

    # Up move is larger AND exceeds threshold
    up_condition = (max_up_move > max_down_move) & (max_up_move > threshold)

    # Down move is larger AND exceeds threshold
    down_condition = (max_down_move > max_up_move) & (max_down_move > threshold)

    labels[up_condition] = 1
    labels[down_condition] = -1

    df['y_direction'] = labels.astype(int)

    # Also store the move magnitudes for analysis
    df['future_up_move'] = max_up_move
    df['future_down_move'] = max_down_move

    return df


def add_trend_aligned_labels(
    df: pd.DataFrame,
    horizon: int = 12,
    threshold_atr_mult: float = 0.5,
    atr_col: str = 'atr_14',
    trend_col: str = 'trend'
) -> pd.DataFrame:
    """
    Create labels for "good entry in trend direction".

    This is what Model 4 actually predicts:
        1: Good entry point (price moves in trend direction)
        0: Bad entry point (price moves against trend or insufficient move)

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC, ATR, and trend column
    horizon : int
        Number of bars to look ahead
    threshold_atr_mult : float
        Required move as multiple of ATR
    atr_col : str
        Name of ATR column
    trend_col : str
        Name of trend column (1 for bullish, -1 for bearish)

    Returns:
    --------
    DataFrame with 'y_good_entry' column
    """
    df = df.copy()

    # First add directional labels
    df = add_directional_labels(df, horizon, threshold_atr_mult, atr_col)

    # Good entry = direction matches trend
    df['y_good_entry'] = (
        (df[trend_col] == df['y_direction']) &
        (df['y_direction'] != 0)
    ).astype(int)

    return df
