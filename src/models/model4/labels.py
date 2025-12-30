"""
Labeling for Mean Reversion Model

The model predicts: "Given price is stretched from VWAP, will it revert
to VWAP before hitting the stop loss?"

This is a CONDITIONAL prediction - we only create labels for bars where
a setup exists (price stretched > threshold from VWAP).
"""
import numpy as np
import pandas as pd


def add_reversion_labels(
    df: pd.DataFrame,
    zscore_threshold: float = 2.0,
    stop_atr_mult: float = 1.5,
    max_bars: int = 30,
    atr_col: str = 'atr_14'
) -> pd.DataFrame:
    """
    Create labels for mean reversion outcomes.

    Label = 1: Price reverted to VWAP before hitting stop
    Label = 0: Price hit stop before reverting to VWAP
    Label = NaN: No setup (price not stretched) - filtered during training

    Parameters:
    -----------
    df : pd.DataFrame
        Must have 'vwap_zscore', 'vwap', 'close', 'high', 'low', and ATR
    zscore_threshold : float
        Minimum z-score to qualify as a setup
    stop_atr_mult : float
        Stop distance as multiple of ATR
    max_bars : int
        Maximum bars to hold trade

    Returns:
    --------
    DataFrame with 'y_reversion' and 'setup_direction' columns
    """
    df = df.copy()

    n = len(df)
    labels = np.full(n, np.nan)  # NaN = no setup

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    vwap = df['vwap'].values
    zscore = df['vwap_zscore'].values
    atr = df[atr_col].values

    for i in range(n - max_bars):
        z = zscore[i]

        # Check if setup exists
        if abs(z) < zscore_threshold:
            continue  # No setup

        entry_price = close[i]
        target = vwap[i]  # Target is VWAP
        stop_distance = stop_atr_mult * atr[i]

        if z > zscore_threshold:
            # SHORT setup: price above VWAP
            stop = entry_price + stop_distance

            # Check future bars
            for j in range(i + 1, min(i + max_bars + 1, n)):
                # Check stop hit (price went higher)
                if high[j] >= stop:
                    labels[i] = 0  # Stop hit - loss
                    break
                # Check target hit (price reached VWAP)
                if low[j] <= target:
                    labels[i] = 1  # Target hit - win
                    break
            else:
                # Time stop - consider it a loss (didn't revert in time)
                labels[i] = 0

        elif z < -zscore_threshold:
            # LONG setup: price below VWAP
            stop = entry_price - stop_distance

            for j in range(i + 1, min(i + max_bars + 1, n)):
                # Check stop hit (price went lower)
                if low[j] <= stop:
                    labels[i] = 0  # Stop hit - loss
                    break
                # Check target hit (price reached VWAP)
                if high[j] >= target:
                    labels[i] = 1  # Target hit - win
                    break
            else:
                labels[i] = 0

    df['y_reversion'] = labels

    # Also store setup direction for analysis
    df['setup_direction'] = np.where(
        df['vwap_zscore'] > zscore_threshold, -1,  # SHORT setup
        np.where(df['vwap_zscore'] < -zscore_threshold, 1, 0)  # LONG setup
    )

    return df


def analyze_label_distribution(df: pd.DataFrame) -> dict:
    """Analyze the distribution of reversion labels."""

    # Only look at rows with setups
    setups = df[df['y_reversion'].notna()]

    total_setups = len(setups)
    if total_setups == 0:
        return {
            'total_setups': 0,
            'win_rate': 0,
            'wins': 0,
            'losses': 0,
            'long_setups': 0,
            'long_win_rate': 0,
            'short_setups': 0,
            'short_win_rate': 0,
        }

    wins = (setups['y_reversion'] == 1).sum()
    losses = (setups['y_reversion'] == 0).sum()

    # By direction
    long_setups = setups[setups['setup_direction'] == 1]
    short_setups = setups[setups['setup_direction'] == -1]

    return {
        'total_setups': total_setups,
        'win_rate': wins / total_setups if total_setups > 0 else 0,
        'wins': int(wins),
        'losses': int(losses),
        'long_setups': len(long_setups),
        'long_win_rate': float(long_setups['y_reversion'].mean()) if len(long_setups) > 0 else 0,
        'short_setups': len(short_setups),
        'short_win_rate': float(short_setups['y_reversion'].mean()) if len(short_setups) > 0 else 0,
    }
