"""
Labeling for Model #3: CMF and MACD Strategy

Uses triple-barrier labeling similar to Model #1.
"""

import pandas as pd
import numpy as np
from typing import Optional


def add_triple_barrier_labels(
    df: pd.DataFrame,
    h_max: int = 60,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0
) -> pd.DataFrame:
    """
    Add triple-barrier labels for Model #3.
    
    Args:
        df: DataFrame with OHLCV and ATR
        h_max: Maximum horizon in bars
        tp_mult: Take-profit multiplier (ATR)
        sl_mult: Stop-loss multiplier (ATR)
        
    Returns:
        DataFrame with y_tb_60 label added
    """
    df = df.copy()
    
    # Ensure ATR exists
    if 'atr' not in df.columns:
        # Calculate ATR if not present
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(14, min_periods=1).mean()
    
    close = df['close'].values
    atr = df['atr'].values
    n = len(df)
    
    labels = np.zeros(n)
    
    for t in range(n - 1):
        if np.isnan(close[t]) or np.isnan(atr[t]) or atr[t] <= 0:
            labels[t] = np.nan
            continue
        
        # Compute barriers
        upper = close[t] * (1 + tp_mult * atr[t] / close[t])
        lower = close[t] * (1 - sl_mult * atr[t] / close[t])
        
        # Walk forward
        label = 0
        for h in range(1, min(h_max + 1, n - t)):
            future_close = close[t + h]
            
            if np.isnan(future_close):
                continue
            
            # Check barriers
            if future_close >= upper:
                label = 1  # Upper barrier hit first
                break
            elif future_close <= lower:
                label = -1  # Lower barrier hit first
                break
        
        labels[t] = label
    
    # Last h_max bars don't have enough future data
    labels[n - h_max:] = np.nan
    
    df['y_tb_60'] = labels
    
    return df

