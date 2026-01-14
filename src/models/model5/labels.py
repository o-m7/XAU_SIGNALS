"""
labels.py - Target Variable Creation

Labels for supervised learning and backtesting.
"""

import numpy as np
import pandas as pd
from typing import Tuple

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func=None, *args, **kwargs):
        if func is None:
            def decorator(f):
                return f
            return decorator
        return func

from .config import Model5Config


@njit
def label_mean_reversion_outcomes(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    zscore: np.ndarray,
    atr: np.ndarray,
    zscore_threshold: float = 2.0,
    target_zscore: float = 0.0,
    stop_atr_mult: float = 1.5,
    max_bars: int = 15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create labels for mean reversion trades.
    
    LONG setup: zscore < -threshold
    Target: price returns to mean (zscore = 0)
    Stop: 1.5 ATR below entry
    
    Returns:
        labels: 1 = win (target hit), 0 = loss (stop or time)
        pnl: Actual PnL in price units
        bars_held: Bars until exit
        direction: 1 = LONG, -1 = SHORT, 0 = no setup
    """
    n = len(close)
    labels = np.full(n, np.nan)
    pnl = np.full(n, np.nan)
    bars_held = np.full(n, np.nan)
    direction = np.zeros(n)
    
    for i in range(n - max_bars):
        z = zscore[i]
        
        # Skip if not a setup or invalid data
        if np.isnan(z) or np.isnan(atr[i]) or atr[i] == 0:
            continue
        
        if abs(z) < zscore_threshold:
            continue
        
        entry_price = close[i]
        
        # LONG setup (oversold)
        if z < -zscore_threshold:
            direction[i] = 1
            stop_price = entry_price - (stop_atr_mult * atr[i])
            
            # Target: where zscore would be ~0
            # Approximate: target = entry + |z| * std
            # Since z = (price - mean) / std, mean = price - z*std
            # So target (mean) = entry - z*std ≈ entry + |z|*atr (rough approximation)
            # Better: just use a fixed ATR target that approximates mean reversion
            target_price = entry_price + (abs(z) * 0.5 * atr[i])  # Partial reversion
            
            # Simulate trade
            for j in range(i + 1, min(i + max_bars + 1, n)):
                # Stop hit
                if low[j] <= stop_price:
                    labels[i] = 0
                    pnl[i] = stop_price - entry_price
                    bars_held[i] = j - i
                    break
                
                # Target hit
                if high[j] >= target_price:
                    labels[i] = 1
                    pnl[i] = target_price - entry_price
                    bars_held[i] = j - i
                    break
            
            # Time stop
            if np.isnan(labels[i]):
                final_price = close[min(i + max_bars, n - 1)]
                pnl[i] = final_price - entry_price
                labels[i] = 1 if pnl[i] > 0 else 0
                bars_held[i] = max_bars
    
    return labels, pnl, bars_held, direction


def add_labels(df: pd.DataFrame, config=None) -> pd.DataFrame:
    """
    Add trade outcome labels to DataFrame.
    """
    if config is None:
        config = Model5Config()
    
    labels, pnl, bars_held, direction = label_mean_reversion_outcomes(
        df['close'].values,
        df['high'].values,
        df['low'].values,
        df['zscore_20'].values,
        df['atr_14'].values,
        zscore_threshold=config.zscore_entry_threshold,
        stop_atr_mult=config.stop_atr_multiple,
        max_bars=config.max_bars_in_trade
    )
    
    df['y'] = labels
    df['trade_pnl'] = pnl
    df['bars_held'] = bars_held
    df['direction'] = direction
    
    # Stats
    valid = df[df['y'].notna()]
    if len(valid) > 0:
        wins = (valid['y'] == 1).sum()
        total = len(valid)
        print(f"Labels: {total} setups, {wins} wins ({100*wins/total:.1f}%)")
        print(f"Avg PnL: ${valid['trade_pnl'].mean():.2f}")
        print(f"Avg bars held: {valid['bars_held'].mean():.1f}")
    
    return df

