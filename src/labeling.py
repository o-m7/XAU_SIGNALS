#!/usr/bin/env python3
"""
Improved Labeling System for Trading Models

Creates high-quality labels using triple barrier method with:
- Short horizons (15-30 bars max)
- Positive risk/reward ratio (1:1.5)
- Proper ATR validation
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range with proper validation.
    
    Args:
        df: DataFrame with high, low, close columns
        period: ATR period (default 14)
    
    Returns:
        ATR series with minimum floor to avoid zero/negative values
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean()
    
    # Floor ATR at 0.1% of price to avoid zero/negative
    min_atr = close * 0.001
    atr = atr.clip(lower=min_atr)
    
    return atr


def create_triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int = 15,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    atr_column: str = 'atr_14'
) -> pd.Series:
    """
    Create triple barrier labels with improved parameters.
    
    Args:
        df: DataFrame with OHLC and ATR data
        horizon: Maximum bars to hold (15 or 30 recommended)
        tp_mult: Take profit multiplier (ATR units)
        sl_mult: Stop loss multiplier (ATR units)
        atr_column: Name of ATR column (or will calculate)
    
    Returns:
        Series with labels: +1 (TP hit first), -1 (SL hit first), 0 (neither)
    """
    if atr_column not in df.columns:
        atr = calculate_atr(df)
    else:
        atr = df[atr_column].copy()
        # Ensure positive ATR
        min_atr = df['close'] * 0.001
        atr = atr.clip(lower=min_atr)
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    atr_vals = atr.values
    
    n = len(df)
    labels = np.zeros(n)
    
    for i in range(n - horizon):
        entry_price = close[i]
        current_atr = atr_vals[i]
        
        if np.isnan(current_atr) or current_atr <= 0:
            labels[i] = 0
            continue
        
        tp_price = entry_price + (tp_mult * current_atr)
        sl_price = entry_price - (sl_mult * current_atr)
        
        # Look forward up to horizon bars
        tp_hit_bar = None
        sl_hit_bar = None
        
        for j in range(1, horizon + 1):
            if i + j >= n:
                break
            
            bar_high = high[i + j]
            bar_low = low[i + j]
            
            # Check if TP hit
            if bar_high >= tp_price and tp_hit_bar is None:
                tp_hit_bar = j
            
            # Check if SL hit
            if bar_low <= sl_price and sl_hit_bar is None:
                sl_hit_bar = j
            
            # If both hit, we already know which was first
            if tp_hit_bar is not None and sl_hit_bar is not None:
                break
        
        # Determine label
        if tp_hit_bar is not None and sl_hit_bar is None:
            labels[i] = 1  # TP hit, SL not hit
        elif sl_hit_bar is not None and tp_hit_bar is None:
            labels[i] = -1  # SL hit, TP not hit
        elif tp_hit_bar is not None and sl_hit_bar is not None:
            # Both hit - which was first?
            if tp_hit_bar < sl_hit_bar:
                labels[i] = 1
            elif sl_hit_bar < tp_hit_bar:
                labels[i] = -1
            else:
                # Same bar - assume stop hit first (conservative)
                labels[i] = -1
        else:
            labels[i] = 0  # Neither hit within horizon
    
    return pd.Series(labels, index=df.index, name=f'y_tb_{horizon}')


def create_directional_labels(
    df: pd.DataFrame,
    horizon: int = 15,
    min_move_pct: float = 0.001
) -> pd.Series:
    """
    Create simple directional labels based on future returns.
    
    Args:
        df: DataFrame with close prices
        horizon: Bars to look ahead
        min_move_pct: Minimum move to classify (default 0.1%)
    
    Returns:
        Series with labels: +1 (up), -1 (down), 0 (neutral)
    """
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    
    labels = pd.Series(0, index=df.index)
    labels[future_return > min_move_pct] = 1
    labels[future_return < -min_move_pct] = -1
    
    return labels


def add_all_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all label columns to dataframe.
    
    Adds:
    - y_tb_15: 15-bar triple barrier (main)
    - y_tb_30: 30-bar triple barrier (alternative)
    - y_dir_15: 15-bar directional
    """
    df = df.copy()
    
    # Ensure ATR exists
    if 'atr_14' not in df.columns:
        df['atr_14'] = calculate_atr(df)
    
    # Triple barrier labels (recommended)
    print("  Creating y_tb_15 labels (15-bar triple barrier)...")
    df['y_tb_15'] = create_triple_barrier_labels(df, horizon=15, tp_mult=1.5, sl_mult=1.0)
    
    print("  Creating y_tb_30 labels (30-bar triple barrier)...")
    df['y_tb_30'] = create_triple_barrier_labels(df, horizon=30, tp_mult=1.5, sl_mult=1.0)
    
    # Directional labels (simpler alternative)
    print("  Creating y_dir_15 labels (15-bar directional)...")
    df['y_dir_15'] = create_directional_labels(df, horizon=15)
    
    # Print label distribution
    for col in ['y_tb_15', 'y_tb_30', 'y_dir_15']:
        dist = df[col].value_counts(normalize=True).sort_index()
        print(f"  {col}: up={dist.get(1, 0):.1%}, down={dist.get(-1, 0):.1%}, neutral={dist.get(0, 0):.1%}")
    
    return df


if __name__ == "__main__":
    # Test with sample data
    print("Testing labeling module...")
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2024-01-01', periods=n, freq='1min')
    close = 2000 + np.cumsum(np.random.randn(n) * 0.5)
    
    df = pd.DataFrame({
        'open': close + np.random.randn(n) * 0.1,
        'high': close + np.abs(np.random.randn(n)) * 0.5,
        'low': close - np.abs(np.random.randn(n)) * 0.5,
        'close': close,
        'volume': np.random.randint(100, 1000, n)
    }, index=dates)
    
    df = add_all_labels(df)
    print("\nLabeling complete!")
    print(f"Shape: {df.shape}")
