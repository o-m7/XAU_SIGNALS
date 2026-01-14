#!/usr/bin/env python3
"""
Labels v2 - Validated Labeling System

Creates labels with verification that they actually predict price movement correctly.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def calculate_future_return(df: pd.DataFrame, horizon: int = 30) -> pd.Series:
    """
    Calculate actual future return over horizon bars.
    
    Args:
        df: DataFrame with 'close' column
        horizon: Number of bars to look ahead
    
    Returns:
        Series with future returns (positive = up, negative = down)
    """
    future_close = df['close'].shift(-horizon)
    returns = (future_close - df['close']) / df['close']
    return returns


def create_direction_labels(
    df: pd.DataFrame,
    horizon: int = 30,
    min_move_pct: float = 0.0005  # 0.05% minimum move
) -> pd.Series:
    """
    Create simple directional labels based on future price movement.
    
    Labels:
        1 = Price goes UP by at least min_move_pct
       -1 = Price goes DOWN by at least min_move_pct
        0 = Price doesn't move enough (neutral)
    
    Args:
        df: DataFrame with OHLC data
        horizon: Bars to look ahead
        min_move_pct: Minimum move to classify as up/down
    
    Returns:
        Series with labels
    """
    future_return = calculate_future_return(df, horizon)
    
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[future_return > min_move_pct] = 1
    labels[future_return < -min_move_pct] = -1
    
    return labels


def create_triple_barrier_labels(
    df: pd.DataFrame,
    horizon: int = 30,
    tp_mult: float = 1.0,
    sl_mult: float = 1.0,
    atr_column: str = 'atr_14'
) -> pd.Series:
    """
    Create triple barrier labels.
    
    Labels:
        1 = Take profit hit first (UP)
       -1 = Stop loss hit first (DOWN)
        0 = Neither hit within horizon (NEUTRAL)
    
    Args:
        df: DataFrame with OHLC and ATR
        horizon: Maximum bars to hold
        tp_mult: Take profit in ATR multiples
        sl_mult: Stop loss in ATR multiples
        atr_column: Name of ATR column
    
    Returns:
        Series with labels
    """
    # Get or calculate ATR
    if atr_column in df.columns:
        atr = df[atr_column].copy()
    elif 'ATR_14' in df.columns:
        atr = df['ATR_14'].copy()
    else:
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
    
    # Ensure positive ATR
    min_atr = df['close'] * 0.001
    atr = atr.clip(lower=min_atr)
    
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    atr_vals = atr.values
    
    n = len(df)
    labels = np.zeros(n, dtype=int)
    
    for i in range(n - horizon):
        entry = close[i]
        current_atr = atr_vals[i]
        
        if np.isnan(current_atr) or current_atr <= 0:
            continue
        
        tp_price = entry + (tp_mult * current_atr)
        sl_price = entry - (sl_mult * current_atr)
        
        tp_bar = None
        sl_bar = None
        
        # Check future bars
        for j in range(1, horizon + 1):
            if i + j >= n:
                break
            
            bar_high = high[i + j]
            bar_low = low[i + j]
            
            if tp_bar is None and bar_high >= tp_price:
                tp_bar = j
            if sl_bar is None and bar_low <= sl_price:
                sl_bar = j
            
            if tp_bar is not None and sl_bar is not None:
                break
        
        # Determine label
        if tp_bar is not None and sl_bar is None:
            labels[i] = 1  # UP
        elif sl_bar is not None and tp_bar is None:
            labels[i] = -1  # DOWN
        elif tp_bar is not None and sl_bar is not None:
            if tp_bar < sl_bar:
                labels[i] = 1  # TP hit first
            else:
                labels[i] = -1  # SL hit first
        # else: 0 (neither hit)
    
    return pd.Series(labels, index=df.index, name=f'y_tb_{horizon}')


def validate_labels(
    df: pd.DataFrame,
    labels: pd.Series,
    horizon: int = 30,
    sample_size: int = 1000
) -> dict:
    """
    Validate that labels actually correspond to correct price movement.
    
    Args:
        df: DataFrame with price data
        labels: Series with labels (1, -1, 0)
        horizon: Bars to check ahead
        sample_size: Number of samples to verify
    
    Returns:
        Dict with validation results
    """
    results = {
        'label_1_correct': 0,
        'label_1_total': 0,
        'label_neg1_correct': 0,
        'label_neg1_total': 0,
        'label_1_accuracy': 0.0,
        'label_neg1_accuracy': 0.0,
        'overall_accuracy': 0.0,
        'is_valid': False
    }
    
    # Sample indices for each label
    label_1_idx = labels[labels == 1].index[:sample_size]
    label_neg1_idx = labels[labels == -1].index[:sample_size]
    
    # Check label = 1 (should be UP)
    for idx in label_1_idx:
        loc = df.index.get_loc(idx)
        if loc + horizon < len(df):
            future_close = df.iloc[loc + horizon]['close']
            current_close = df.iloc[loc]['close']
            if future_close > current_close:
                results['label_1_correct'] += 1
            results['label_1_total'] += 1
    
    # Check label = -1 (should be DOWN)
    for idx in label_neg1_idx:
        loc = df.index.get_loc(idx)
        if loc + horizon < len(df):
            future_close = df.iloc[loc + horizon]['close']
            current_close = df.iloc[loc]['close']
            if future_close < current_close:
                results['label_neg1_correct'] += 1
            results['label_neg1_total'] += 1
    
    # Calculate accuracies
    if results['label_1_total'] > 0:
        results['label_1_accuracy'] = results['label_1_correct'] / results['label_1_total']
    if results['label_neg1_total'] > 0:
        results['label_neg1_accuracy'] = results['label_neg1_correct'] / results['label_neg1_total']
    
    total_correct = results['label_1_correct'] + results['label_neg1_correct']
    total_samples = results['label_1_total'] + results['label_neg1_total']
    if total_samples > 0:
        results['overall_accuracy'] = total_correct / total_samples
    
    # Labels are valid if both have >50% accuracy (better than random)
    results['is_valid'] = (
        results['label_1_accuracy'] > 0.50 and
        results['label_neg1_accuracy'] > 0.50
    )
    
    return results


def create_validated_labels(
    df: pd.DataFrame,
    horizon: int = 30,
    method: str = 'direction',  # 'direction' or 'triple_barrier'
    verbose: bool = True
) -> Tuple[pd.Series, dict]:
    """
    Create labels and validate them.
    
    Args:
        df: DataFrame with OHLC data
        horizon: Bars to look ahead
        method: Labeling method
        verbose: Print validation results
    
    Returns:
        Tuple of (labels Series, validation dict)
    """
    if verbose:
        print(f"\n[Creating {method} labels with horizon={horizon}]")
    
    # Create labels
    if method == 'direction':
        labels = create_direction_labels(df, horizon)
    elif method == 'triple_barrier':
        labels = create_triple_barrier_labels(df, horizon)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Print distribution
    if verbose:
        dist = labels.value_counts(normalize=True).sort_index()
        print(f"  Label distribution:")
        print(f"    DOWN (-1): {dist.get(-1, 0):.1%}")
        print(f"    NEUTRAL (0): {dist.get(0, 0):.1%}")
        print(f"    UP (+1): {dist.get(1, 0):.1%}")
    
    # Validate
    validation = validate_labels(df, labels, horizon)
    
    if verbose:
        print(f"  Validation results:")
        print(f"    Label=1 accuracy: {validation['label_1_accuracy']:.1%} ({validation['label_1_correct']}/{validation['label_1_total']})")
        print(f"    Label=-1 accuracy: {validation['label_neg1_accuracy']:.1%} ({validation['label_neg1_correct']}/{validation['label_neg1_total']})")
        print(f"    Overall accuracy: {validation['overall_accuracy']:.1%}")
        print(f"    Labels valid: {'YES' if validation['is_valid'] else 'NO'}")
    
    return labels, validation


def create_binary_labels_for_training(
    labels: pd.Series,
    direction: str = 'both'  # 'both', 'long_only', 'short_only'
) -> Tuple[pd.Series, pd.Series]:
    """
    Convert ternary labels to binary for ML training.
    
    Args:
        labels: Series with -1, 0, 1 labels
        direction: Which signals to train on
    
    Returns:
        Tuple of (X indices mask, y binary labels)
    """
    if direction == 'both':
        # Use all non-neutral labels
        mask = labels != 0
        y = (labels[mask] == 1).astype(int)
    elif direction == 'long_only':
        # Only UP vs NEUTRAL (for long-only model)
        mask = labels >= 0
        y = (labels[mask] == 1).astype(int)
    elif direction == 'short_only':
        # Only DOWN vs NEUTRAL (for short-only model)
        mask = labels <= 0
        y = (labels[mask] == -1).astype(int)
    else:
        raise ValueError(f"Unknown direction: {direction}")
    
    return mask, y


if __name__ == "__main__":
    print("Labels v2 - Test")
    
    # Create sample data
    np.random.seed(42)
    n = 10000
    dates = pd.date_range('2024-01-01', periods=n, freq='1min')
    
    # Create trending data (easier to label)
    trend = np.cumsum(np.random.randn(n) * 0.5 + 0.01)  # Slight upward bias
    close = 2000 + trend
    
    df = pd.DataFrame({
        'open': close + np.random.randn(n) * 0.1,
        'high': close + np.abs(np.random.randn(n)) * 0.5,
        'low': close - np.abs(np.random.randn(n)) * 0.5,
        'close': close,
        'volume': np.random.randint(100, 1000, n)
    }, index=dates)
    
    # Test direction labels
    labels, validation = create_validated_labels(df, horizon=30, method='direction')
    
    # Test triple barrier
    labels_tb, validation_tb = create_validated_labels(df, horizon=30, method='triple_barrier')

