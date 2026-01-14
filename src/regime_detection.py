"""
Regime Detection Module

Detects market regimes to determine when shorting is appropriate.
Only allows shorting in bearish/downtrend regimes, not randomly on every slight high.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def detect_bearish_regime(row: pd.Series) -> bool:
    """
    Detect if current market is in a bearish regime suitable for shorting.
    
    Returns True only when multiple bearish indicators align:
    - Price below medium-term moving average (downtrend)
    - Negative momentum
    - High volatility (not just noise)
    - Negative order flow imbalance
    
    Args:
        row: Series with market features
        
    Returns:
        True if bearish regime detected, False otherwise
    """
    # Check if we have required features
    required_features = ['mid', 'sigma']
    if not all(f in row.index for f in required_features):
        return False
    
    # Initialize bearish score
    bearish_score = 0
    max_score = 0
    
    # 1. Price trend (30% weight)
    if 'momentum_30' in row.index and not pd.isna(row['momentum_30']):
        max_score += 3
        if row['momentum_30'] < -0.0001:  # Negative 30-min momentum
            bearish_score += 3
        elif row['momentum_30'] < 0:
            bearish_score += 1
    
    # 2. Medium-term trend (25% weight)
    if 'momentum_15' in row.index and not pd.isna(row['momentum_15']):
        max_score += 2.5
        if row['momentum_15'] < -0.0001:  # Negative 15-min momentum
            bearish_score += 2.5
        elif row['momentum_15'] < 0:
            bearish_score += 1
    
    # 3. Order flow imbalance (25% weight)
    if 'imbalance' in row.index and not pd.isna(row['imbalance']):
        max_score += 2.5
        if row['imbalance'] < -0.1:  # Strong sell pressure
            bearish_score += 2.5
        elif row['imbalance'] < -0.05:
            bearish_score += 1.5
        elif row['imbalance'] < 0:
            bearish_score += 0.5
    
    # 4. Volatility regime (20% weight) - need sufficient volatility for meaningful moves
    if 'sigma' in row.index and not pd.isna(row['sigma']):
        max_score += 2
        # Check if volatility is elevated (not just noise)
        if 'sigma_60' in row.index and not pd.isna(row['sigma_60']):
            vol_ratio = row['sigma'] / (row['sigma_60'] + 1e-8)
            if vol_ratio > 1.2:  # Elevated volatility
                bearish_score += 2
            elif vol_ratio > 1.0:
                bearish_score += 1
    
    # Require at least 60% of max score to confirm bearish regime
    if max_score == 0:
        return False
    
    bearish_ratio = bearish_score / max_score
    return bearish_ratio >= 0.60


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime detection features to dataframe.
    
    Features added:
    - is_bearish_regime: Boolean flag for bearish regime
    - regime_score: Continuous score (0-1) for bearishness
    """
    df = df.copy()
    
    # Initialize regime columns
    df['is_bearish_regime'] = False
    df['regime_score'] = 0.0
    
    # Compute regime for each row
    for idx in df.index:
        row = df.loc[idx]
        
        # Check bearish regime
        is_bearish = detect_bearish_regime(row)
        df.loc[idx, 'is_bearish_regime'] = is_bearish
        
        # Compute continuous regime score
        score = compute_regime_score(row)
        df.loc[idx, 'regime_score'] = score
    
    return df


def compute_regime_score(row: pd.Series) -> float:
    """
    Compute continuous regime score (0-1) where 1 = strongly bearish.
    
    Returns:
        Float between 0 and 1
    """
    score = 0.0
    max_score = 0.0
    
    # Momentum components
    if 'momentum_30' in row.index and not pd.isna(row['momentum_30']):
        max_score += 0.3
        if row['momentum_30'] < -0.0001:
            score += 0.3
        elif row['momentum_30'] < 0:
            score += 0.15
    
    if 'momentum_15' in row.index and not pd.isna(row['momentum_15']):
        max_score += 0.25
        if row['momentum_15'] < -0.0001:
            score += 0.25
        elif row['momentum_15'] < 0:
            score += 0.125
    
    # Order flow
    if 'imbalance' in row.index and not pd.isna(row['imbalance']):
        max_score += 0.25
        if row['imbalance'] < -0.1:
            score += 0.25
        elif row['imbalance'] < -0.05:
            score += 0.15
        elif row['imbalance'] < 0:
            score += 0.05
    
    # Volatility
    if 'sigma' in row.index and not pd.isna(row['sigma']):
        max_score += 0.2
        if 'sigma_60' in row.index and not pd.isna(row['sigma_60']):
            vol_ratio = row['sigma'] / (row['sigma_60'] + 1e-8)
            if vol_ratio > 1.2:
                score += 0.2
            elif vol_ratio > 1.0:
                score += 0.1
    
    if max_score == 0:
        return 0.0
    
    return min(score / max_score, 1.0)


def filter_short_signals_by_regime(
    signals: pd.Series,
    df: pd.DataFrame,
    regime_threshold: float = 0.6
) -> pd.Series:
    """
    Filter short signals to only allow them in bearish regimes.
    
    Args:
        signals: Series with signals (-1, 0, 1)
        df: DataFrame with regime features
        regime_threshold: Minimum regime score to allow shorting (0-1)
        
    Returns:
        Filtered signals Series
    """
    filtered = signals.copy()
    
    # Only filter SHORT signals (-1)
    short_mask = (signals == -1)
    
    if short_mask.sum() == 0:
        return filtered
    
    # Check regime for each short signal
    for idx in signals[short_mask].index:
        if idx in df.index:
            row = df.loc[idx]
            
            # Check if bearish regime
            is_bearish = row.get('is_bearish_regime', False)
            regime_score = row.get('regime_score', 0.0)
            
            # Only allow short if in bearish regime
            if not is_bearish or regime_score < regime_threshold:
                filtered.loc[idx] = 0  # Convert to FLAT
    
    return filtered

