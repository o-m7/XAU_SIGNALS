"""
Market Regime Labeling for Model #2

This module implements the 5-regime classification system:
1. TRENDING_MOMENTUM
2. MEAN_REVERTING
3. BREAKOUT_CONSOLIDATION
4. HIGH_VOLATILITY_CHAOTIC
5. LOW_LIQUIDITY_THIN
"""

import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# Regime constants
REGIME_TRENDING = 0
REGIME_MEAN_REVERTING = 1
REGIME_BREAKOUT = 2
REGIME_HIGH_VOL = 3
REGIME_LOW_LIQUIDITY = 4

REGIME_NAMES = {
    REGIME_TRENDING: "TRENDING_MOMENTUM",
    REGIME_MEAN_REVERTING: "MEAN_REVERTING",
    REGIME_BREAKOUT: "BREAKOUT_CONSOLIDATION",
    REGIME_HIGH_VOL: "HIGH_VOLATILITY_CHAOTIC",
    REGIME_LOW_LIQUIDITY: "LOW_LIQUIDITY_THIN",
}


def classify_regime_rule_based(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify market regime using rule-based heuristics.
    
    This creates initial labels for training the regime classifier.
    
    Logic:
    - LOW_LIQUIDITY: Wide spreads, low volume, few trades
    - HIGH_VOLATILITY: High vol percentile, expanding vol, wide ranges
    - TRENDING_MOMENTUM: High OFI persistence, strong direction, volume clusters
    - MEAN_REVERTING: Low vol, price far from VWAP, depth balance
    - BREAKOUT_CONSOLIDATION: Range compression, low vol, tight BB
    
    Args:
        df: DataFrame with all features
        
    Returns:
        DataFrame with 'regime' column added
    """
    df = df.copy()
    
    # Initialize regime as unknown
    df['regime'] = -1
    
    # Define thresholds (these can be tuned)
    SPREAD_PCT_THRESHOLD = df['spread_pct'].quantile(0.90) if 'spread_pct' in df.columns else 0.001
    VOL_PERCENTILE_HIGH = 0.75
    VOL_PERCENTILE_LOW = 0.25
    OFI_PERSISTENCE_HIGH = 0.3
    VWAP_DISTANCE_THRESHOLD = 0.005  # 0.5% from VWAP
    BB_SQUEEZE_THRESHOLD = 0.7  # Below 0.7 = consolidation
    
    # 1. LOW_LIQUIDITY: Wide spreads, low volume, slow updates
    if 'spread_pct' in df.columns and 'volume' in df.columns:
        low_liquidity_mask = (
            (df['spread_pct'] > SPREAD_PCT_THRESHOLD) |
            (df['volume'] < df['volume'].rolling(60, min_periods=1).quantile(0.1))
        )
        df.loc[low_liquidity_mask, 'regime'] = REGIME_LOW_LIQUIDITY
    
    # 2. HIGH_VOLATILITY: High vol, expanding, wide ranges
    if 'vol_percentile' in df.columns and 'vol_expanding' in df.columns:
        high_vol_mask = (
            (df['vol_percentile'] > VOL_PERCENTILE_HIGH) &
            (df['vol_expanding'] == 1) &
            (df['range_pct'] > df['range_pct'].rolling(60, min_periods=1).quantile(0.75))
        )
        # Only override if not already labeled as low liquidity
        df.loc[(df['regime'] == -1) & high_vol_mask, 'regime'] = REGIME_HIGH_VOL
    
    # 3. TRENDING_MOMENTUM: High OFI persistence, strong vol imbalance, directional
    if 'ofi_persistence' in df.columns and 'cum_volume_imbalance' in df.columns:
        trending_mask = (
            (abs(df['ofi_persistence']) > OFI_PERSISTENCE_HIGH) &
            (abs(df['cum_volume_imbalance']) > 0.15) &
            (df['vol_percentile'] > 0.4) &  # Moderate to high vol
            (df['vol_percentile'] < VOL_PERCENTILE_HIGH)  # But not chaotic
        )
        df.loc[(df['regime'] == -1) & trending_mask, 'regime'] = REGIME_TRENDING
    
    # 4. BREAKOUT_CONSOLIDATION: Low vol, tight BB, range compression
    if 'bb_squeeze' in df.columns and 'range_compression' in df.columns:
        breakout_mask = (
            (df['vol_percentile'] < VOL_PERCENTILE_LOW) &
            (df['bb_squeeze'] < BB_SQUEEZE_THRESHOLD) &
            (df['range_compression'] < 0.8)
        )
        df.loc[(df['regime'] == -1) & breakout_mask, 'regime'] = REGIME_BREAKOUT
    
    # 5. MEAN_REVERTING: Everything else with moderate conditions
    # Low vol, price away from fair value, balanced depth
    if 'vwap_distance' in df.columns:
        mean_reverting_mask = (
            (df['vol_percentile'] < 0.6) &
            (abs(df['vwap_distance']) > VWAP_DISTANCE_THRESHOLD) &
            (df['vol_expanding'] == 0)
        )
        df.loc[(df['regime'] == -1) & mean_reverting_mask, 'regime'] = REGIME_MEAN_REVERTING
    
    # Default remaining to MEAN_REVERTING (safest regime)
    df.loc[df['regime'] == -1, 'regime'] = REGIME_MEAN_REVERTING
    
    # Smooth regime labels (reduce noise)
    df['regime'] = df['regime'].rolling(5, min_periods=1, center=True).apply(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0],
        raw=False
    ).astype(int)
    
    return df


def compute_regime_confidence_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute confidence scores for each regime.
    
    Instead of hard classification, compute probability/score for each regime.
    This will be used as soft labels for training.
    
    Args:
        df: DataFrame with features and regime labels
        
    Returns:
        DataFrame with confidence scores added
    """
    df = df.copy()
    
    # Initialize confidence scores
    for regime_id, regime_name in REGIME_NAMES.items():
        df[f'regime_conf_{regime_name}'] = 0.0
    
    # Compute scores based on feature values
    
    # TRENDING confidence
    if 'ofi_persistence' in df.columns:
        df[f'regime_conf_{REGIME_NAMES[REGIME_TRENDING]}'] = (
            0.5 * abs(df['ofi_persistence']) / (abs(df['ofi_persistence']).max() + 1e-8) +
            0.3 * abs(df['cum_volume_imbalance']) / (abs(df['cum_volume_imbalance']).max() + 1e-8) +
            0.2 * (df['vol_percentile'] - 0.25).clip(0, 0.5) * 2  # Prefer moderate vol
        )
    
    # MEAN_REVERTING confidence
    if 'vwap_distance' in df.columns:
        df[f'regime_conf_{REGIME_NAMES[REGIME_MEAN_REVERTING]}'] = (
            0.4 * abs(df['vwap_zscore']) / (abs(df['vwap_zscore']).max() + 1e-8) +
            0.3 * (1 - df['vol_percentile']) +  # Low vol preferred
            0.3 * (1 - abs(df['ofi_persistence'])) / (1 + abs(df['ofi_persistence']))  # Low OFI persistence
        )
    
    # BREAKOUT confidence
    if 'bb_squeeze' in df.columns:
        df[f'regime_conf_{REGIME_NAMES[REGIME_BREAKOUT]}'] = (
            0.5 * (1 - df['bb_squeeze']) +  # Tight BB
            0.3 * df['range_compression'] +  # Compressed range
            0.2 * (1 - df['vol_percentile'])  # Low vol
        )
    
    # HIGH_VOL confidence
    if 'vol_percentile' in df.columns:
        df[f'regime_conf_{REGIME_NAMES[REGIME_HIGH_VOL]}'] = (
            0.6 * df['vol_percentile'] +
            0.2 * df['vol_expanding'] +
            0.2 * df['spread_expansion'].clip(0, 3) / 3
        )
    
    # LOW_LIQUIDITY confidence
    if 'spread_pct' in df.columns:
        spread_percentile = df['spread_pct'].rolling(60, min_periods=1).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8),
            raw=False
        )
        df[f'regime_conf_{REGIME_NAMES[REGIME_LOW_LIQUIDITY]}'] = (
            0.7 * spread_percentile +
            0.3 * (1 - df['volume'] / df['volume'].rolling(60, min_periods=1).max())
        )
    
    # Normalize confidences to sum to 1 (convert to probabilities)
    conf_cols = [f'regime_conf_{name}' for name in REGIME_NAMES.values()]
    conf_sum = df[conf_cols].sum(axis=1) + 1e-8
    for col in conf_cols:
        df[col] = df[col] / conf_sum
    
    return df


def add_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add regime labels and confidence scores to DataFrame.
    
    Args:
        df: DataFrame with features
        
    Returns:
        DataFrame with regime labels and confidences
    """
    logger.info("Creating regime labels...")
    
    # Rule-based initial classification
    df = classify_regime_rule_based(df)
    
    # Compute confidence scores
    df = compute_regime_confidence_scores(df)
    
    # Log regime distribution
    regime_dist = df['regime'].value_counts()
    logger.info("Regime distribution:")
    for regime_id, count in regime_dist.items():
        regime_name = REGIME_NAMES.get(regime_id, "UNKNOWN")
        pct = count / len(df) * 100
        logger.info(f"  {regime_name}: {count:,} ({pct:.1f}%)")
    
    return df


def get_regime_name(regime_id: int) -> str:
    """Get human-readable name for regime ID."""
    return REGIME_NAMES.get(regime_id, f"UNKNOWN_{regime_id}")


def get_feature_columns_for_regime_classifier() -> list:
    """
    Get list of feature columns to use for regime classifier.
    
    Returns:
        List of feature column names
    """
    return [
        # Order flow
        'ofi', 'ofi_persistence', 'ofi_normalized',
        'signed_volume_sum', 'volume_imbalance', 'cum_volume_imbalance',
        
        # Toxicity
        'toxicity', 'toxicity_zscore',
        
        # Depth
        'depth_imbalance', 'depth_shock',
        
        # Spread
        'spread_pct', 'spread_percentile', 'spread_expansion',
        
        # Trade intensity
        'trade_burst', 'quote_trade_ratio',
        
        # Quote stability
        'quote_stability', 'stability_trend',
        
        # VWAP
        'vwap_distance', 'vwap_zscore', 'vwap_slope',
        
        # Volatility
        'vol_percentile', 'vol_expanding', 'parkinson_vol',
        
        # Range
        'range_compression', 'bb_position', 'bb_width', 'bb_squeeze',
        
        # Volume profile
        'poc_distance', 'value_area_width', 'profile_skew', 'high_volume_node_count',
    ]

