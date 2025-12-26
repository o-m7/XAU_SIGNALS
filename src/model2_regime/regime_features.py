"""
Volume Profile and Market Structure Features for Model #2

This module implements:
- Volume at Price (VAP) distribution
- Point of Control (POC)
- Value Area High/Low
- Volume profile skewness
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def compute_volume_profile(
    df: pd.DataFrame,
    price_column: str = 'close',
    volume_column: str = 'volume',
    lookback: int = 60,
    num_bins: int = 20
) -> pd.DataFrame:
    """
    Compute volume profile features.
    
    Args:
        df: DataFrame with price and volume
        price_column: Column name for price
        volume_column: Column name for volume
        lookback: Window for volume profile (e.g., 60 bars = 5 hours on 5-min bars)
        num_bins: Number of price bins
        
    Returns:
        DataFrame with volume profile features
    """
    df = df.copy()
    
    def calc_profile(window):
        """Calculate volume profile for a window."""
        if len(window) < 2:
            return {
                'poc': window[price_column].iloc[-1] if len(window) > 0 else np.nan,
                'vah': window[price_column].iloc[-1] if len(window) > 0 else np.nan,
                'val': window[price_column].iloc[-1] if len(window) > 0 else np.nan,
                'poc_distance': 0.0,
                'value_area_width': 0.0,
                'profile_skew': 0.0,
                'high_volume_node_count': 0
            }
        
        prices = window[price_column].values
        volumes = window[volume_column].values
        
        # Create price bins
        price_range = prices.max() - prices.min()
        if price_range == 0:
            return {
                'poc': prices[-1],
                'vah': prices[-1],
                'val': prices[-1],
                'poc_distance': 0.0,
                'value_area_width': 0.0,
                'profile_skew': 0.0,
                'high_volume_node_count': 0
            }
        
        bins = np.linspace(prices.min(), prices.max(), num_bins + 1)
        
        # Aggregate volume by price bin
        volume_by_price = np.zeros(num_bins)
        for i in range(len(prices)):
            bin_idx = min(int((prices[i] - bins[0]) / (price_range / num_bins)), num_bins - 1)
            volume_by_price[bin_idx] += volumes[i]
        
        # Point of Control (POC) - price level with most volume
        poc_idx = np.argmax(volume_by_price)
        poc = (bins[poc_idx] + bins[poc_idx + 1]) / 2
        
        # Value Area (70% of volume)
        total_volume = volume_by_price.sum()
        value_area_volume = total_volume * 0.70
        
        # Find value area starting from POC
        cumsum = volume_by_price[poc_idx]
        low_idx = poc_idx
        high_idx = poc_idx
        
        while cumsum < value_area_volume and (low_idx > 0 or high_idx < num_bins - 1):
            # Expand to whichever side has more volume
            add_low = volume_by_price[low_idx - 1] if low_idx > 0 else 0
            add_high = volume_by_price[high_idx + 1] if high_idx < num_bins - 1 else 0
            
            if add_low > add_high and low_idx > 0:
                low_idx -= 1
                cumsum += add_low
            elif high_idx < num_bins - 1:
                high_idx += 1
                cumsum += add_high
            else:
                break
        
        vah = bins[high_idx + 1]  # Value Area High
        val = bins[low_idx]        # Value Area Low
        
        # Current price distance from POC
        current_price = prices[-1]
        poc_distance = (current_price - poc) / poc if poc != 0 else 0
        
        # Value area width (normalized by price)
        value_area_width = (vah - val) / ((vah + val) / 2 + 1e-8)
        
        # Profile skewness (is volume clustered at top or bottom?)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        profile_skew = np.sum(volume_by_price * (bin_centers - poc)) / (total_volume + 1e-8)
        
        # High volume nodes (how many price levels have significant volume?)
        volume_threshold = total_volume / num_bins  # Average volume per bin
        high_volume_node_count = int(np.sum(volume_by_price > volume_threshold * 1.5))
        
        return {
            'poc': poc,
            'vah': vah,
            'val': val,
            'poc_distance': poc_distance,
            'value_area_width': value_area_width,
            'profile_skew': profile_skew,
            'high_volume_node_count': high_volume_node_count
        }
    
    # Rolling calculation
    results = []
    for i in range(len(df)):
        start_idx = max(0, i - lookback + 1)
        window = df.iloc[start_idx:i+1]
        result = calc_profile(window)
        result['index'] = df.index[i]
        results.append(result)
    
    profile_df = pd.DataFrame(results).set_index('index')
    
    # Merge back
    df = pd.concat([df, profile_df], axis=1)
    
    return df


def compute_vwap_features(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Compute VWAP-based features.
    
    Args:
        df: DataFrame with price and volume
        lookback: Window for VWAP calculation
        
    Returns:
        DataFrame with VWAP features
    """
    df = df.copy()
    
    # Rolling VWAP
    df['vwap_rolling'] = (
        (df['close'] * df['volume']).rolling(lookback, min_periods=1).sum() /
        df['volume'].rolling(lookback, min_periods=1).sum()
    )
    
    # Distance from VWAP
    df['vwap_distance'] = (df['close'] - df['vwap_rolling']) / (df['vwap_rolling'] + 1e-8)
    
    # VWAP distance in standard deviations
    df['vwap_distance_std'] = df['vwap_distance'].rolling(lookback, min_periods=2).std()
    df['vwap_zscore'] = df['vwap_distance'] / (df['vwap_distance_std'] + 1e-8)
    
    # VWAP trend (is VWAP rising or falling?)
    df['vwap_slope'] = df['vwap_rolling'].diff() / (df['vwap_rolling'] + 1e-8)
    
    # Price crossovers with VWAP
    df['above_vwap'] = (df['close'] > df['vwap_rolling']).astype(int)
    df['vwap_cross'] = df['above_vwap'].diff()
    
    return df


def compute_volatility_regime_features(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Compute volatility regime indicators.
    
    Args:
        df: DataFrame with price data
        lookback: Window for volatility calculation
        
    Returns:
        DataFrame with volatility regime features
    """
    df = df.copy()
    
    # Returns
    df['returns'] = df['close'].pct_change()
    
    # Realized volatility (standard deviation of returns)
    df['realized_vol'] = df['returns'].rolling(lookback, min_periods=2).std()
    
    # Volatility percentile (is vol high or low historically?)
    df['vol_percentile'] = df['realized_vol'].rolling(lookback * 2, min_periods=lookback).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8) if len(x) > 1 else 0.5,
        raw=False
    )
    
    # Volatility regime (quantize into buckets)
    df['vol_regime'] = pd.cut(
        df['vol_percentile'],
        bins=[0, 0.25, 0.5, 0.75, 1.0],
        labels=['very_low', 'low', 'high', 'very_high']
    )
    
    # Volatility expansion/contraction rate
    df['vol_change'] = df['realized_vol'].pct_change()
    df['vol_expanding'] = (df['vol_change'] > 0).astype(int)
    
    # Parkinson volatility (uses high-low range, more efficient)
    df['parkinson_vol'] = np.sqrt(
        1.0 / (4 * np.log(2)) * 
        (np.log(df['high'] / df['low']) ** 2).rolling(lookback, min_periods=1).mean()
    )
    
    # Garman-Klass volatility (even more efficient)
    df['gk_vol'] = np.sqrt(
        0.5 * (np.log(df['high'] / df['low']) ** 2) -
        (2 * np.log(2) - 1) * (np.log(df['close'] / df['open']) ** 2)
    ).rolling(lookback, min_periods=1).mean()
    
    return df


def compute_range_features(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Compute range and consolidation features.
    
    Args:
        df: DataFrame with OHLC data
        lookback: Window for range calculation
        
    Returns:
        DataFrame with range features
    """
    df = df.copy()
    
    # True Range
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    
    # ATR
    df['atr'] = df['true_range'].rolling(lookback, min_periods=1).mean()
    
    # Range as % of price
    df['range_pct'] = (df['high'] - df['low']) / (df['close'] + 1e-8)
    
    # Range compression (is range getting tighter?)
    df['range_compression'] = (
        df['range_pct'] / 
        df['range_pct'].rolling(lookback, min_periods=1).mean()
    )
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(lookback, min_periods=1).mean()
    df['bb_std'] = df['close'].rolling(lookback, min_periods=2).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    
    # BB position (where is price within bands?)
    df['bb_position'] = (
        (df['close'] - df['bb_lower']) / 
        (df['bb_upper'] - df['bb_lower'] + 1e-8)
    )
    
    # BB width (volatility indicator)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-8)
    
    # Squeeze (low BB width = consolidation)
    df['bb_squeeze'] = df['bb_width'] / df['bb_width'].rolling(lookback, min_periods=1).mean()
    
    return df


def build_regime_features(df: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """
    Build complete set of regime detection features.
    
    Args:
        df: DataFrame with market data
        lookback: Lookback window for features
        
    Returns:
        DataFrame with all regime features
    """
    logger.info("Building regime detection features...")
    
    df = compute_vwap_features(df, lookback)
    df = compute_volatility_regime_features(df, lookback)
    df = compute_range_features(df, lookback)
    df = compute_volume_profile(df, lookback=lookback)
    
    logger.info(f"Built regime features, total columns: {len(df.columns)}")
    
    return df

