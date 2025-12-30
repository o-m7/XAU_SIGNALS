"""
Order Flow and Market Microstructure Features for Model #2

This module implements advanced order flow indicators:
- Order Flow Imbalance (OFI)
- Order flow toxicity
- Liquidity provision cost
- Tick pressure
- Quote stability
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def compute_order_flow_imbalance(
    df: pd.DataFrame,
    lookback: int = 20
) -> pd.DataFrame:
    """
    Compute Order Flow Imbalance (OFI) features.
    
    OFI measures the net change in liquidity at the bid and ask.
    High positive OFI → buying pressure
    High negative OFI → selling pressure
    
    Args:
        df: DataFrame with bid_size, ask_size columns
        lookback: Window for rolling metrics
        
    Returns:
        DataFrame with OFI features added
    """
    df = df.copy()
    
    if 'bid_size' in df.columns and 'ask_size' in df.columns:
        # Instantaneous OFI
        df['ofi'] = (df['bid_size'].diff() - df['ask_size'].diff())
        
        # Cumulative OFI over window
        df['ofi_cumsum'] = df['ofi'].rolling(lookback, min_periods=1).sum()
        
        # OFI persistence (autocorrelation)
        df['ofi_persistence'] = df['ofi'].rolling(lookback, min_periods=2).apply(
            lambda x: x.autocorr() if len(x) > 1 else 0, raw=False
        )
        
        # OFI normalized by total depth
        total_depth = df['bid_size'] + df['ask_size']
        df['ofi_normalized'] = df['ofi'] / (total_depth + 1e-8)
        
    else:
        logger.warning("bid_size/ask_size not available, using fallback OFI")
        # Fallback: use volume and price direction
        df['ofi'] = 0.0
        df['ofi_cumsum'] = 0.0
        df['ofi_persistence'] = 0.0
        df['ofi_normalized'] = 0.0
    
    return df


def compute_signed_volume_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Compute signed volume features.
    
    Uses tick rule to infer trade direction when not explicitly available.
    
    Args:
        df: DataFrame with trade data
        lookback: Window for rolling metrics
        
    Returns:
        DataFrame with signed volume features
    """
    df = df.copy()
    
    if 'buy_volume' in df.columns and 'sell_volume' in df.columns:
        # Direct signed volume
        df['signed_volume'] = df['buy_volume'] - df['sell_volume']
    else:
        # Tick rule: if price increased, assume buy; if decreased, assume sell
        price_change = df['close'].diff()
        df['signed_volume'] = np.where(price_change > 0, df['volume'], 
                                      np.where(price_change < 0, -df['volume'], 0))
    
    # Rolling signed volume
    df['signed_volume_sum'] = df['signed_volume'].rolling(lookback, min_periods=1).sum()
    
    # Volume imbalance ratio
    if 'buy_volume' in df.columns:
        total_vol = df['buy_volume'] + df['sell_volume'] + 1e-8
        df['volume_imbalance'] = (df['buy_volume'] - df['sell_volume']) / total_vol
    else:
        df['volume_imbalance'] = df['signed_volume'] / (df['volume'] + 1e-8)
    
    # Cumulative volume imbalance
    df['cum_volume_imbalance'] = df['volume_imbalance'].rolling(lookback, min_periods=1).mean()
    
    return df


def compute_liquidity_toxicity(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Compute order flow toxicity indicator.
    
    Toxicity measures how "informed" recent orders are.
    High toxicity = smart money active → market makers widen spreads
    
    Formula: toxicity = signed_volume / sqrt(total_volume) * price_impact
    
    Args:
        df: DataFrame with volume and price data
        lookback: Window for computation
        
    Returns:
        DataFrame with toxicity features
    """
    df = df.copy()
    
    # Ensure signed volume exists
    if 'signed_volume' not in df.columns:
        df = compute_signed_volume_features(df, lookback)
    
    # Price impact (how much price moved per unit volume)
    price_change = df['close'].diff()
    df['price_impact'] = price_change / (df['volume'] + 1e-8)
    
    # Toxicity = signed_vol / sqrt(total_vol) * price_impact
    df['toxicity'] = (
        df['signed_volume'] / (np.sqrt(df['volume']) + 1e-8) * 
        abs(df['price_impact'])
    )
    
    # Rolling toxicity
    df['toxicity_mean'] = df['toxicity'].rolling(lookback, min_periods=1).mean()
    df['toxicity_std'] = df['toxicity'].rolling(lookback, min_periods=2).std()
    
    # Toxicity z-score
    df['toxicity_zscore'] = (
        (df['toxicity'] - df['toxicity_mean']) / (df['toxicity_std'] + 1e-8)
    )
    
    return df


def compute_depth_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Compute market depth and liquidity features.
    
    Args:
        df: DataFrame with bid/ask size data
        lookback: Window for rolling metrics
        
    Returns:
        DataFrame with depth features
    """
    df = df.copy()
    
    if 'avg_bid_size' in df.columns and 'avg_ask_size' in df.columns:
        # Total depth at best bid/ask
        df['total_depth'] = df['avg_bid_size'] + df['avg_ask_size']
        
        # Depth imbalance
        df['depth_imbalance'] = (
            (df['avg_bid_size'] - df['avg_ask_size']) / 
            (df['total_depth'] + 1e-8)
        )
        
        # Depth change rate
        df['depth_change'] = df['total_depth'].pct_change()
        
        # Rolling depth metrics
        df['depth_mean'] = df['total_depth'].rolling(lookback, min_periods=1).mean()
        df['depth_std'] = df['total_depth'].rolling(lookback, min_periods=2).std()
        
        # Depth shock indicator (sudden depth drop)
        df['depth_shock'] = (df['total_depth'] - df['depth_mean']) / (df['depth_std'] + 1e-8)
        
    else:
        logger.warning("Depth data not available, using placeholder features")
        df['total_depth'] = np.nan
        df['depth_imbalance'] = 0.0
        df['depth_change'] = 0.0
        df['depth_mean'] = np.nan
        df['depth_std'] = 0.0
        df['depth_shock'] = 0.0
    
    return df


def compute_spread_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Compute bid-ask spread features.
    
    Args:
        df: DataFrame with spread data
        lookback: Window for rolling metrics
        
    Returns:
        DataFrame with spread features
    """
    df = df.copy()
    
    if 'avg_spread' in df.columns and 'avg_mid' in df.columns:
        # Relative spread (spread as % of mid)
        df['spread_pct'] = df['avg_spread'] / (df['avg_mid'] + 1e-8)
        
        # Rolling spread metrics
        df['spread_mean'] = df['avg_spread'].rolling(lookback, min_periods=1).mean()
        df['spread_std'] = df['avg_spread'].rolling(lookback, min_periods=2).std()
        
        # Spread percentile (is spread wide or tight?)
        df['spread_percentile'] = df['avg_spread'].rolling(lookback, min_periods=1).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-8) if len(x) > 1 else 0.5,
            raw=False
        )
        
        # Spread expansion (sudden widening)
        df['spread_expansion'] = (
            (df['avg_spread'] - df['spread_mean']) / (df['spread_std'] + 1e-8)
        )
        
    else:
        logger.warning("Spread data not available")
        df['spread_pct'] = 0.0
        df['spread_mean'] = 0.0
        df['spread_std'] = 0.0
        df['spread_percentile'] = 0.5
        df['spread_expansion'] = 0.0
    
    return df


def compute_trade_intensity_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Compute trade arrival and quote update intensity.
    
    Args:
        df: DataFrame with trade count and quote update data
        lookback: Window for rolling metrics
        
    Returns:
        DataFrame with intensity features
    """
    df = df.copy()
    
    # Trade arrival rate
    if 'trades' in df.columns:
        df['trade_rate'] = df['trades']
        df['trade_rate_mean'] = df['trade_rate'].rolling(lookback, min_periods=1).mean()
        df['trade_rate_std'] = df['trade_rate'].rolling(lookback, min_periods=2).std()
        
        # Trade intensity burst
        df['trade_burst'] = (
            (df['trade_rate'] - df['trade_rate_mean']) / (df['trade_rate_std'] + 1e-8)
        )
    else:
        df['trade_rate'] = 0
        df['trade_rate_mean'] = 0
        df['trade_burst'] = 0
    
    # Quote update rate
    if 'quote_updates' in df.columns:
        df['quote_rate'] = df['quote_updates']
        df['quote_rate_mean'] = df['quote_rate'].rolling(lookback, min_periods=1).mean()
        
        # Quote-to-trade ratio (high ratio = lots of quote updates, few trades)
        df['quote_trade_ratio'] = df['quote_rate'] / (df['trade_rate'] + 1e-8)
    else:
        df['quote_rate'] = 0
        df['quote_trade_ratio'] = 1.0
    
    return df


def compute_quote_stability_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Compute quote stability metrics.
    
    Args:
        df: DataFrame with quote data
        lookback: Window for metrics
        
    Returns:
        DataFrame with stability features
    """
    df = df.copy()
    
    if 'quote_stability' in df.columns:
        # Already computed
        df['stability_mean'] = df['quote_stability'].rolling(lookback, min_periods=1).mean()
        
        # Stability trend (increasing or decreasing)
        df['stability_trend'] = df['quote_stability'].diff()
        
    else:
        # Approximate stability from price volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(lookback, min_periods=2).std()
        df['quote_stability'] = 1.0 / (df['volatility'] + 1e-8)
        df['stability_mean'] = df['quote_stability'].rolling(lookback, min_periods=1).mean()
        df['stability_trend'] = df['quote_stability'].diff()
    
    return df


def compute_synthetic_order_flow(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Compute synthetic order flow features (from Model #1).
    
    Synthetic order flow = net_tick_pressure * volume
    This detects genuine buying/selling vs fakeouts.
    
    Args:
        df: DataFrame with OHLCV and quote data
        lookback: Window for rolling metrics
        
    Returns:
        DataFrame with synthetic order flow features added
    """
    df = df.copy()
    
    # Need mid price for tick direction
    if 'avg_mid' in df.columns:
        mid = df['avg_mid']
    elif 'close' in df.columns:
        mid = df['close']
    else:
        logger.warning("No mid/close price available for synthetic order flow")
        df['synthetic_order_flow'] = 0.0
        df['flow_cvd_60'] = 0.0
        df['flow_divergence'] = 0.0
        return df
    
    # Compute tick direction (up = +1, down = -1, unchanged = 0)
    tick_dir = np.sign(mid.diff()).fillna(0)
    
    # Net tick pressure (sum of tick directions per bar)
    # For 5-min bars, we aggregate tick directions
    net_tick_pressure = tick_dir
    
    # Synthetic order flow = net_tick_pressure * volume
    df['synthetic_order_flow'] = net_tick_pressure * df['volume']
    
    # Cumulative Volume Delta (CVD) over 60 bars (5 hours on 5-min bars)
    df['flow_cvd_60'] = df['synthetic_order_flow'].rolling(60, min_periods=10).sum()
    
    # Flow divergence: Compare price return vs flow trend
    ret_60 = mid.pct_change(60)
    ret_mean = ret_60.rolling(100, min_periods=20).mean()
    ret_std = ret_60.rolling(100, min_periods=20).std()
    ret_norm = (ret_60 - ret_mean) / (ret_std + 1e-8)
    
    flow_mean = df['flow_cvd_60'].rolling(100, min_periods=20).mean()
    flow_std = df['flow_cvd_60'].rolling(100, min_periods=20).std()
    flow_norm = (df['flow_cvd_60'] - flow_mean) / (flow_std + 1e-8)
    
    df['flow_divergence'] = flow_norm - ret_norm
    
    # Fill NaNs
    df['synthetic_order_flow'] = df['synthetic_order_flow'].fillna(0)
    df['flow_cvd_60'] = df['flow_cvd_60'].fillna(0)
    df['flow_divergence'] = df['flow_divergence'].fillna(0)
    
    return df


def build_order_flow_features(
    df: pd.DataFrame,
    lookback: int = 20
) -> pd.DataFrame:
    """
    Build complete order flow and microstructure feature set.
    
    Args:
        df: DataFrame with base market data
        lookback: Lookback window for rolling features
        
    Returns:
        DataFrame with all order flow features
    """
    logger.info("Building order flow features...")
    
    df = compute_order_flow_imbalance(df, lookback)
    df = compute_signed_volume_features(df, lookback)
    df = compute_liquidity_toxicity(df, lookback)
    df = compute_depth_features(df, lookback)
    df = compute_spread_features(df, lookback)
    df = compute_trade_intensity_features(df, lookback)
    df = compute_quote_stability_features(df, lookback)
    
    # Add synthetic order flow (NEW - from Model #1)
    df = compute_synthetic_order_flow(df, lookback)
    
    logger.info(f"Built {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])} order flow features")
    
    return df

