#!/usr/bin/env python3
"""
Microstructure Features - Order Flow & Spread Dynamics

Generates 7 microstructure features from quote-level data:
1. micro_velocity:        Quote update rate (ticks/minute)
2. micro_entropy:         Shannon entropy of tick directions
3. effort_ratio:          Price range per velocity unit
4. spread_zscore:         Normalized bid-ask spread
5. synthetic_order_flow:  Net tick pressure (FIXED - no volume multiplication)
6. flow_cvd_60:           Cumulative Volume Delta over 60 minutes
7. flow_divergence:       Price vs flow divergence

CRITICAL FIX (2026-01-13):
- OLD: synthetic_order_flow = net_tick_pressure * volume (semantic mismatch)
- NEW: synthetic_order_flow = net_tick_pressure (already represents flow)

Rationale:
- net_tick_pressure = up_ticks - down_ticks (e.g., 150 - 100 = +50)
- This ALREADY represents net buying/selling pressure
- Bar volume is independent (total traded volume, not tick-level)
- Multiplying them creates an artificial scaling that doesn't match training expectations
"""

import pandas as pd
import numpy as np
import scipy.stats
from typing import Optional


def calculate_entropy(series: pd.Series) -> float:
    """
    Calculate Shannon entropy of price direction sequence.

    Args:
        series: Series of tick directions (+1, 0, -1)

    Returns:
        Entropy value (higher = more random, lower = more directional)

    Example:
        All up ticks: entropy ~ 0.0 (deterministic)
        Random ticks: entropy ~ 1.0 (maximum uncertainty)
    """
    try:
        counts = series.value_counts()
        if len(counts) == 0:
            return 1.0
        return scipy.stats.entropy(counts)
    except Exception:
        return 1.0  # Default to maximum entropy on error


def compute_microstructure_features(
    bars: pd.DataFrame,
    quotes: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Compute microstructure features from quotes and bars.

    Features (7):
    - micro_velocity: Quote update rate (ticks/minute)
    - micro_entropy: Shannon entropy of tick directions
    - effort_ratio: Price range per velocity unit (market efficiency)
    - spread_zscore: Normalized bid-ask spread (liquidity regime)
    - synthetic_order_flow: Net tick pressure (FIXED - no volume mult)
    - flow_cvd_60: Cumulative Volume Delta over 60 minutes
    - flow_divergence: Normalized divergence between price and flow

    Args:
        bars: DataFrame with OHLCV data (timestamp index)
        quotes: Optional DataFrame with bid_price, ask_price (timestamp index)

    Returns:
        DataFrame with 7 microstructure features (same index as bars)

    Notes:
        - If quotes is None, returns DataFrame with NaN features
        - All features use NO lookahead bias (only past/current data)
        - synthetic_order_flow uses net_tick_pressure directly (FIXED)
    """
    # If no quotes, return NaN features
    if quotes is None or quotes.empty:
        return pd.DataFrame(
            index=bars.index,
            columns=[
                'micro_velocity', 'micro_entropy', 'effort_ratio',
                'spread_zscore', 'synthetic_order_flow',
                'flow_cvd_60', 'flow_divergence'
            ],
            data=np.nan
        )

    print("  Computing microstructure features...")
    print(f"    Quotes: {len(quotes):,} | Bars: {len(bars):,}")

    # 1. Ensure datetime index
    if not isinstance(quotes.index, pd.DatetimeIndex):
        if "timestamp" in quotes.columns:
            quotes = quotes.set_index("timestamp")
        else:
            raise ValueError("Quotes must have DatetimeIndex or 'timestamp' column")

    quotes = quotes.copy()

    # 2. Ensure numeric types for price columns
    print("    Ensuring numeric types...")
    quotes['ask_price'] = pd.to_numeric(quotes['ask_price'], errors='coerce')
    quotes['bid_price'] = pd.to_numeric(quotes['bid_price'], errors='coerce')

    # 3. Compute mid price and spread (vectorized)
    print("    Computing mid/spread...")
    quotes['mid'] = (quotes['ask_price'] + quotes['bid_price']) / 2
    quotes['spread'] = quotes['ask_price'] - quotes['bid_price']

    # 4. Compute tick direction (sign of mid price change)
    # Note: First tick has no previous, set to 0 (neutral)
    quotes['tick_dir'] = np.sign(quotes['mid'].diff()).fillna(0)

    # 5. Resample to 1-minute bars (align with bars frequency)
    print("    Resampling quotes to 1-minute...")

    # FEATURE A: QUOTE VELOCITY (ticks per minute)
    feat_velocity = quotes['mid'].resample('1min').count().to_frame(name='micro_velocity')

    # FEATURE B: SPREAD STATISTICS
    feat_spread = quotes['spread'].resample('1min').agg(['mean', 'max', 'std'])
    feat_spread.columns = ['spread_avg', 'spread_max', 'spread_vol']

    # FEATURE C: MICRO-ENTROPY (Shannon entropy of tick directions)
    print("    Computing entropy...")
    feat_entropy = quotes['tick_dir'].resample('1min').apply(calculate_entropy)
    feat_entropy = feat_entropy.to_frame(name='micro_entropy')

    # FEATURE D: NET TICK PRESSURE (up_ticks - down_ticks)
    # This is the raw synthetic order flow (no volume multiplication)
    print("    Computing order flow...")
    feat_flow = quotes['tick_dir'].resample('1min').sum().to_frame(name='net_tick_pressure')

    # 5. Merge quote features
    micro_df = pd.concat([feat_velocity, feat_spread, feat_entropy, feat_flow], axis=1)

    # 6. Join with bar data
    combined = bars[['high', 'low', 'close', 'volume']].join(micro_df, how='left')

    # 7. Compute derived features

    # FEATURE E: EFFORT RATIO (price range / velocity)
    # Measures how much price moved per tick update
    # High effort_ratio = large moves with few ticks (efficient market)
    range_val = combined['high'] - combined['low']
    combined['effort_ratio'] = range_val / (combined['micro_velocity'] + 1)

    print("    Computing rolling features...")

    # FEATURE F: SPREAD Z-SCORE (normalized spread)
    # Rolling 60-minute normalization (measures liquidity regime)
    spread_mean = combined['spread_avg'].rolling(60, min_periods=10).mean()
    spread_std = combined['spread_avg'].rolling(60, min_periods=10).std()
    combined['spread_zscore'] = (combined['spread_avg'] - spread_mean) / (spread_std + 1e-8)

    # FEATURE G: SYNTHETIC ORDER FLOW (FIXED!)
    # OLD (WRONG): net_tick_pressure * volume
    # NEW (CORRECT): net_tick_pressure directly
    #
    # Rationale:
    # - net_tick_pressure already represents net buying/selling (up - down ticks)
    # - Example: +50 means 50 more up ticks than down ticks
    # - This IS the order flow signal
    # - Bar volume is independent (total trades, not tick direction)
    # - Multiplying by volume creates semantic mismatch
    print("    Computing synthetic order flow (FIXED)...")
    combined['synthetic_order_flow'] = combined['net_tick_pressure']

    print("    Computing cumulative flow and divergence...")

    # FEATURE H: CUMULATIVE VOLUME DELTA (60-minute rolling sum)
    # Smooths out noise to see flow trend
    # Positive CVD = sustained buying pressure
    combined['flow_cvd_60'] = combined['synthetic_order_flow'].rolling(60, min_periods=10).sum()

    # FEATURE I: FLOW DIVERGENCE (price vs flow normalized difference)
    # Detects when price and flow disagree (potential reversal signal)
    # High price + low flow = bearish divergence
    # Low price + high flow = bullish divergence

    # Normalize 60-bar price return
    ret_60 = combined['close'].pct_change(60)
    ret_mean = ret_60.rolling(100, min_periods=20).mean()
    ret_std = ret_60.rolling(100, min_periods=20).std()
    ret_norm = (ret_60 - ret_mean) / (ret_std + 1e-4)

    # Normalize 60-bar flow CVD
    flow_mean = combined['flow_cvd_60'].rolling(100, min_periods=20).mean()
    flow_std = combined['flow_cvd_60'].rolling(100, min_periods=20).std()
    flow_norm = (combined['flow_cvd_60'] - flow_mean) / (flow_std + 1e-4)

    # Divergence = flow_norm - ret_norm
    # Positive divergence: Flow stronger than price (bullish)
    # Negative divergence: Price stronger than flow (bearish)
    combined['flow_divergence'] = flow_norm - ret_norm

    # 8. Fill NaN values with sensible defaults
    combined['micro_velocity'] = combined['micro_velocity'].fillna(0)
    combined['micro_entropy'] = combined['micro_entropy'].fillna(1.0)  # Max entropy (random)
    combined['spread_zscore'] = combined['spread_zscore'].fillna(0)
    combined['synthetic_order_flow'] = combined['synthetic_order_flow'].fillna(0)
    combined['flow_cvd_60'] = combined['flow_cvd_60'].fillna(0)
    combined['flow_divergence'] = combined['flow_divergence'].fillna(0)
    combined['effort_ratio'] = combined['effort_ratio'].fillna(0)

    # 9. Return only microstructure features
    feature_cols = [
        'micro_velocity',
        'micro_entropy',
        'effort_ratio',
        'spread_zscore',
        'synthetic_order_flow',
        'flow_cvd_60',
        'flow_divergence'
    ]

    print(f"    ✓ Microstructure features complete: {len(feature_cols)} features")
    return combined[feature_cols]


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def generate_micro_features(quotes: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy function name for backward compatibility.

    DEPRECATED: Use compute_microstructure_features() instead.

    Args:
        quotes: Quote data (bid/ask prices)
        bars: OHLCV bar data

    Returns:
        DataFrame with 7 microstructure features
    """
    import warnings
    warnings.warn(
        "generate_micro_features() is deprecated. Use compute_microstructure_features() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return compute_microstructure_features(bars, quotes)


if __name__ == "__main__":
    print("=" * 80)
    print("MICROSTRUCTURE FEATURES MODULE")
    print("=" * 80)
    print()
    print("Features generated (7):")
    print("  1. micro_velocity:       Quote update rate (ticks/minute)")
    print("  2. micro_entropy:        Shannon entropy of tick directions")
    print("  3. effort_ratio:         Price range per velocity unit")
    print("  4. spread_zscore:        Normalized bid-ask spread")
    print("  5. synthetic_order_flow: Net tick pressure (FIXED)")
    print("  6. flow_cvd_60:          Cumulative Volume Delta (60-min)")
    print("  7. flow_divergence:      Price vs flow divergence")
    print()
    print("CRITICAL FIX (2026-01-13):")
    print("  OLD: synthetic_order_flow = net_tick_pressure * volume")
    print("  NEW: synthetic_order_flow = net_tick_pressure")
    print()
    print("Rationale: net_tick_pressure already represents order flow.")
    print("Multiplying by volume creates semantic mismatch.")
    print("=" * 80)
