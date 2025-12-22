"""
Alpha Signals Module - Academically Validated Alpha Sources.

Based on established market microstructure research:
- Hasbrouck (1991): Order flow and price discovery
- Bouchaud (2004): Order flow impact
- Biais & Hillion: Microprice and market making
- Gatheral: No-dynamic-arbitrage, liquidity
- Andersen & Bollerslev: High-frequency volatility

CRITICAL: These are ALPHA SIGNALS, not direction predictions.
The model predicts WHEN edge exists, not which direction.
Direction comes from microstructure after alpha detection.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class AlphaConfig:
    """Configuration for alpha signal thresholds."""
    # Alpha A: Order Flow Imbalance
    ofi_threshold: float = 0.3
    ofi_persistence_bars: int = 3
    
    # Alpha B: Microprice Dislocation
    micro_dislocation_threshold: float = 0.0002  # 0.02%
    
    # Alpha C: Liquidity Shock
    spread_expansion_threshold: float = 1.5  # 1.5x normal
    depth_collapse_threshold: float = 0.7   # 70% of normal
    
    # Alpha D: Volatility Breakout
    vol_ratio_threshold: float = 1.5  # Short vol 1.5x long vol
    
    # Cost floor for all alphas
    cost_floor: float = 0.0003  # 0.03% minimum move


# =============================================================================
# ALPHA SIGNAL A: PERSISTENT ORDER FLOW (Hasbrouck/Bouchaud)
# =============================================================================

def compute_alpha_a_order_flow(
    df: pd.DataFrame,
    horizon_minutes: int,
    config: AlphaConfig = AlphaConfig()
) -> pd.Series:
    """
    Alpha A: Persistent Order Flow Signal.
    
    Research basis: Hasbrouck (1991), Bouchaud (2004)
    
    When order flow imbalance exceeds a threshold AND persists,
    short-horizon returns have statistically significant drift
    in the direction of the flow.
    
    Label = 1 if:
    - |OFI| > threshold
    - Future return has same sign as OFI
    - |future_return| > cost_floor
    
    Since we don't have trade data, we proxy OFI using:
    - Spread changes (widening = selling pressure)
    - Price momentum vs expected (deviation from VWAP)
    """
    n = len(df)
    labels = np.zeros(n, dtype=int)
    
    # Proxy for order flow using spread dynamics and price behavior
    mid = df["mid"].values
    spread_pct = df["spread_pct"].values
    
    # OFI proxy: negative spread change + positive momentum = buying
    # positive spread change + negative momentum = selling
    spread_change = np.diff(spread_pct, prepend=spread_pct[0])
    price_change = np.diff(mid, prepend=mid[0]) / mid
    
    # Smooth OFI proxy
    ofi_proxy = np.zeros(n)
    for i in range(config.ofi_persistence_bars, n):
        # Buying pressure: spread tightening + price rising
        # Selling pressure: spread widening + price falling
        recent_spread_change = spread_change[i-config.ofi_persistence_bars:i].mean()
        recent_price_change = price_change[i-config.ofi_persistence_bars:i].sum()
        
        # Positive = buying, Negative = selling
        ofi_proxy[i] = recent_price_change - recent_spread_change * 10
    
    # Normalize OFI
    ofi_std = pd.Series(ofi_proxy).rolling(60, min_periods=30).std().values
    ofi_normalized = np.where(ofi_std > 0, ofi_proxy / ofi_std, 0)
    
    # Generate labels
    valid_end = n - horizon_minutes
    
    for i in range(60, valid_end):  # Skip warm-up period
        ofi = ofi_normalized[i]
        
        if abs(ofi) < config.ofi_threshold:
            continue
        
        # Compute future return
        future_mid = mid[i + horizon_minutes] if i + horizon_minutes < n else mid[-1]
        future_return = (future_mid / mid[i]) - 1
        
        # Check if return aligns with OFI direction and exceeds cost
        same_sign = (ofi > 0 and future_return > 0) or (ofi < 0 and future_return < 0)
        exceeds_cost = abs(future_return) > config.cost_floor
        
        if same_sign and exceeds_cost:
            labels[i] = 1
    
    return pd.Series(labels, index=df.index, name=f"alpha_a_{horizon_minutes}m")


# =============================================================================
# ALPHA SIGNAL B: MICROPRICE DISLOCATION (Biais/Hillion)
# =============================================================================

def compute_alpha_b_microprice(
    df: pd.DataFrame,
    horizon_minutes: int,
    config: AlphaConfig = AlphaConfig()
) -> pd.Series:
    """
    Alpha B: Microprice Dislocation/Reversion Signal.
    
    Research basis: Biais & Hillion, market-making literature
    
    When midprice deviates from "fair value" (estimated by
    recent price behavior), price tends to revert.
    
    Since we don't have size data for true microprice, we estimate
    fair value using VWAP and recent price distribution.
    
    Label = 1 if:
    - |dislocation| > threshold
    - Price reverts toward fair value within horizon
    """
    n = len(df)
    labels = np.zeros(n, dtype=int)
    
    mid = df["mid"].values
    
    # Estimate fair value using rolling VWAP or mean
    if "vwap" in df.columns:
        fair_value = df["vwap"].shift(1).values
    else:
        # Use rolling mean as fair value proxy
        fair_value = pd.Series(mid).shift(1).rolling(20, min_periods=10).mean().values
    
    # Compute dislocation: how far mid is from fair value
    dislocation = np.where(
        fair_value > 0,
        (mid - fair_value) / fair_value,
        0
    )
    
    valid_end = n - horizon_minutes
    
    for i in range(30, valid_end):
        disl = dislocation[i]
        
        if abs(disl) < config.micro_dislocation_threshold:
            continue
        
        # Check for reversion within horizon
        future_mids = mid[i+1:i+1+horizon_minutes]
        if len(future_mids) == 0:
            continue
        
        fv = fair_value[i]
        if np.isnan(fv) or fv <= 0:
            continue
        
        # Reversion = price moves back toward fair value
        current_dist = abs(mid[i] - fv)
        future_dists = np.abs(future_mids - fv)
        min_future_dist = np.nanmin(future_dists)
        
        # Did price revert at least halfway?
        reverted = min_future_dist < current_dist * 0.5
        
        if reverted:
            labels[i] = 1
    
    return pd.Series(labels, index=df.index, name=f"alpha_b_{horizon_minutes}m")


# =============================================================================
# ALPHA SIGNAL C: LIQUIDITY SHOCK BREAKOUT (Gatheral)
# =============================================================================

def compute_alpha_c_liquidity_shock(
    df: pd.DataFrame,
    horizon_minutes: int,
    config: AlphaConfig = AlphaConfig()
) -> pd.Series:
    """
    Alpha C: Liquidity Shock Breakout Signal.
    
    Research basis: Bouchaud, Gatheral "no-dynamic-arbitrage"
    
    When liquidity collapses (spread widens significantly),
    follow-through moves increase. This captures liquidity-driven
    volatility events.
    
    Label = 1 if:
    - Spread expands significantly (> 1.5x normal)
    - Volatility increases
    - Follow-through move > 0.5σ
    """
    n = len(df)
    labels = np.zeros(n, dtype=int)
    
    mid = df["mid"].values
    spread_pct = df["spread_pct"].values
    sigma = df["sigma"].values if "sigma" in df.columns else np.ones(n) * 0.0001
    
    # Compute normalized spread (spread vs rolling median)
    spread_median = pd.Series(spread_pct).rolling(60, min_periods=30).median().values
    normalized_spread = np.where(
        spread_median > 0,
        spread_pct / spread_median,
        1.0
    )
    
    # Compute volatility increase
    sigma_short = pd.Series(np.log(mid / np.roll(mid, 1))).rolling(5, min_periods=3).std().values
    sigma_long = pd.Series(np.log(mid / np.roll(mid, 1))).rolling(30, min_periods=15).std().values
    vol_ratio = np.where(sigma_long > 0, sigma_short / sigma_long, 1.0)
    
    valid_end = n - horizon_minutes
    
    for i in range(60, valid_end):
        # Check for liquidity shock conditions
        spread_expanded = normalized_spread[i] > config.spread_expansion_threshold
        vol_increased = vol_ratio[i] > 1.2
        
        if not (spread_expanded and vol_increased):
            continue
        
        # Check for follow-through move
        future_mids = mid[i+1:i+1+horizon_minutes]
        if len(future_mids) == 0:
            continue
        
        sigma_t = sigma[i] if not np.isnan(sigma[i]) else 0.0001
        max_move = max(
            np.nanmax(future_mids) / mid[i] - 1,
            1 - np.nanmin(future_mids) / mid[i]
        )
        
        # Follow-through move > 0.5 sigma
        if max_move > 0.5 * sigma_t:
            labels[i] = 1
    
    return pd.Series(labels, index=df.index, name=f"alpha_c_{horizon_minutes}m")


# =============================================================================
# ALPHA SIGNAL D: VOLATILITY EXPANSION BREAKOUT (Andersen/Bollerslev)
# =============================================================================

def compute_alpha_d_volatility_breakout(
    df: pd.DataFrame,
    horizon_minutes: int,
    config: AlphaConfig = AlphaConfig()
) -> pd.Series:
    """
    Alpha D: Volatility Expansion Breakout Signal.
    
    Research basis: Andersen & Bollerslev, high-frequency vol forecasting
    
    When short-term volatility expands relative to long-term,
    breakout probability rises. This is standard HF breakout behavior.
    
    Label = 1 if:
    - σ_short / σ_long > threshold
    - Move > 0.5σ in any direction within horizon
    """
    n = len(df)
    labels = np.zeros(n, dtype=int)
    
    mid = df["mid"].values
    log_ret = np.log(mid / np.roll(mid, 1))
    log_ret[0] = 0
    
    # Short and long volatility
    sigma_short = pd.Series(log_ret).rolling(5, min_periods=3).std().values
    sigma_long = pd.Series(log_ret).rolling(30, min_periods=15).std().values
    
    vol_ratio = np.where(sigma_long > 0, sigma_short / sigma_long, 1.0)
    
    valid_end = n - horizon_minutes
    
    for i in range(30, valid_end):
        if vol_ratio[i] < config.vol_ratio_threshold:
            continue
        
        # Check for breakout move
        future_mids = mid[i+1:i+1+horizon_minutes]
        if len(future_mids) == 0:
            continue
        
        sigma_t = sigma_short[i] if not np.isnan(sigma_short[i]) else 0.0001
        max_move = max(
            np.nanmax(future_mids) / mid[i] - 1,
            1 - np.nanmin(future_mids) / mid[i]
        )
        
        # Breakout > 0.5 sigma
        if max_move > 0.5 * sigma_t:
            labels[i] = 1
    
    return pd.Series(labels, index=df.index, name=f"alpha_d_{horizon_minutes}m")


# =============================================================================
# MASTER ALPHA LABELING FUNCTION
# =============================================================================

def generate_all_alpha_labels(
    df: pd.DataFrame,
    horizons: Dict[str, int] = {"5m": 5, "15m": 15, "30m": 30},
    config: AlphaConfig = AlphaConfig(),
    verbose: bool = False
) -> pd.DataFrame:
    """
    Generate labels for all 4 alpha signals across all horizons.
    
    Returns DataFrame with columns:
    - alpha_a_5m, alpha_a_15m, alpha_a_30m (Order Flow)
    - alpha_b_5m, alpha_b_15m, alpha_b_30m (Microprice)
    - alpha_c_5m, alpha_c_15m, alpha_c_30m (Liquidity Shock)
    - alpha_d_5m, alpha_d_15m, alpha_d_30m (Vol Breakout)
    """
    df = df.copy()
    
    for horizon_name, horizon_minutes in horizons.items():
        if verbose:
            print(f"\nGenerating alpha labels for {horizon_name}...")
        
        # Alpha A: Order Flow
        alpha_a = compute_alpha_a_order_flow(df, horizon_minutes, config)
        df[f"alpha_a_{horizon_minutes}m"] = alpha_a
        if verbose:
            print(f"  Alpha A (Order Flow): {alpha_a.sum():,} signals ({100*alpha_a.mean():.1f}%)")
        
        # Alpha B: Microprice
        alpha_b = compute_alpha_b_microprice(df, horizon_minutes, config)
        df[f"alpha_b_{horizon_minutes}m"] = alpha_b
        if verbose:
            print(f"  Alpha B (Microprice): {alpha_b.sum():,} signals ({100*alpha_b.mean():.1f}%)")
        
        # Alpha C: Liquidity Shock
        alpha_c = compute_alpha_c_liquidity_shock(df, horizon_minutes, config)
        df[f"alpha_c_{horizon_minutes}m"] = alpha_c
        if verbose:
            print(f"  Alpha C (Liquidity): {alpha_c.sum():,} signals ({100*alpha_c.mean():.1f}%)")
        
        # Alpha D: Vol Breakout
        alpha_d = compute_alpha_d_volatility_breakout(df, horizon_minutes, config)
        df[f"alpha_d_{horizon_minutes}m"] = alpha_d
        if verbose:
            print(f"  Alpha D (Vol Breakout): {alpha_d.sum():,} signals ({100*alpha_d.mean():.1f}%)")
        
        # Combined: any alpha active
        df[f"any_alpha_{horizon_minutes}m"] = (
            (alpha_a == 1) | (alpha_b == 1) | (alpha_c == 1) | (alpha_d == 1)
        ).astype(int)
        if verbose:
            print(f"  Any Alpha: {df[f'any_alpha_{horizon_minutes}m'].sum():,} signals")
    
    return df


def get_alpha_statistics(df: pd.DataFrame) -> Dict:
    """Get statistics about alpha signal distributions."""
    stats = {}
    
    for col in df.columns:
        if col.startswith("alpha_") or col.startswith("any_alpha_"):
            signals = df[col]
            stats[col] = {
                "total_signals": int(signals.sum()),
                "signal_rate": float(signals.mean()),
            }
    
    return stats

