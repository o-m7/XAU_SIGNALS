"""
Labeling Module - Microstructure-Driven Directional Labels.

Option C Implementation:
Labels are based on microstructure signals at time t (no future data).
This produces dense, learnable labels without class collapse.

Label Rules (SYMMETRIC):
  y = +1 (LONG setup) when:
    - imbalance > +θ_imb
    - microprice > mid
    - sigma_slope > θ_sigma
    - spread_pct < θ_spread * spread_med_60

  y = -1 (SHORT setup) when:
    - imbalance < -θ_imb
    - microprice < mid
    - sigma_slope > θ_sigma
    - spread_pct < θ_spread * spread_med_60

  y = 0 otherwise

NO LEAKAGE: Labels use only time-t features, never future returns.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import warnings


# =============================================================================
# HORIZON-SPECIFIC THRESHOLDS
# =============================================================================

HORIZON_THRESHOLDS = {
    5: {
        "imb_threshold": 0.12,       # Looser for 5m (more signals)
        "sigma_slope_threshold": 0.0,
        "spread_mult": 1.25,
    },
    15: {
        "imb_threshold": 0.15,
        "sigma_slope_threshold": 0.0,
        "spread_mult": 1.20,
    },
    30: {
        "imb_threshold": 0.18,       # Stricter for 30m (fewer, higher quality)
        "sigma_slope_threshold": 0.0,
        "spread_mult": 1.15,
    },
}


# =============================================================================
# MICROSTRUCTURE SETUP LABELING
# =============================================================================

def label_microstructure_setups(
    df: pd.DataFrame,
    horizon: int,
    imb_threshold: float = None,
    sigma_slope_threshold: float = None,
    spread_mult: float = None,
) -> pd.Series:
    """
    Generate directional microstructure setup labels using only time-t features.
    
    NO LEAKAGE: This function does NOT use future returns for labeling.
    Horizon is used only to select thresholds.
    
    Args:
        df: DataFrame with microstructure features
        horizon: Horizon in minutes (5, 15, or 30) for threshold selection
        imb_threshold: Override imbalance threshold
        sigma_slope_threshold: Override sigma slope threshold
        spread_mult: Override spread multiplier
        
    Returns:
        Series of labels: +1 (LONG), -1 (SHORT), 0 (FLAT)
    """
    # Get thresholds
    if horizon in HORIZON_THRESHOLDS:
        defaults = HORIZON_THRESHOLDS[horizon]
    else:
        defaults = HORIZON_THRESHOLDS[15]  # fallback
    
    imb_th = imb_threshold if imb_threshold is not None else defaults["imb_threshold"]
    sigma_th = sigma_slope_threshold if sigma_slope_threshold is not None else defaults["sigma_slope_threshold"]
    spread_m = spread_mult if spread_mult is not None else defaults["spread_mult"]
    
    n = len(df)
    labels = np.zeros(n, dtype=int)
    
    # Check required columns
    required = ["mid", "spread_pct", "sigma_slope", "spread_med_60"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check for imbalance/microprice
    has_imbalance = "imbalance" in df.columns and not df["imbalance"].isna().all()
    has_microprice = "microprice" in df.columns and not df["microprice"].isna().all()
    
    if not has_imbalance:
        warnings.warn(
            "Imbalance column is missing or all NaN. "
            "Using momentum as fallback for direction signal.",
            UserWarning
        )
    
    # Get arrays for vectorized operations
    mid = df["mid"].values
    spread_pct = df["spread_pct"].values
    sigma_slope = df["sigma_slope"].values
    spread_med = df["spread_med_60"].values
    
    if has_imbalance:
        imbalance = df["imbalance"].values
    else:
        # Fallback: use momentum as imbalance proxy
        # Scale momentum to match typical imbalance range [-1, 1]
        if "momentum_5" in df.columns:
            mom = df["momentum_5"].values
            mom_std = np.nanstd(mom[~np.isnan(mom)])
            if mom_std > 0:
                imbalance = (mom / mom_std) * 0.5  # Scale to ~[-1, 1] range
            else:
                imbalance = np.zeros(n)
        else:
            imbalance = np.zeros(n)
    
    if has_microprice:
        microprice = df["microprice"].values
        use_microprice_condition = True
    else:
        # Fallback: skip microprice condition entirely
        microprice = mid
        use_microprice_condition = False
    
    # Compute spread regime threshold
    spread_threshold = spread_m * spread_med
    
    # Vectorized labeling (SYMMETRIC rules)
    for i in range(n):
        # Skip if any required value is NaN
        if (np.isnan(mid[i]) or np.isnan(spread_pct[i]) or 
            np.isnan(sigma_slope[i]) or np.isnan(spread_med[i]) or
            np.isnan(imbalance[i]) or np.isnan(microprice[i])):
            continue
        
        # Common conditions
        vol_ok = sigma_slope[i] > sigma_th
        spread_ok = spread_pct[i] < spread_threshold[i]
        
        if not (vol_ok and spread_ok):
            continue
        
        # LONG conditions (SYMMETRIC with SHORT)
        imb_long = imbalance[i] > imb_th
        
        # SHORT conditions (SYMMETRIC with LONG)
        imb_short = imbalance[i] < -imb_th
        
        # Apply microprice condition only if available
        if use_microprice_condition:
            micro_long = microprice[i] > mid[i]
            micro_short = microprice[i] < mid[i]
            
            if imb_long and micro_long:
                labels[i] = 1
            elif imb_short and micro_short:
                labels[i] = -1
        else:
            # Without microprice, use imbalance/momentum only
            if imb_long:
                labels[i] = 1
            elif imb_short:
                labels[i] = -1
    
    return pd.Series(labels, index=df.index, name=f"y_{horizon}m")


def generate_labels_for_all_horizons(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate microstructure-based labels for all horizons.
    
    Adds columns: y_5m, y_15m, y_30m
    
    NO LEAKAGE: Uses only time-t features.
    """
    df = df.copy()
    
    for horizon in [5, 15, 30]:
        labels = label_microstructure_setups(df, horizon)
        df[f"y_{horizon}m"] = labels
        
        if verbose:
            dist = labels.value_counts(normalize=True)
            n_long = (labels == 1).sum()
            n_short = (labels == -1).sum()
            n_flat = (labels == 0).sum()
            total = len(labels)
            
            print(f"\n{horizon}m Labels:")
            print(f"  LONG (+1):  {n_long:,} ({100*n_long/total:.1f}%)")
            print(f"  SHORT (-1): {n_short:,} ({100*n_short/total:.1f}%)")
            print(f"  FLAT (0):   {n_flat:,} ({100*n_flat/total:.1f}%)")
    
    return df


# =============================================================================
# LABEL VALIDATION
# =============================================================================

def validate_label_distribution(
    df: pd.DataFrame,
    label_cols: list = ["y_5m", "y_15m", "y_30m"],
    max_flat_pct: float = 0.85,
    min_directional_pct: float = 0.05
) -> Tuple[bool, Dict]:
    """
    Validate that label distributions are acceptable.
    
    FAILS if:
    - Any horizon has > max_flat_pct class 0
    - Any horizon has < min_directional_pct longs or shorts
    
    Returns:
        (passed: bool, details: dict)
    """
    results = {"passed": True, "issues": [], "distributions": {}}
    
    for col in label_cols:
        if col not in df.columns:
            results["issues"].append(f"Missing column: {col}")
            results["passed"] = False
            continue
        
        labels = df[col]
        total = len(labels)
        
        n_long = (labels == 1).sum()
        n_short = (labels == -1).sum()
        n_flat = (labels == 0).sum()
        
        pct_long = n_long / total
        pct_short = n_short / total
        pct_flat = n_flat / total
        
        results["distributions"][col] = {
            "long": pct_long,
            "short": pct_short,
            "flat": pct_flat,
        }
        
        # Check for issues
        if pct_flat > max_flat_pct:
            results["issues"].append(
                f"{col}: {pct_flat*100:.1f}% FLAT (max allowed: {max_flat_pct*100:.0f}%)"
            )
            results["passed"] = False
        
        if pct_long < min_directional_pct:
            results["issues"].append(
                f"{col}: Only {pct_long*100:.1f}% LONG (min required: {min_directional_pct*100:.0f}%)"
            )
            results["passed"] = False
        
        if pct_short < min_directional_pct:
            results["issues"].append(
                f"{col}: Only {pct_short*100:.1f}% SHORT (min required: {min_directional_pct*100:.0f}%)"
            )
            results["passed"] = False
    
    return results["passed"], results


def check_label_directional_drift(
    df: pd.DataFrame,
    horizon: int,
    label_col: str = None
) -> Dict:
    """
    Sanity check: verify that labels have correct directional drift.
    
    Uses FORWARD RETURNS for validation only (NOT for training).
    
    Expectation:
    - mean(fwd_return | y=+1) should be >= 0
    - mean(-fwd_return | y=-1) should be >= 0 (inverted for comparability)
    
    If either is negative, warns about potential signal inversion.
    """
    if label_col is None:
        label_col = f"y_{horizon}m"
    
    if label_col not in df.columns:
        return {"error": f"Label column {label_col} not found"}
    
    if "mid" not in df.columns:
        return {"error": "mid column not found"}
    
    # Compute forward returns (FOR VALIDATION ONLY)
    fwd_ret_col = f"fwd_ret_{horizon}m"
    df = df.copy()
    df[fwd_ret_col] = df["mid"].shift(-horizon) / df["mid"] - 1.0
    
    # Get returns by label
    long_mask = df[label_col] == 1
    short_mask = df[label_col] == -1
    
    long_returns = df.loc[long_mask, fwd_ret_col].dropna()
    short_returns = df.loc[short_mask, fwd_ret_col].dropna()
    
    results = {
        "horizon": horizon,
        "n_long": len(long_returns),
        "n_short": len(short_returns),
        "mean_fwd_ret_long": float(long_returns.mean()) if len(long_returns) > 0 else np.nan,
        "mean_fwd_ret_short": float(short_returns.mean()) if len(short_returns) > 0 else np.nan,
        "warnings": [],
    }
    
    # Inverted short return for comparison
    results["mean_fwd_ret_short_inverted"] = -results["mean_fwd_ret_short"]
    
    # Check for issues
    if results["n_long"] > 100 and results["mean_fwd_ret_long"] < 0:
        results["warnings"].append(
            f"LONG signals have negative forward returns ({results['mean_fwd_ret_long']*100:.3f}%)"
        )
    
    if results["n_short"] > 100 and results["mean_fwd_ret_short"] > 0:
        # For shorts, positive fwd_ret means we lost money
        results["warnings"].append(
            f"SHORT signals have positive forward returns ({results['mean_fwd_ret_short']*100:.3f}%)"
        )
    
    return results


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def get_label_statistics(df: pd.DataFrame) -> Dict:
    """Get statistics about label distributions."""
    stats = {}
    for col in ["y_5m", "y_15m", "y_30m"]:
        if col not in df.columns:
            continue
        labels = df[col]
        stats[col] = {
            "n_long": int((labels == 1).sum()),
            "n_short": int((labels == -1).sum()),
            "n_flat": int((labels == 0).sum()),
            "pct_long": float((labels == 1).mean() * 100),
            "pct_short": float((labels == -1).mean() * 100),
            "pct_flat": float((labels == 0).mean() * 100),
        }
    return stats
