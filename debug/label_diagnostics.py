#!/usr/bin/env python3
"""
Label Diagnostics - Validate Microstructure-Based Labels.

This script:
1. Loads 2024 data
2. Builds microstructure features
3. Generates labels (y_5m, y_15m, y_30m)
4. Validates label distributions
5. Checks directional drift using forward returns (sanity only)

HARD FAILS if:
- Any horizon has > 85% class 0
- Any horizon has < 5% longs or < 5% shorts
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from feature_engineering import build_feature_matrix
from labeling import (
    generate_labels_for_all_horizons,
    validate_label_distribution,
    check_label_directional_drift
)


def load_2024_data(minute_dir: str, quotes_dir: str) -> pd.DataFrame:
    """Load 2024 data."""
    minute_path = Path(minute_dir) / "XAUUSD_minute_2024.parquet"
    quotes_path = Path(quotes_dir) / "XAUUSD_quotes_2024.parquet"
    
    if not minute_path.exists():
        raise FileNotFoundError(f"Minute data not found: {minute_path}")
    if not quotes_path.exists():
        raise FileNotFoundError(f"Quotes data not found: {quotes_path}")
    
    m = pd.read_parquet(minute_path)
    q = pd.read_parquet(quotes_path)
    
    for d in [m, q]:
        if "timestamp" in d.columns:
            d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
            d.set_index("timestamp", inplace=True)
    
    m = m.reset_index()
    q = q.reset_index()
    
    # Check what columns are available in quotes
    quote_cols = ["timestamp", "bid_price", "ask_price"]
    if "bid_size" in q.columns:
        quote_cols.append("bid_size")
    if "ask_size" in q.columns:
        quote_cols.append("ask_size")
    
    df = pd.merge_asof(
        m.sort_values("timestamp"),
        q.sort_values("timestamp")[quote_cols],
        on="timestamp",
        direction="backward"
    ).set_index("timestamp")
    
    return df.dropna(subset=["bid_price", "ask_price"])


def main():
    print("=" * 70)
    print("LABEL DIAGNOSTICS - Microstructure-Based Labels")
    print("=" * 70)
    print()
    
    # Paths
    DATA_DIR = PROJECT_ROOT.parent / "Data"
    MINUTE_DIR = DATA_DIR / "ohlcv_minute"
    QUOTES_DIR = DATA_DIR / "quotes"
    
    # ==========================================================================
    # STEP 1: Load Data
    # ==========================================================================
    print("STEP 1: Loading 2024 Data")
    print("-" * 40)
    
    try:
        df = load_2024_data(str(MINUTE_DIR), str(QUOTES_DIR))
        print(f"✓ Loaded {len(df):,} rows")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return False
    
    print()
    
    # ==========================================================================
    # STEP 2: Build Features
    # ==========================================================================
    print("STEP 2: Building Microstructure Features")
    print("-" * 40)
    
    df = build_feature_matrix(df)
    
    # Check for key features
    features_status = {
        "mid": "mid" in df.columns,
        "spread_pct": "spread_pct" in df.columns,
        "imbalance": "imbalance" in df.columns and not df["imbalance"].isna().all(),
        "microprice": "microprice" in df.columns and not df["microprice"].isna().all(),
        "sigma_slope": "sigma_slope" in df.columns,
        "spread_med_60": "spread_med_60" in df.columns,
    }
    
    for feat, status in features_status.items():
        if status:
            print(f"  ✓ {feat}")
        else:
            print(f"  ⚠ {feat} (missing or all NaN)")
    
    if not features_status["imbalance"]:
        print("\n⚠ WARNING: No bid_size/ask_size in data.")
        print("  Imbalance and microprice cannot be computed.")
        print("  Falling back to momentum-based labels.")
    
    print()
    
    # ==========================================================================
    # STEP 3: Generate Labels
    # ==========================================================================
    print("STEP 3: Generating Microstructure-Based Labels")
    print("-" * 40)
    
    df = generate_labels_for_all_horizons(df, verbose=True)
    
    print()
    
    # ==========================================================================
    # STEP 4: Validate Label Distribution
    # ==========================================================================
    print("STEP 4: Validating Label Distribution")
    print("-" * 40)
    
    passed, details = validate_label_distribution(df)
    
    print(f"\nDistributions:")
    for col, dist in details["distributions"].items():
        print(f"\n  {col}:")
        print(f"    LONG:  {dist['long']*100:.1f}%")
        print(f"    SHORT: {dist['short']*100:.1f}%")
        print(f"    FLAT:  {dist['flat']*100:.1f}%")
    
    if passed:
        print("\n✓ Label distribution PASSED")
    else:
        print("\n❌ Label distribution FAILED:")
        for issue in details["issues"]:
            print(f"  - {issue}")
    
    print()
    
    # ==========================================================================
    # STEP 5: Check Directional Drift (Sanity Check)
    # ==========================================================================
    print("STEP 5: Checking Directional Drift (Forward Returns)")
    print("-" * 40)
    print("Note: This uses future returns for VALIDATION ONLY, not training.")
    
    drift_issues = []
    
    for horizon in [5, 15, 30]:
        results = check_label_directional_drift(df, horizon)
        
        print(f"\n  {horizon}m Horizon:")
        print(f"    LONG signals:  n={results['n_long']:,}, mean fwd ret = {results['mean_fwd_ret_long']*100:.4f}%")
        print(f"    SHORT signals: n={results['n_short']:,}, mean fwd ret = {results['mean_fwd_ret_short']*100:.4f}%")
        print(f"    (SHORT inverted for comparison: {results['mean_fwd_ret_short_inverted']*100:.4f}%)")
        
        if results["warnings"]:
            for warn in results["warnings"]:
                print(f"    ⚠ {warn}")
                drift_issues.append(f"{horizon}m: {warn}")
    
    print()
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    all_passed = True
    
    # Check 1: Label distribution
    if passed:
        print("\n✓ Label distribution: PASSED")
    else:
        print("\n❌ Label distribution: FAILED")
        for issue in details["issues"]:
            print(f"    - {issue}")
        all_passed = False
    
    # Check 2: Directional drift
    if not drift_issues:
        print("\n✓ Directional drift: No major issues")
    else:
        print("\n⚠ Directional drift: Warnings detected")
        for issue in drift_issues:
            print(f"    - {issue}")
    
    # Check 3: Feature availability
    if features_status["imbalance"]:
        print("\n✓ Imbalance feature: Available")
    else:
        print("\n⚠ Imbalance feature: Using momentum fallback")
    
    print("\n" + "=" * 70)
    
    if all_passed:
        print("RESULT: All critical checks PASSED")
        return True
    else:
        print("RESULT: Some checks FAILED - labels may cause model collapse")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

