#!/usr/bin/env python3
"""
Diagnostics Script - Full Pipeline Audit

This script runs comprehensive diagnostics on the XAUUSD signal pipeline
to identify the root cause of:
1. Short bias
2. Negative expectancy
3. Sign/shift errors

Run this script to get a complete audit report.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from debug.sanity_checks import (
    load_year_data, check_mid_price_trend, check_forward_returns,
    baseline_trend_strategy, check_sl_tp_ordering,
    check_label_distribution, check_feature_directionality
)


def main():
    print("=" * 70)
    print("XAUUSD SIGNAL PIPELINE - FULL DIAGNOSTICS")
    print("=" * 70)
    print()
    
    # Paths
    DATA_DIR = PROJECT_ROOT.parent / "Data"
    MINUTE_DIR = DATA_DIR / "ohlcv_minute"
    QUOTES_DIR = DATA_DIR / "quotes"
    
    print(f"Data directory: {DATA_DIR}")
    print()
    
    # ==========================================================================
    # SECTION 1: Load 2024 Data
    # ==========================================================================
    print("SECTION 1: LOADING 2024 DATA")
    print("-" * 40)
    
    try:
        df = load_year_data(str(MINUTE_DIR), str(QUOTES_DIR), 2024)
        print(f"✓ Loaded {len(df):,} rows")
        print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    print()
    
    # ==========================================================================
    # SECTION 2: Mid Price and Trend Check
    # ==========================================================================
    print("\nSECTION 2: MID PRICE AND TREND")
    print("-" * 40)
    
    trend_results = check_mid_price_trend(df)
    
    # CRITICAL CHECK: 2024 was a bull year for gold
    if trend_results["total_return_pct"] < 10:
        print("\n⚠ CRITICAL WARNING:")
        print("   Gold gained ~25-30% in 2024, but this data shows less.")
        print("   This could indicate:")
        print("   - Wrong symbol/data source")
        print("   - Data processing error")
        print("   - Inverted bid/ask")
    
    print()
    
    # ==========================================================================
    # SECTION 3: Forward Returns
    # ==========================================================================
    print("\nSECTION 3: FORWARD RETURNS")
    print("-" * 40)
    
    for horizon in [5, 15, 30]:
        print(f"\n--- {horizon}-minute horizon ---")
        fwd_results = check_forward_returns(df, horizon)
    
    print()
    
    # ==========================================================================
    # SECTION 4: Baseline Trend Strategy
    # ==========================================================================
    print("\nSECTION 4: BASELINE TREND STRATEGY")
    print("-" * 40)
    print("Testing if past and future returns are aligned correctly...")
    print()
    
    baseline_issues = []
    for horizon in [5, 15, 30]:
        print(f"\n--- {horizon}-minute horizon ---")
        baseline_results = baseline_trend_strategy(df, horizon)
        
        if baseline_results["mean_strat_ret"] < -0.0001:
            baseline_issues.append(f"{horizon}m: Baseline very negative ({baseline_results['mean_strat_ret']*100:.4f}%)")
    
    if baseline_issues:
        print("\n❌ BASELINE ISSUES DETECTED:")
        for issue in baseline_issues:
            print(f"   - {issue}")
        print("\n   This suggests a shift/sign bug in return calculations!")
    else:
        print("\n✓ Baseline strategies are not anomalously negative")
    
    print()
    
    # ==========================================================================
    # SECTION 5: Build Features and Check
    # ==========================================================================
    print("\nSECTION 5: FEATURE ENGINEERING CHECK")
    print("-" * 40)
    
    # Build basic features
    df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    df["spread"] = df["ask_price"] - df["bid_price"]
    df["spread_pct"] = df["spread"] / df["mid"]
    df["log_ret"] = np.log(df["mid"] / df["mid"].shift(1))
    
    # Volatility
    df["sigma"] = df["log_ret"].shift(1).rolling(60, min_periods=30).std()
    
    # Momentum (LAGGED - no look-ahead)
    df["momentum_5"] = df["mid"].shift(1).pct_change(5)
    df["momentum_15"] = df["mid"].shift(1).pct_change(15)
    df["momentum_30"] = df["mid"].shift(1).pct_change(30)
    
    # VWAP distance
    if "vwap" in df.columns:
        df["vwap_deviation"] = (df["mid"] - df["vwap"].shift(1)) / df["vwap"].shift(1)
    
    print("Features built:")
    print(f"  - mid, spread, spread_pct")
    print(f"  - log_ret, sigma")
    print(f"  - momentum_5, momentum_15, momentum_30")
    
    # Check feature stats
    print("\nFeature statistics:")
    for col in ["mid", "spread_pct", "sigma", "momentum_5"]:
        if col in df.columns:
            print(f"  {col}: mean={df[col].mean():.6f}, std={df[col].std():.6f}")
    
    print()
    
    # ==========================================================================
    # SECTION 6: SL/TP Ordering
    # ==========================================================================
    print("\nSECTION 6: SL/TP ORDERING")
    print("-" * 40)
    
    # Compute SL/TP
    k1, k2 = 1.0, 2.0
    df["sl_price_long"] = df["mid"] * (1 - k1 * df["sigma"])
    df["tp_price_long"] = df["mid"] * (1 + k2 * df["sigma"])
    df["sl_price_short"] = df["mid"] * (1 + k1 * df["sigma"])
    df["tp_price_short"] = df["mid"] * (1 - k2 * df["sigma"])
    
    sltp_results = check_sl_tp_ordering(df)
    
    print()
    
    # ==========================================================================
    # SECTION 7: Feature Directionality
    # ==========================================================================
    print("\nSECTION 7: FEATURE DIRECTIONALITY")
    print("-" * 40)
    
    dir_issues = []
    for horizon in [5, 15, 30]:
        print(f"\n--- {horizon}-minute forward returns ---")
        dir_results = check_feature_directionality(
            df, horizon, 
            ["log_ret", "momentum_5", "momentum_15", "momentum_30"]
        )
        
        # Check for unexpected signs
        for feat, res in dir_results.items():
            if res.get("correlation") is not None:
                corr = res["correlation"]
                if feat.startswith("momentum") and corr < -0.02:
                    dir_issues.append(f"{feat} has negative corr ({corr:.4f}) with {horizon}m returns")
    
    if dir_issues:
        print("\n⚠ DIRECTIONALITY ISSUES:")
        for issue in dir_issues:
            print(f"   - {issue}")
    
    print()
    
    # ==========================================================================
    # SECTION 8: Simulate Simple Labels
    # ==========================================================================
    print("\nSECTION 8: LABEL SIMULATION")
    print("-" * 40)
    
    # Create simple labels based on forward returns
    for horizon in [5, 15, 30]:
        df[f"fwd_ret_{horizon}m"] = df["mid"].shift(-horizon) / df["mid"] - 1.0
        
        # Simple threshold-based labels
        threshold = 0.001  # 0.1%
        df[f"y_{horizon}m"] = 0
        df.loc[df[f"fwd_ret_{horizon}m"] > threshold, f"y_{horizon}m"] = 1
        df.loc[df[f"fwd_ret_{horizon}m"] < -threshold, f"y_{horizon}m"] = -1
    
    label_results = check_label_distribution(df, ["y_5m", "y_15m", "y_30m"])
    
    print()
    
    # ==========================================================================
    # SECTION 9: Direction Bias Analysis
    # ==========================================================================
    print("\nSECTION 9: DIRECTION BIAS ANALYSIS")
    print("-" * 40)
    
    for horizon in [5, 15, 30]:
        fwd_col = f"fwd_ret_{horizon}m"
        valid = df[fwd_col].dropna()
        
        n_up = (valid > 0).sum()
        n_down = (valid < 0).sum()
        total = len(valid)
        
        print(f"\n{horizon}m forward returns:")
        print(f"  Up:   {n_up:,} ({100*n_up/total:.1f}%)")
        print(f"  Down: {n_down:,} ({100*n_down/total:.1f}%)")
        
        if n_up / total > 0.52:
            print(f"  ✓ Slight upward bias (consistent with bull market)")
        elif n_down / total > 0.52:
            print(f"  ⚠ Downward bias despite bull market!")
        else:
            print(f"  ~ Roughly balanced")
    
    print()
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)
    
    critical_issues = []
    warnings = []
    
    # Check trend
    if trend_results["total_return_pct"] < 10:
        warnings.append("2024 price gain less than expected (~25-30%)")
    if trend_results["total_return_pct"] < 0:
        critical_issues.append("Price DOWN in 2024 (should be UP)")
    
    # Check baseline
    if baseline_issues:
        critical_issues.append("Baseline trend strategy anomalously negative")
    
    # Check SL/TP
    if "error" not in sltp_results:
        if sltp_results.get("long_fully_correct_pct", 100) < 95:
            critical_issues.append("LONG SL/TP ordering incorrect")
        if sltp_results.get("short_fully_correct_pct", 100) < 95:
            critical_issues.append("SHORT SL/TP ordering incorrect")
    
    # Check directionality
    if dir_issues:
        warnings.extend(dir_issues)
    
    if critical_issues:
        print("\n❌ CRITICAL ISSUES:")
        for issue in critical_issues:
            print(f"   - {issue}")
    
    if warnings:
        print("\n⚠ WARNINGS:")
        for warn in warnings:
            print(f"   - {warn}")
    
    if not critical_issues and not warnings:
        print("\n✓ No critical issues detected")
        print("  The pipeline math appears correct.")
        print("  If results are still poor, the issue may be:")
        print("  - Model capacity/features insufficient")
        print("  - Strategy edge doesn't exist in this data")
        print("  - Overfitting to training data")
    
    print()
    print("=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

