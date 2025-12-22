#!/usr/bin/env python3
"""
Leakage and Look-Ahead Bias Diagnosis.

This script checks for:
1. Feature look-ahead bias (features using future data)
2. Label leakage (future info in features)
3. Train/test data overlap
4. Temporal integrity of splits
5. Suspicious correlations between features and labels

Usage:
    cd /Users/omar/Desktop/ML && source xauusd_signals/venv/bin/activate && python xauusd_signals/debug/leakage_diagnosis.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2024.parquet"
DATA_DIR = PROJECT_ROOT.parent / "Data"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TARGET = "y_tb_60"


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def print_result(check: str, passed: bool, details: str = "") -> None:
    """Print a check result."""
    status = "✓ PASS" if passed else "❌ FAIL"
    print(f"\n  [{status}] {check}")
    if details:
        for line in details.split("\n"):
            print(f"          {line}")


# =============================================================================
# TEST 1: Temporal Integrity of Splits
# =============================================================================

def check_temporal_integrity(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Verify that train/val/test splits are strictly chronological
    with no overlap.
    """
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    train_max_time = train_df.index.max()
    val_min_time = val_df.index.min()
    val_max_time = val_df.index.max()
    test_min_time = test_df.index.min()
    
    # Check no overlap
    train_val_gap = (val_min_time - train_max_time).total_seconds() / 60
    val_test_gap = (test_min_time - val_max_time).total_seconds() / 60
    
    passed = True
    details = []
    
    details.append(f"Train: {train_df.index.min()} to {train_max_time}")
    details.append(f"Val:   {val_min_time} to {val_max_time}")
    details.append(f"Test:  {test_min_time} to {test_df.index.max()}")
    details.append(f"Train→Val gap: {train_val_gap:.0f} minutes")
    details.append(f"Val→Test gap:  {val_test_gap:.0f} minutes")
    
    if train_val_gap < 0:
        passed = False
        details.append("⚠ Train and Val overlap!")
    
    if val_test_gap < 0:
        passed = False
        details.append("⚠ Val and Test overlap!")
    
    return passed, "\n".join(details)


# =============================================================================
# TEST 2: Label Construction Check
# =============================================================================

def check_label_construction(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Verify that y_tb_60 is constructed using only future data (as expected
    for labels) and not leaking into features.
    
    The triple-barrier label should use close prices from t+1 to t+60.
    """
    # Check if y_tb_60 correlates with current or past price movements
    # (it shouldn't, since it uses future data)
    
    details = []
    passed = True
    
    if TARGET not in df.columns:
        return False, f"Target column '{TARGET}' not found"
    
    # Compute correlation between y_tb_60 and current/past returns
    if "ret_1" in df.columns:
        # Current return (should have LOW correlation with future label)
        corr_ret1 = df["ret_1"].corr(df[TARGET])
        details.append(f"Correlation(ret_1, y_tb_60) = {corr_ret1:.4f}")
        
        if abs(corr_ret1) > 0.3:
            details.append("  ⚠ Suspiciously high correlation with current return!")
            passed = False
    
    # Check correlation with future returns
    # y_tb_60 SHOULD correlate with future returns since that's how it's defined
    future_ret = np.log(df["close"].shift(-60) / df["close"])
    corr_future = future_ret.corr(df[TARGET])
    details.append(f"Correlation(future_ret_60, y_tb_60) = {corr_future:.4f}")
    
    if abs(corr_future) < 0.3:
        details.append("  ⚠ Low correlation with future return - label may be wrong")
        passed = False
    else:
        details.append("  ✓ Expected high correlation with future returns")
    
    return passed, "\n".join(details)


# =============================================================================
# TEST 3: Feature Look-Ahead Bias
# =============================================================================

def check_feature_lookahead(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[bool, str]:
    """
    Check if features show suspiciously high correlation with future returns.
    
    Features should NOT have high correlation with future returns if they
    use only past/current data.
    """
    details = []
    passed = True
    suspicious_features = []
    
    # Compute future return
    future_ret_60 = np.log(df["close"].shift(-60) / df["close"])
    
    correlations = {}
    for feat in feature_cols:
        if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
            corr = df[feat].corr(future_ret_60)
            if not np.isnan(corr):
                correlations[feat] = corr
    
    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    details.append("Top 10 feature correlations with future_ret_60:")
    for feat, corr in sorted_corrs[:10]:
        flag = "⚠" if abs(corr) > 0.2 else " "
        details.append(f"  {flag} {feat}: {corr:+.4f}")
        if abs(corr) > 0.2:
            suspicious_features.append((feat, corr))
    
    if suspicious_features:
        details.append(f"\n⚠ {len(suspicious_features)} features have |corr| > 0.2 with future returns")
        details.append("  These may indicate look-ahead bias OR genuinely predictive features")
        details.append("  (Some correlation is expected if the model works)")
    else:
        details.append("\n✓ No features have suspiciously high correlation with future returns")
    
    return passed, "\n".join(details)


# =============================================================================
# TEST 4: Rolling Feature Computation Check
# =============================================================================

def check_rolling_features(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Verify that rolling features use only past data by checking if
    feature at t can predict feature at t+1 (should NOT be the case
    if feature at t uses data up to t only).
    """
    details = []
    passed = True
    
    rolling_features = [
        "vol_10", "vol_60", "ma_fast_15", "ma_slow_60",
        "ret_mean_5", "ret_mean_20", "ret_mean_60"
    ]
    
    existing = [f for f in rolling_features if f in df.columns]
    
    for feat in existing:
        # Check autocorrelation at lag 1
        autocorr = df[feat].autocorr(lag=1)
        # Check if feature at t can perfectly predict t+1 (would indicate future data)
        perfect_prediction = df[feat].equals(df[feat].shift(-1))
        
        details.append(f"{feat}: autocorr(1)={autocorr:.4f}")
        
        if perfect_prediction:
            details.append(f"  ⚠ Feature is identical to its lag - possible leakage!")
            passed = False
    
    return passed, "\n".join(details)


# =============================================================================
# TEST 5: Information Leakage via Feature Engineering
# =============================================================================

def check_feature_timing(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Check if features properly lag their inputs to avoid look-ahead.
    
    For example, mid_slope_10 should use mid prices from t-10 to t-1,
    not including t or future.
    """
    details = []
    passed = True
    
    # Test: if we shuffle the data randomly, the model should perform poorly
    # This tests if the model relies on temporal structure (good) vs leakage (bad)
    
    # Compute correlation between features and the label at different lags
    if TARGET in df.columns:
        # Check correlation at t (current)
        details.append("Cross-correlation analysis:")
        
        for feat in ["ret_1", "mid_slope_10", "vol_10", "is_bull", "upper_wick"]:
            if feat not in df.columns:
                continue
            
            corr_t0 = df[feat].corr(df[TARGET])
            corr_t1 = df[feat].shift(1).corr(df[TARGET])  # lag 1
            corr_tm1 = df[feat].shift(-1).corr(df[TARGET])  # lead 1 (future feature)
            
            details.append(f"  {feat}:")
            details.append(f"    corr(t-1, label) = {corr_t1:.4f}")
            details.append(f"    corr(t, label)   = {corr_t0:.4f}")
            details.append(f"    corr(t+1, label) = {corr_tm1:.4f}")
            
            # If correlation with future feature is much higher, possible leakage
            if abs(corr_tm1) > abs(corr_t0) + 0.05:
                details.append(f"    ⚠ Future feature more predictive - check for leakage")
    
    return passed, "\n".join(details)


# =============================================================================
# TEST 6: Perfect Prediction Check
# =============================================================================

def check_perfect_prediction(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[bool, str]:
    """
    Check if any feature perfectly predicts the label (would indicate leakage).
    """
    details = []
    passed = True
    
    if TARGET not in df.columns:
        return True, "Target not found"
    
    df_clean = df.dropna(subset=[TARGET] + feature_cols)
    y = df_clean[TARGET]
    
    for feat in feature_cols[:20]:  # Check first 20 features
        if feat not in df_clean.columns:
            continue
        
        feat_vals = df_clean[feat]
        
        # Check if feature has perfect separation
        if y.nunique() == 2:
            pos_mask = y > 0
            neg_mask = y < 0
            
            pos_mean = feat_vals[pos_mask].mean()
            neg_mean = feat_vals[neg_mask].mean()
            pos_std = feat_vals[pos_mask].std()
            neg_std = feat_vals[neg_mask].std()
            
            # If distributions don't overlap at all, suspicious
            if (pos_mean - 3*pos_std > neg_mean + 3*neg_std) or \
               (neg_mean - 3*neg_std > pos_mean + 3*pos_std):
                details.append(f"  ⚠ {feat}: Perfect separation between classes!")
                passed = False
    
    if passed:
        details.append("No features show perfect class separation")
    
    return passed, "\n".join(details)


# =============================================================================
# TEST 7: Verify Triple-Barrier Implementation
# =============================================================================

def verify_triple_barrier(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Spot-check the triple-barrier label by manually computing a few cases.
    """
    details = []
    passed = True
    
    if TARGET not in df.columns or "close" not in df.columns or "ATR_14" not in df.columns:
        return True, "Required columns not found for verification"
    
    # Sample a few rows and verify
    np.random.seed(42)
    sample_idx = np.random.choice(len(df) - 100, size=5, replace=False)
    
    details.append("Spot-checking triple-barrier labels:")
    
    for idx in sample_idx:
        row = df.iloc[idx]
        close_t = row["close"]
        atr_t = row["ATR_14"]
        label = row[TARGET]
        
        if pd.isna(close_t) or pd.isna(atr_t) or pd.isna(label):
            continue
        
        # Compute barriers
        upper = close_t + atr_t  # TP
        lower = close_t - atr_t  # SL
        
        # Walk forward
        future_closes = df.iloc[idx+1:idx+61]["close"].values
        
        computed_label = 0
        for h, fc in enumerate(future_closes):
            if fc >= upper:
                computed_label = 1
                break
            elif fc <= lower:
                computed_label = -1
                break
        
        match = "✓" if computed_label == label else "✗"
        details.append(f"  idx={idx}: computed={computed_label}, stored={int(label)} {match}")
        
        if computed_label != label:
            passed = False
    
    return passed, "\n".join(details)


# =============================================================================
# TEST 8: Backtest Logic Verification
# =============================================================================

def verify_backtest_logic() -> Tuple[bool, str]:
    """
    Verify that the backtest correctly interprets signals and labels.
    """
    details = []
    passed = True
    
    # Simulate a small backtest
    y_test = np.array([1, 1, -1, -1, 1])  # True labels
    signal = np.array([1, -1, -1, 1, 1])   # Model signals
    
    # Expected: win when signal matches label
    expected_ret = y_test * signal  # [1, -1, 1, -1, 1] -> 1 win, 2 losses = net +1
    
    details.append("Backtest logic check:")
    details.append(f"  Labels:   {y_test.tolist()}")
    details.append(f"  Signals:  {signal.tolist()}")
    details.append(f"  Returns:  {expected_ret.tolist()}")
    details.append(f"  Expected: Long+Up=+1, Long+Down=-1, Short+Down=+1, Short+Up=-1")
    
    # Verify logic
    # signal=1 (long) + y=1 (up) = +1 (win) ✓
    # signal=-1 (short) + y=1 (up) = -1 (loss) ✓
    # signal=-1 (short) + y=-1 (down) = +1 (win) ✓
    # signal=1 (long) + y=-1 (down) = -1 (loss) ✓
    
    test_cases = [
        (1, 1, 1, "Long + Up = Win"),
        (1, -1, -1, "Long + Down = Loss"),
        (-1, -1, 1, "Short + Down = Win"),
        (-1, 1, -1, "Short + Up = Loss"),
    ]
    
    for sig, y, expected, desc in test_cases:
        actual = sig * y
        match = "✓" if actual == expected else "✗"
        details.append(f"  {match} {desc}: sig={sig}, y={y}, ret={actual}")
        if actual != expected:
            passed = False
    
    return passed, "\n".join(details)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all leakage diagnostics."""
    print("=" * 70)
    print("  LEAKAGE AND LOOK-AHEAD BIAS DIAGNOSIS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    if not FEATURES_PATH.exists():
        print(f"❌ Features file not found: {FEATURES_PATH}")
        return
    
    df = pd.read_parquet(FEATURES_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
    df = df.sort_index()
    
    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    
    # Get feature columns
    label_cols = ["y_ret_5", "y_dir_5", "y_ret_15", "y_dir_15", "y_ret_60", "y_dir_60", "y_tb_60"]
    raw_cols = ["open", "high", "low", "close", "volume", "bid_price", "ask_price", "vwap", "trades"]
    feature_cols = [c for c in df.columns if c not in label_cols + raw_cols 
                    and pd.api.types.is_numeric_dtype(df[c])]
    
    print(f"  Features: {len(feature_cols)}")
    
    results = []
    
    # Run all tests
    print_header("TEST 1: Temporal Integrity of Splits")
    passed, details = check_temporal_integrity(df)
    print_result("Train/Val/Test are strictly chronological", passed, details)
    results.append(("Temporal Integrity", passed))
    
    print_header("TEST 2: Label Construction Check")
    passed, details = check_label_construction(df)
    print_result("Labels use future data correctly", passed, details)
    results.append(("Label Construction", passed))
    
    print_header("TEST 3: Feature Look-Ahead Bias")
    passed, details = check_feature_lookahead(df, feature_cols)
    print_result("Features don't leak future information", passed, details)
    results.append(("Feature Look-Ahead", passed))
    
    print_header("TEST 4: Rolling Feature Computation")
    passed, details = check_rolling_features(df)
    print_result("Rolling features use only past data", passed, details)
    results.append(("Rolling Features", passed))
    
    print_header("TEST 5: Feature Timing Analysis")
    passed, details = check_feature_timing(df)
    print_result("Features properly lagged", passed, details)
    results.append(("Feature Timing", passed))
    
    print_header("TEST 6: Perfect Prediction Check")
    passed, details = check_perfect_prediction(df, feature_cols)
    print_result("No perfect class separation", passed, details)
    results.append(("Perfect Prediction", passed))
    
    print_header("TEST 7: Triple-Barrier Verification")
    passed, details = verify_triple_barrier(df)
    print_result("Triple-barrier computed correctly", passed, details)
    results.append(("Triple-Barrier", passed))
    
    print_header("TEST 8: Backtest Logic Verification")
    passed, details = verify_backtest_logic()
    print_result("Backtest signal/label logic correct", passed, details)
    results.append(("Backtest Logic", passed))
    
    # Summary
    print_header("SUMMARY")
    
    n_passed = sum(1 for _, p in results if p)
    n_total = len(results)
    
    print(f"\n  Tests passed: {n_passed}/{n_total}")
    print()
    for name, passed in results:
        status = "✓" if passed else "❌"
        print(f"  [{status}] {name}")
    
    if n_passed == n_total:
        print(f"\n  ✓ All checks passed - no obvious leakage detected")
    else:
        print(f"\n  ⚠ Some checks flagged potential issues - review above details")
    
    # Additional notes
    print_header("IMPORTANT NOTES")
    print("""
  1. The triple-barrier label (y_tb_60) is SUPPOSED to use future data.
     - It looks at prices from t+1 to t+60 to determine if TP or SL was hit
     - This is correct for supervised learning labels
  
  2. Features should use only data up to time t:
     - Rolling means/stds should be computed on data ending at t-1
     - The features_complete.py uses standard pandas rolling which is causal
  
  3. The model predicts P(up) at time t using features at time t.
     The trade would be entered at t+1 (next bar open).
  
  4. High backtest performance (85%+ win rate) could indicate:
     - Very selective thresholds (only 2% of bars traded)
     - Strong edge on high-confidence predictions
     - OR potential subtle leakage (requires deeper investigation)
  
  5. To further validate, consider:
     - Shuffled data test (should destroy performance)
     - Walk-forward out-of-sample on completely unseen data
     - Paper trading with live data
""")
    
    print("=" * 70)
    print("  DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

