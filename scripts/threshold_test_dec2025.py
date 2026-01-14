#!/usr/bin/env python3
"""
Threshold Test on Out-of-Sample Data: December 2025 to Today

Tests different threshold combinations for Model 1 and Model 3
on December 2025 out-of-sample data.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Model paths
MODEL1_PATH = PROJECT_ROOT / "models" / "y_tb_60_hgb_tuned.joblib"
MODEL3_PATH = PROJECT_ROOT / "models" / "model3_cmf_macd" / "model3_cmf_macd.joblib"

# Data path
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"

# Thresholds to test - focusing on 0.60/0.30 configuration
LONG_THRESHOLDS = [0.60]  # Test Long=0.6
SHORT_THRESHOLDS = [0.30]  # Test Short=0.3

# Test period: December 2025 onwards
TEST_START = "2025-12-01"


def run_backtest(y: np.ndarray, proba_up: np.ndarray, threshold_long: float, threshold_short: float) -> Dict:
    """
    Run simple R-based backtest.
    
    Returns dict with metrics.
    """
    # Generate signals
    signal = np.zeros_like(proba_up, dtype=int)
    signal[proba_up >= threshold_long] = 1   # LONG
    signal[proba_up <= threshold_short] = -1  # SHORT
    
    # Filter to trades only
    trade_mask = signal != 0
    n_trades = int(trade_mask.sum())
    
    if n_trades == 0:
        return {
            "threshold_long": threshold_long,
            "threshold_short": threshold_short,
            "n_trades": 0,
            "n_long": 0,
            "n_short": 0,
            "win_rate": 0,
            "long_wr": 0,
            "short_wr": 0,
            "avg_r": 0,
            "cum_r": 0,
            "sharpe": 0,
            "long_pct": 0,
            "short_pct": 0
        }
    
    y_trades = y[trade_mask]
    signal_trades = signal[trade_mask]
    
    # Trade returns: +1R if signal matches label, -1R otherwise
    trade_ret = y_trades * signal_trades  # +1 or -1
    
    # Overall stats
    win_rate = float((trade_ret > 0).mean())
    avg_r = float(trade_ret.mean())
    cum_r = float(trade_ret.sum())
    std_r = float(trade_ret.std())
    sharpe = float(avg_r / (std_r + 1e-8) * np.sqrt(252 * 24 * 60))  # Annualized for minute data
    
    # Long/Short breakdown
    long_mask = signal_trades == 1
    short_mask = signal_trades == -1
    
    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())
    
    long_wr = float((trade_ret[long_mask] > 0).mean()) if n_long > 0 else 0
    short_wr = float((trade_ret[short_mask] > 0).mean()) if n_short > 0 else 0
    
    return {
        "threshold_long": threshold_long,
        "threshold_short": threshold_short,
        "n_trades": n_trades,
        "n_long": n_long,
        "n_short": n_short,
        "win_rate": win_rate,
        "long_wr": long_wr,
        "short_wr": short_wr,
        "avg_r": avg_r,
        "cum_r": cum_r,
        "sharpe": sharpe,
        "long_pct": n_long / n_trades * 100 if n_trades > 0 else 0,
        "short_pct": n_short / n_trades * 100 if n_trades > 0 else 0
    }


def test_model1(df_test: pd.DataFrame) -> List[Dict]:
    """Test Model 1 (y_tb_60) on December 2025 data."""
    print("\n" + "=" * 80)
    print("MODEL 1: y_tb_60_hgb_tuned")
    print("=" * 80)
    
    # Load model
    if not MODEL1_PATH.exists():
        print(f"❌ Model 1 not found: {MODEL1_PATH}")
        return []
    
    artifact = joblib.load(MODEL1_PATH)
    model = artifact["model"]
    feature_cols = artifact["features"]
    
    print(f"✅ Loaded Model 1")
    print(f"   Features: {len(feature_cols)}")
    
    # Prepare data
    available_features = [f for f in feature_cols if f in df_test.columns]
    missing_features = [f for f in feature_cols if f not in df_test.columns]
    
    if missing_features:
        print(f"⚠️  Missing {len(missing_features)} features, using {len(available_features)} available")
    
    df_clean = df_test.dropna(subset=available_features + ['y_tb_60'])
    df_clean = df_clean[df_clean['y_tb_60'] != 0]
    
    if len(df_clean) == 0:
        print("❌ No valid data after filtering")
        return []
    
    print(f"   Valid rows: {len(df_clean):,}")
    print(f"   Date range: {df_clean.index.min().date()} to {df_clean.index.max().date()}")
    
    # Get predictions
    X = df_clean[available_features].values
    y = df_clean['y_tb_60'].values
    y = np.where(y == 1, 1, -1)  # Ensure +1/-1 format
    
    proba = model.predict_proba(X)[:, 1]  # P(up)
    
    print(f"   Proba range: [{proba.min():.3f}, {proba.max():.3f}]")
    print(f"   Proba mean: {proba.mean():.3f}")
    
    # Test all threshold combinations
    print(f"\n   Testing {len(LONG_THRESHOLDS)} x {len(SHORT_THRESHOLDS)} threshold combinations...")
    results = []
    
    for tl in LONG_THRESHOLDS:
        for ts in SHORT_THRESHOLDS:
            if ts >= tl:  # Skip invalid (overlapping)
                continue
            res = run_backtest(y, proba, tl, ts)
            if res["n_trades"] >= 50:  # Min trades filter
                results.append(res)
    
    # Sort by Sharpe
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    
    return results


def test_model3(df_test: pd.DataFrame) -> List[Dict]:
    """Test Model 3 (CMF/MACD) on December 2025 data."""
    print("\n" + "=" * 80)
    print("MODEL 3: CMF/MACD Classifier")
    print("=" * 80)
    
    # Load model
    if not MODEL3_PATH.exists():
        print(f"❌ Model 3 not found: {MODEL3_PATH}")
        return []
    
    artifact = joblib.load(MODEL3_PATH)
    model = artifact["model"]
    feature_cols = artifact["features"]
    
    print(f"✅ Loaded Model 3")
    print(f"   Features: {len(feature_cols)}")
    
    # Import Model 3 feature building
    try:
        from src.model3_cmf_macd.features import build_cmf_macd_features, get_feature_columns_for_model3
        from src.model3_cmf_macd.labeling import add_triple_barrier_labels
        
        # Build CMF/MACD features
        df_with_features = build_cmf_macd_features(df_test.copy())
        
        # Add labels
        df_with_features = add_triple_barrier_labels(df_with_features, h_max=60, tp_mult=1.0, sl_mult=1.0)
        
        # Get feature columns
        model3_feature_cols = get_feature_columns_for_model3()
        available_features = [f for f in model3_feature_cols if f in df_with_features.columns]
        
        df_clean = df_with_features.dropna(subset=available_features + ['y_tb_60'])
        df_clean = df_clean[df_with_features['y_tb_60'] != 0]
        
    except ImportError as e:
        print(f"❌ Error importing Model 3 modules: {e}")
        return []
    
    if len(df_clean) == 0:
        print("❌ No valid data after filtering")
        return []
    
    print(f"   Valid rows: {len(df_clean):,}")
    print(f"   Date range: {df_clean.index.min().date()} to {df_clean.index.max().date()}")
    
    # Get predictions
    X = df_clean[available_features].values
    y = df_clean['y_tb_60'].values
    y = np.where(y == 1, 1, -1)  # Ensure +1/-1 format
    
    proba = model.predict_proba(X)[:, 1]  # P(up)
    
    print(f"   Proba range: [{proba.min():.3f}, {proba.max():.3f}]")
    print(f"   Proba mean: {proba.mean():.3f}")
    
    # Test all threshold combinations
    print(f"\n   Testing {len(LONG_THRESHOLDS)} x {len(SHORT_THRESHOLDS)} threshold combinations...")
    results = []
    
    for tl in LONG_THRESHOLDS:
        for ts in SHORT_THRESHOLDS:
            if ts >= tl:  # Skip invalid (overlapping)
                continue
            res = run_backtest(y, proba, tl, ts)
            if res["n_trades"] >= 50:  # Min trades filter
                results.append(res)
    
    # Sort by Sharpe
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    
    return results


def print_results_table(results: List[Dict], model_name: str):
    """Print formatted results table."""
    if not results:
        print(f"\n❌ No valid results for {model_name}")
        return
    
    print(f"\n" + "=" * 100)
    print(f"{model_name} - TOP 20 THRESHOLD COMBINATIONS (Sorted by Sharpe)")
    print("=" * 100)
    print(f"\n{'Rank':<5} {'Long':<6} {'Short':<6} {'Trades':<8} {'Longs':<7} {'Shorts':<7} "
          f"{'Win%':<7} {'Long%':<7} {'Short%':<7} {'AvgR':<8} {'CumR':<9} {'Sharpe':<8}")
    print("-" * 100)
    
    for i, r in enumerate(results[:20]):
        marker = " ***" if i == 0 else ""
        print(f"{i+1:<5} {r['threshold_long']:<6.2f} {r['threshold_short']:<6.2f} {r['n_trades']:<8,} "
              f"{r['n_long']:<7,} {r['n_short']:<7,} "
              f"{r['win_rate']*100:<6.1f}% {r['long_wr']*100:<6.1f}% {r['short_wr']*100:<6.1f}% "
              f"{r['avg_r']:>+7.4f} {r['cum_r']:>+8.1f} {r['sharpe']:>7.2f}{marker}")
    
    # Best configuration summary
    best = results[0]
    print(f"\n" + "=" * 100)
    print(f"{model_name} - BEST THRESHOLD CONFIGURATION")
    print("=" * 100)
    print(f"\n  Long Threshold:  {best['threshold_long']:.2f}  (P(up) >= {best['threshold_long']})")
    print(f"  Short Threshold: {best['threshold_short']:.2f}  (P(up) <= {best['threshold_short']})")
    print(f"\n  Total Trades:    {best['n_trades']:,}")
    print(f"    - Long trades:   {best['n_long']:,} ({best['long_pct']:.1f}%)")
    print(f"    - Short trades:  {best['n_short']:,} ({best['short_pct']:.1f}%)")
    print(f"\n  Win Rate:        {best['win_rate']*100:.1f}%")
    print(f"    - Long win%:     {best['long_wr']*100:.1f}%")
    print(f"    - Short win%:    {best['short_wr']*100:.1f}%")
    print(f"\n  Performance:")
    print(f"    - Avg R/trade:   {best['avg_r']:+.4f}")
    print(f"    - Cumulative R:  {best['cum_r']:+.1f}")
    print(f"    - Sharpe Ratio:  {best['sharpe']:.2f}")


def main():
    print("=" * 80)
    print("THRESHOLD TEST: DECEMBER 2025 OUT-OF-SAMPLE DATA")
    print("=" * 80)
    print(f"\nTest Period: {TEST_START} onwards")
    print(f"Thresholds to test:")
    print(f"  Long:  {LONG_THRESHOLDS}")
    print(f"  Short: {SHORT_THRESHOLDS}")
    
    # Load data
    print(f"\n[1] Loading features data...")
    if not FEATURES_PATH.exists():
        print(f"❌ Features file not found: {FEATURES_PATH}")
        return
    
    df = pd.read_parquet(FEATURES_PATH)
    print(f"   Total rows: {len(df):,}")
    
    # Filter to December 2025 onwards
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    df_test = df[df.index >= TEST_START].copy()
    print(f"   Test period rows: {len(df_test):,}")
    
    if len(df_test) == 0:
        print(f"❌ No data found for period >= {TEST_START}")
        print(f"   Available date range: {df.index.min().date()} to {df.index.max().date()}")
        return
    
    print(f"   Date range: {df_test.index.min().date()} to {df_test.index.max().date()}")
    
    # Test Model 1
    results_model1 = test_model1(df_test)
    print_results_table(results_model1, "MODEL 1")
    
    # Test Model 3
    results_model3 = test_model3(df_test)
    print_results_table(results_model3, "MODEL 3")
    
    # Comparison
    if results_model1 and results_model3:
        print(f"\n" + "=" * 100)
        print("COMPARISON: BEST THRESHOLDS FOR EACH MODEL")
        print("=" * 100)
        
        best1 = results_model1[0]
        best3 = results_model3[0]
        
        print(f"\n{'Model':<10} {'Long':<6} {'Short':<6} {'Trades':<8} {'Win%':<7} {'CumR':<9} {'Sharpe':<8}")
        print("-" * 60)
        print(f"{'Model 1':<10} {best1['threshold_long']:<6.2f} {best1['threshold_short']:<6.2f} "
              f"{best1['n_trades']:<8,} {best1['win_rate']*100:<6.1f}% "
              f"{best1['cum_r']:>+8.1f} {best1['sharpe']:>7.2f}")
        print(f"{'Model 3':<10} {best3['threshold_long']:<6.2f} {best3['threshold_short']:<6.2f} "
              f"{best3['n_trades']:<8,} {best3['win_rate']*100:<6.1f}% "
              f"{best3['cum_r']:>+8.1f} {best3['sharpe']:>7.2f}")
    
    print(f"\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()

