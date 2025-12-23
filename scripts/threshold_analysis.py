#!/usr/bin/env python3
"""
Threshold Analysis - Out of Sample Backtest

Tests different threshold combinations on the test set and ranks them.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "y_tb_60_hgb_tuned.joblib"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2024.parquet"

# Thresholds to test
LONG_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
SHORT_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

TARGET = "y_tb_60"

def run_detailed_backtest(y, proba_up, tl, ts):
    """
    Run backtest with detailed stats.
    
    Returns dict with all metrics including long/short breakdown.
    """
    signal = np.zeros_like(proba_up, dtype=int)
    signal[proba_up >= tl] = 1   # LONG
    signal[proba_up <= ts] = -1  # SHORT
    
    # Filter to trades only
    trade_mask = signal != 0
    n_trades = int(trade_mask.sum())
    
    if n_trades == 0:
        return {
            "tl": tl, "ts": ts, "n_trades": 0,
            "n_long": 0, "n_short": 0,
            "win_rate": 0, "long_wr": 0, "short_wr": 0,
            "avg_r": 0, "cum_r": 0, "sharpe": 0
        }
    
    y_trades = y[trade_mask]
    signal_trades = signal[trade_mask]
    
    # Trade returns: +1R if signal matches label, -1R otherwise
    trade_ret = y_trades * signal_trades  # +1 or -1
    
    # Overall stats
    win_rate = float((trade_ret > 0).mean())
    avg_r = float(trade_ret.mean())
    cum_r = float(trade_ret.sum())
    sharpe = float(avg_r / (trade_ret.std() + 1e-8) * np.sqrt(252))
    
    # Long/Short breakdown
    long_mask = signal_trades == 1
    short_mask = signal_trades == -1
    
    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())
    
    long_wr = float((trade_ret[long_mask] > 0).mean()) if n_long > 0 else 0
    short_wr = float((trade_ret[short_mask] > 0).mean()) if n_short > 0 else 0
    
    return {
        "tl": tl, "ts": ts, "n_trades": n_trades,
        "n_long": n_long, "n_short": n_short,
        "win_rate": win_rate, "long_wr": long_wr, "short_wr": short_wr,
        "avg_r": avg_r, "cum_r": cum_r, "sharpe": sharpe
    }


def main():
    print("=" * 80)
    print("THRESHOLD ANALYSIS - OUT OF SAMPLE BACKTEST")
    print("=" * 80)
    
    # Load model
    print(f"\n1. Loading model: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["features"]
    print(f"   Features: {len(feature_cols)}")
    
    # Load features
    print(f"\n2. Loading features: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    print(f"   Total rows: {len(df):,}")
    
    # Filter valid rows
    label_cols = ["y_ret_5", "y_dir_5", "y_ret_15", "y_dir_15", "y_ret_60", "y_dir_60", "y_tb_60"]
    meta_cols = ["timestamp", "symbol", "open", "high", "low", "close", "volume", "bid_price", "ask_price"]
    
    # Get target and filter
    mask = df[TARGET].notna() & (df[TARGET] != 0)
    df_valid = df[mask].copy()
    print(f"   Valid rows (y_tb_60 != 0): {len(df_valid):,}")
    
    # Prepare X, y
    X = df_valid[feature_cols].values
    y = df_valid[TARGET].values
    y = np.where(y == 1, 1, -1)  # Ensure +1/-1 format
    
    # Time-based split (use TEST set for out-of-sample)
    n = len(X)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    X_test = X[val_end:]
    y_test = y[val_end:]
    
    print(f"\n3. Data split:")
    print(f"   Train: {train_end:,} samples")
    print(f"   Val: {val_end - train_end:,} samples")
    print(f"   Test: {len(X_test):,} samples (OUT OF SAMPLE)")
    
    # Get predictions on TEST set
    print(f"\n4. Generating predictions on test set...")
    proba_test = model.predict_proba(X_test)[:, 1]
    print(f"   Proba range: [{proba_test.min():.3f}, {proba_test.max():.3f}]")
    print(f"   Proba mean: {proba_test.mean():.3f}")
    
    # Test all threshold combinations
    print(f"\n5. Testing threshold combinations...")
    print(f"   Long thresholds: {LONG_THRESHOLDS}")
    print(f"   Short thresholds: {SHORT_THRESHOLDS}")
    
    results = []
    for tl in LONG_THRESHOLDS:
        for ts in SHORT_THRESHOLDS:
            if ts >= tl:  # Skip invalid (overlapping)
                continue
            res = run_detailed_backtest(y_test, proba_test, tl, ts)
            if res["n_trades"] >= 50:  # Min trades filter
                results.append(res)
    
    # Sort by Sharpe
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    
    # Print results table
    print(f"\n" + "=" * 100)
    print("OUT-OF-SAMPLE BACKTEST RESULTS (Sorted by Sharpe)")
    print("=" * 100)
    print(f"\n{'Rank':<5} {'Long':<6} {'Short':<6} {'Trades':<8} {'Longs':<7} {'Shorts':<7} "
          f"{'Win%':<7} {'Long%':<7} {'Short%':<7} {'AvgR':<8} {'CumR':<9} {'Sharpe':<8}")
    print("-" * 100)
    
    for i, r in enumerate(results[:20]):
        marker = " ***" if i == 0 else ""
        print(f"{i+1:<5} {r['tl']:<6.2f} {r['ts']:<6.2f} {r['n_trades']:<8,} "
              f"{r['n_long']:<7,} {r['n_short']:<7,} "
              f"{r['win_rate']*100:<6.1f}% {r['long_wr']*100:<6.1f}% {r['short_wr']*100:<6.1f}% "
              f"{r['avg_r']:>+7.4f} {r['cum_r']:>+8.1f} {r['sharpe']:>7.2f}{marker}")
    
    # Summary
    if results:
        best = results[0]
        print(f"\n" + "=" * 100)
        print("BEST THRESHOLD CONFIGURATION")
        print("=" * 100)
        print(f"\n  Long Threshold:  {best['tl']:.2f}  (P(up) >= {best['tl']})")
        print(f"  Short Threshold: {best['ts']:.2f}  (P(up) <= {best['ts']})")
        print(f"\n  Total Trades:    {best['n_trades']:,}")
        print(f"    - Long trades:   {best['n_long']:,} ({best['n_long']/best['n_trades']*100:.1f}%)")
        print(f"    - Short trades:  {best['n_short']:,} ({best['n_short']/best['n_trades']*100:.1f}%)")
        print(f"\n  Win Rate:        {best['win_rate']*100:.1f}%")
        print(f"    - Long win%:     {best['long_wr']*100:.1f}%")
        print(f"    - Short win%:    {best['short_wr']*100:.1f}%")
        print(f"\n  Performance:")
        print(f"    - Avg R/trade:   {best['avg_r']:+.4f}")
        print(f"    - Cumulative R:  {best['cum_r']:+.1f}")
        print(f"    - Sharpe Ratio:  {best['sharpe']:.2f}")
        
        # Compare 0.60/0.40 vs 0.65/0.35 vs 0.75/0.25 specifically
        print(f"\n" + "=" * 100)
        print("COMPARISON: COMMON THRESHOLD PAIRS")
        print("=" * 100)
        
        pairs_to_compare = [
            (0.60, 0.40), (0.60, 0.35), (0.60, 0.30),
            (0.65, 0.40), (0.65, 0.35), (0.65, 0.30),
            (0.70, 0.35), (0.70, 0.30), (0.70, 0.25),
            (0.75, 0.30), (0.75, 0.25), (0.75, 0.20),
        ]
        
        print(f"\n{'Long':<6} {'Short':<6} {'Trades':<8} {'Win%':<7} {'Long%':<7} {'Short%':<7} {'CumR':<9} {'Sharpe':<8}")
        print("-" * 70)
        
        for tl, ts in pairs_to_compare:
            r = run_detailed_backtest(y_test, proba_test, tl, ts)
            if r["n_trades"] > 0:
                print(f"{tl:<6.2f} {ts:<6.2f} {r['n_trades']:<8,} "
                      f"{r['win_rate']*100:<6.1f}% {r['long_wr']*100:<6.1f}% {r['short_wr']*100:<6.1f}% "
                      f"{r['cum_r']:>+8.1f} {r['sharpe']:>7.2f}")
    
    print(f"\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()

