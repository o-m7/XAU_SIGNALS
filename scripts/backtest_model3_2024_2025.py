#!/usr/bin/env python3
"""
Out-of-Sample Backtest for New Model 3 (2020-2023 trained) on 2024-2025 data.

Tests the new Model 3 with regime detection on 2024-2025 out-of-sample data.
Uses 15-minute validation (y_tb_15 labels) since trades last 15 minutes or less.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.regime_detection import add_regime_features, filter_short_signals_by_regime

# Model path (using the new balanced 15-minute model)
MODEL_PATH = PROJECT_ROOT / "models" / "model3_cmf_macd_2014_2023" / "model3_cmf_macd_2014_2023_15min_balanced.joblib"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"

# Test period
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

# Thresholds to test
LONG_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
SHORT_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]


def run_backtest_15min(df: pd.DataFrame, proba_up: np.ndarray, threshold_long: float, threshold_short: float) -> Dict:
    """
    Run 15-minute validation backtest.
    
    Each trade is validated over the next 15 bars (15 minutes).
    Uses y_tb_15 labels for validation.
    """
    # Generate signals
    signal = np.zeros(len(proba_up), dtype=int)
    signal[proba_up >= threshold_long] = 1   # LONG
    signal[proba_up <= threshold_short] = -1  # SHORT
    
    # Get 15-minute labels
    y_15 = df['y_tb_15'].values
    y_15 = np.where(y_15 == 1, 1, -1)  # Map to +1/-1
    
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
    
    y_trades = y_15[trade_mask]
    signal_trades = signal[trade_mask]
    
    # Trade returns: +1R if signal matches label, -1R otherwise
    trade_ret = y_trades * signal_trades
    
    win_rate = float((trade_ret > 0).mean())
    avg_r = float(trade_ret.mean())
    cum_r = float(trade_ret.sum())
    std_r = float(trade_ret.std())
    sharpe = float(avg_r / (std_r + 1e-8) * np.sqrt(252 * 24 * 60))
    
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


def main():
    print("=" * 80)
    print("OUT-OF-SAMPLE BACKTEST: NEW MODEL 3 (2020-2023 trained) on 2024-2025")
    print("Using 15-minute validation (y_tb_15 labels)")
    print("=" * 80)
    
    # Load model
    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["features"]
    
    print(f"\n✅ Loaded Model 3 (trained on {artifact.get('trained_at', 'unknown')})")
    print(f"   Features: {len(feature_cols)}")
    
    # Load data
    print(f"\n[1] Loading features data...")
    if not FEATURES_PATH.exists():
        print(f"❌ Features file not found: {FEATURES_PATH}")
        return
    
    df = pd.read_parquet(FEATURES_PATH)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    # Filter to test period
    df_test = df[(df.index >= TEST_START) & (df.index <= TEST_END)].copy()
    print(f"   Test period rows: {len(df_test):,}")
    print(f"   Date range: {df_test.index.min().date()} to {df_test.index.max().date()}")
    
    if len(df_test) == 0:
        print(f"❌ No data found for period {TEST_START} to {TEST_END}")
        return
    
    # Build Model 3 features (CMF/MACD)
    print(f"\n[1.2] Building Model 3 features (CMF/MACD)...")
    from src.model3_cmf_macd.features import build_cmf_macd_features
    df_test = build_cmf_macd_features(df_test)
    print(f"   ✓ CMF/MACD features built")
    
    # Prepare data
    available_features = [f for f in feature_cols if f in df_test.columns]
    missing_features = [f for f in feature_cols if f not in df_test.columns]
    
    if missing_features:
        print(f"⚠️  Missing {len(missing_features)} features, using {len(available_features)} available")
        if len(missing_features) <= 10:
            print(f"   Missing: {missing_features}")
    
    # Check for y_tb_15 labels, generate if missing
    if 'y_tb_15' not in df_test.columns:
        print(f"\n[1.5] Generating y_tb_15 labels for 15-minute validation...")
        from src.features_complete import add_triple_barrier_labels
        df_test = add_triple_barrier_labels(df_test, h_max=15, tp_mult=1.0, sl_mult=1.0, horizons=[15])
        print(f"   ✓ Generated y_tb_15 labels")
    
    df_clean = df_test.dropna(subset=available_features + ['y_tb_15'])
    df_clean = df_clean[df_clean['y_tb_15'] != 0]
    
    if len(df_clean) == 0:
        print("❌ No valid data after filtering")
        return
    
    print(f"   Valid rows: {len(df_clean):,}")
    
    # Check label distribution
    y_15_dist = df_clean['y_tb_15'].value_counts()
    print(f"\n   15-minute label distribution:")
    for val, count in y_15_dist.items():
        pct = 100 * count / len(df_clean)
        print(f"     {int(val):+d}: {count:,} ({pct:.1f}%)")
    
    # Get predictions
    X = df_clean[available_features].values
    proba = model.predict_proba(X)[:, 1]  # P(up) - Model 3 predicts probability of class 1 (which is mapped from +1)
    
    print(f"\n   Model predictions:")
    print(f"     Proba range: [{proba.min():.3f}, {proba.max():.3f}]")
    print(f"     Proba mean: {proba.mean():.3f}")
    print(f"     Proba < 0.5: {(proba < 0.5).sum():,} ({(proba < 0.5).sum()/len(proba)*100:.1f}%)")
    print(f"     Proba >= 0.5: {(proba >= 0.5).sum():,} ({(proba >= 0.5).sum()/len(proba)*100:.1f}%)")
    
    # Test all threshold combinations using 15-minute validation
    print(f"\n[2] Testing {len(LONG_THRESHOLDS)} x {len(SHORT_THRESHOLDS)} threshold combinations...")
    print(f"    Using 15-minute validation (y_tb_15 labels)")
    results = []
    
    for tl in LONG_THRESHOLDS:
        for ts in SHORT_THRESHOLDS:
            if ts >= tl:
                continue
            res = run_backtest_15min(df_clean, proba, tl, ts)
            if res["n_trades"] >= 50:
                results.append(res)
    
    # Sort by Sharpe
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    
    # Print results
    if not results:
        print("❌ No valid threshold combinations found")
        return
    
    print(f"\n" + "=" * 100)
    print("TOP 20 THRESHOLD COMBINATIONS (Sorted by Sharpe)")
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
    
    # Best configuration
    best = results[0]
    print(f"\n" + "=" * 100)
    print("BEST THRESHOLD CONFIGURATION")
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
    
    # Apply regime detection
    print(f"\n" + "=" * 100)
    print("APPLYING REGIME DETECTION")
    print("=" * 100)
    
    # Generate signals with best thresholds
    signals = pd.Series(0, index=df_clean.index)
    signals[proba >= best['threshold_long']] = 1   # LONG
    signals[proba <= best['threshold_short']] = -1   # SHORT
    
    print(f"\n  Before regime filtering:")
    print(f"    Total signals: {len(signals[signals != 0]):,}")
    print(f"    Long signals:  {(signals == 1).sum():,}")
    print(f"    Short signals: {(signals == -1).sum():,}")
    
    # Apply regime filtering
    df_clean_with_regime = add_regime_features(df_clean.copy())
    signals_filtered = filter_short_signals_by_regime(
        signals,
        df_clean_with_regime,
        regime_threshold=0.6
    )
    
    print(f"\n  After regime filtering:")
    print(f"    Total signals: {len(signals_filtered[signals_filtered != 0]):,}")
    print(f"    Long signals:  {(signals_filtered == 1).sum():,}")
    print(f"    Short signals: {(signals_filtered == -1).sum():,}")
    print(f"    Removed shorts: {(signals == -1).sum() - (signals_filtered == -1).sum():,}")
    
    # Re-run backtest with regime-filtered signals
    if (signals_filtered == -1).sum() > 0 or (signals_filtered == 1).sum() > 0:
        print(f"\n  Re-running backtest with regime-filtered signals...")
        y_15_filtered = df_clean['y_tb_15'].values
        y_15_filtered = np.where(y_15_filtered == 1, 1, -1)
        
        trade_mask_filtered = signals_filtered != 0
        y_filtered = y_15_filtered[trade_mask_filtered]
        signals_filtered_trades = signals_filtered[trade_mask_filtered].values
        
        trade_ret_filtered = y_filtered * signals_filtered_trades
        win_rate_filtered = float((trade_ret_filtered > 0).mean())
        avg_r_filtered = float(trade_ret_filtered.mean())
        cum_r_filtered = float(trade_ret_filtered.sum())
        
        print(f"\n  Regime-Filtered Results (15-minute validation):")
        print(f"    Trades: {len(trade_ret_filtered):,}")
        print(f"    Win Rate: {win_rate_filtered*100:.1f}%")
        print(f"    Avg R/trade: {avg_r_filtered:+.4f}")
        print(f"    Cumulative R: {cum_r_filtered:+.1f}")
    
    print(f"\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()

