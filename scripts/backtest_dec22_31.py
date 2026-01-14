#!/usr/bin/env python3
"""
Backtest Model 1 and Model 3 on December 22-31, 2025 data.

Tests both models on the last week of December 2025 using 15-minute validation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from typing import Dict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.regime_detection import add_regime_features, filter_short_signals_by_regime

# Model paths
MODEL1_PATH = PROJECT_ROOT / "models" / "y_tb_60_2014_2023.joblib"
MODEL3_PATH = PROJECT_ROOT / "models" / "model3_cmf_macd_2014_2023" / "model3_cmf_macd_2014_2023_15min_balanced.joblib"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"

# Test period (data only goes to Dec 22, 2025)
TEST_START = "2025-12-22"
TEST_END = "2025-12-22"  # Data only available until Dec 22

# Thresholds - will auto-adjust based on probability range
# Default thresholds (will be adjusted if needed)
MODEL1_THRESHOLDS = {"long": 0.55, "short": 0.45}  # Adjusted for single day
MODEL3_THRESHOLDS = {"long": 0.60, "short": 0.40}  # Lower thresholds for more trades


def run_backtest_15min(df: pd.DataFrame, proba_up: np.ndarray, threshold_long: float, threshold_short: float, model_name: str) -> Dict:
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
            "model": model_name,
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
        "model": model_name,
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


def test_model1(df_test: pd.DataFrame) -> Dict:
    """Test Model 1 on the test period."""
    print("\n" + "=" * 80)
    print("MODEL 1 TEST")
    print("=" * 80)
    
    # Load model
    if not MODEL1_PATH.exists():
        print(f"❌ Model 1 not found: {MODEL1_PATH}")
        return None
    
    artifact = joblib.load(MODEL1_PATH)
    model = artifact["model"]
    feature_cols = artifact["features"]
    
    print(f"\n✅ Loaded Model 1")
    print(f"   Features: {len(feature_cols)}")
    
    # Prepare data
    available_features = [f for f in feature_cols if f in df_test.columns]
    missing_features = [f for f in feature_cols if f not in df_test.columns]
    
    if missing_features:
        print(f"⚠️  Missing {len(missing_features)} features, filling with 0")
        # Fill missing features with 0
        for feat in missing_features:
            df_test[feat] = 0.0
        available_features = feature_cols  # Now all features are available
    
    # Check for y_tb_15 labels
    if 'y_tb_15' not in df_test.columns:
        print(f"\n[1.5] Generating y_tb_15 labels for 15-minute validation...")
        from src.features_complete import add_triple_barrier_labels
        df_test = add_triple_barrier_labels(df_test, h_max=15, tp_mult=1.0, sl_mult=1.0, horizons=[15])
        print(f"   ✓ Generated y_tb_15 labels")
    
    # Fill NaN values in features with 0
    df_test[available_features] = df_test[available_features].fillna(0)
    
    # Filter to rows with valid labels (y_tb_15 != 0 and not NaN)
    df_clean = df_test.dropna(subset=['y_tb_15'])
    df_clean = df_clean[df_clean['y_tb_15'] != 0]
    
    if len(df_clean) == 0:
        print("❌ No valid data after filtering")
        print(f"   Original rows: {len(df_test):,}")
        print(f"   After dropna y_tb_15: {len(df_test.dropna(subset=['y_tb_15'])):,}")
        if 'y_tb_15' in df_test.columns:
            y_dist = df_test['y_tb_15'].value_counts()
            print(f"   y_tb_15 distribution: {y_dist.to_dict()}")
            print(f"   y_tb_15 NaN count: {df_test['y_tb_15'].isnull().sum()}")
        return None
    
    print(f"   Valid rows: {len(df_clean):,}")
    
    # Check label distribution
    y_15_dist = df_clean['y_tb_15'].value_counts()
    print(f"\n   15-minute label distribution:")
    for val, count in y_15_dist.items():
        pct = 100 * count / len(df_clean)
        print(f"     {int(val):+d}: {count:,} ({pct:.1f}%)")
    
    # Get predictions
    X = df_clean[available_features].values
    proba = model.predict_proba(X)[:, 1]  # P(up)
    
    print(f"\n   Model predictions:")
    print(f"     Proba range: [{proba.min():.3f}, {proba.max():.3f}]")
    print(f"     Proba mean: {proba.mean():.3f}")
    print(f"     Proba < 0.5: {(proba < 0.5).sum():,} ({(proba < 0.5).sum()/len(proba)*100:.1f}%)")
    print(f"     Proba >= 0.5: {(proba >= 0.5).sum():,} ({(proba >= 0.5).sum()/len(proba)*100:.1f}%)")
    
    # Auto-adjust thresholds if they're outside the probability range
    proba_min, proba_max = proba.min(), proba.max()
    long_thresh = MODEL1_THRESHOLDS["long"]
    short_thresh = MODEL1_THRESHOLDS["short"]
    
    if long_thresh > proba_max:
        long_thresh = proba_max - 0.01
        print(f"   ⚠️  Adjusted long threshold from {MODEL1_THRESHOLDS['long']:.2f} to {long_thresh:.2f} (max proba is {proba_max:.3f})")
    if short_thresh < proba_min:
        short_thresh = proba_min + 0.01
        print(f"   ⚠️  Adjusted short threshold from {MODEL1_THRESHOLDS['short']:.2f} to {short_thresh:.2f} (min proba is {proba_min:.3f})")
    
    # Run backtest
    result = run_backtest_15min(
        df_clean, 
        proba, 
        long_thresh, 
        short_thresh,
        "Model 1"
    )
    
    return result


def test_model3(df_test: pd.DataFrame) -> Dict:
    """Test Model 3 on the test period."""
    print("\n" + "=" * 80)
    print("MODEL 3 TEST")
    print("=" * 80)
    
    # Load model
    if not MODEL3_PATH.exists():
        print(f"❌ Model 3 not found: {MODEL3_PATH}")
        return None
    
    artifact = joblib.load(MODEL3_PATH)
    model = artifact["model"]
    feature_cols = artifact["features"]
    
    print(f"\n✅ Loaded Model 3")
    print(f"   Features: {len(feature_cols)}")
    
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
    
    # Check for y_tb_15 labels
    if 'y_tb_15' not in df_test.columns:
        print(f"\n[1.5] Generating y_tb_15 labels for 15-minute validation...")
        from src.features_complete import add_triple_barrier_labels
        df_test = add_triple_barrier_labels(df_test, h_max=15, tp_mult=1.0, sl_mult=1.0, horizons=[15])
        print(f"   ✓ Generated y_tb_15 labels")
    
    df_clean = df_test.dropna(subset=available_features + ['y_tb_15'])
    df_clean = df_clean[df_clean['y_tb_15'] != 0]
    
    if len(df_clean) == 0:
        print("❌ No valid data after filtering")
        return None
    
    print(f"   Valid rows: {len(df_clean):,}")
    
    # Check label distribution
    y_15_dist = df_clean['y_tb_15'].value_counts()
    print(f"\n   15-minute label distribution:")
    for val, count in y_15_dist.items():
        pct = 100 * count / len(df_clean)
        print(f"     {int(val):+d}: {count:,} ({pct:.1f}%)")
    
    # Get predictions
    X = df_clean[available_features].values
    proba = model.predict_proba(X)[:, 1]  # P(up)
    
    print(f"\n   Model predictions:")
    print(f"     Proba range: [{proba.min():.3f}, {proba.max():.3f}]")
    print(f"     Proba mean: {proba.mean():.3f}")
    print(f"     Proba < 0.5: {(proba < 0.5).sum():,} ({(proba < 0.5).sum()/len(proba)*100:.1f}%)")
    print(f"     Proba >= 0.5: {(proba >= 0.5).sum():,} ({(proba >= 0.5).sum()/len(proba)*100:.1f}%)")
    
    # Run backtest
    result = run_backtest_15min(
        df_clean, 
        proba, 
        MODEL3_THRESHOLDS["long"], 
        MODEL3_THRESHOLDS["short"],
        "Model 3"
    )
    
    return result


def main():
    print("=" * 80)
    print("BACKTEST: DECEMBER 22-31, 2025")
    print("=" * 80)
    print(f"\nTest Period: {TEST_START} to {TEST_END}")
    print(f"Validation: 15-minute triple-barrier labels (y_tb_15)")
    
    # Load data
    print(f"\n[1] Loading features data...")
    if not FEATURES_PATH.exists():
        print(f"❌ Features file not found: {FEATURES_PATH}")
        return
    
    df = pd.read_parquet(FEATURES_PATH)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    # Filter to test period (include all of Dec 22 if TEST_END is same as TEST_START)
    if TEST_END == TEST_START:
        # Include all of the day
        df_test = df[df.index.date >= pd.to_datetime(TEST_START).date()].copy()
        df_test = df_test[df_test.index.date <= pd.to_datetime(TEST_END).date()].copy()
    else:
        df_test = df[(df.index >= TEST_START) & (df.index <= TEST_END)].copy()
    print(f"   Test period rows: {len(df_test):,}")
    print(f"   Date range: {df_test.index.min().date()} to {df_test.index.max().date()}")
    
    if len(df_test) == 0:
        print(f"❌ No data found for period {TEST_START} to {TEST_END}")
        return
    
    # Test Model 1
    result1 = test_model1(df_test.copy())
    
    # Test Model 3
    result3 = test_model3(df_test.copy())
    
    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    if result1 and result3:
        print(f"\n{'Metric':<20} {'Model 1':<20} {'Model 3':<20}")
        print("-" * 60)
        print(f"{'Trades':<20} {result1['n_trades']:<20,} {result3['n_trades']:<20,}")
        print(f"{'Long Trades':<20} {result1['n_long']:<20,} {result3['n_long']:<20,}")
        print(f"{'Short Trades':<20} {result1['n_short']:<20,} {result3['n_short']:<20,}")
        print(f"{'Win Rate':<20} {result1['win_rate']*100:<19.1f}% {result3['win_rate']*100:<19.1f}%")
        print(f"{'Long Win Rate':<20} {result1['long_wr']*100:<19.1f}% {result3['long_wr']*100:<19.1f}%")
        print(f"{'Short Win Rate':<20} {result1['short_wr']*100:<19.1f}% {result3['short_wr']*100:<19.1f}%")
        print(f"{'Avg R/trade':<20} {result1['avg_r']:<20.4f} {result3['avg_r']:<20.4f}")
        print(f"{'Cumulative R':<20} {result1['cum_r']:<20.1f} {result3['cum_r']:<20.1f}")
        print(f"{'Sharpe Ratio':<20} {result1['sharpe']:<20.2f} {result3['sharpe']:<20.2f}")
        
        # Determine winner
        print(f"\n{'='*80}")
        print("WINNER")
        print(f"{'='*80}")
        
        if result1['sharpe'] > result3['sharpe']:
            print(f"✅ Model 1 wins with Sharpe {result1['sharpe']:.2f} vs {result3['sharpe']:.2f}")
        elif result3['sharpe'] > result1['sharpe']:
            print(f"✅ Model 3 wins with Sharpe {result3['sharpe']:.2f} vs {result1['sharpe']:.2f}")
        else:
            print("Tie")
    
    elif result1:
        print("\n✅ Model 1 Results:")
        print(f"   Trades: {result1['n_trades']:,}")
        print(f"   Win Rate: {result1['win_rate']*100:.1f}%")
        print(f"   Sharpe: {result1['sharpe']:.2f}")
    
    elif result3:
        print("\n✅ Model 3 Results:")
        print(f"   Trades: {result3['n_trades']:,}")
        print(f"   Win Rate: {result3['win_rate']*100:.1f}%")
        print(f"   Sharpe: {result3['sharpe']:.2f}")
    
    print(f"\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()

