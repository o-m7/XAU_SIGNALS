#!/usr/bin/env python3
"""
HONEST BACKTEST - Using LIVE Thresholds

Rerun backtests for all 3 production models using:
1. Actual LIVE thresholds currently deployed
2. 2025 OOS data (out-of-sample)
3. Realistic costs and slippage
4. Prediction distribution analysis

This will show if backtest performance matches live reality.
"""

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import backtest engine
sys.path.insert(0, str(Path(__file__).parent))
from scripts.backtest_engine_v2 import run_backtest, print_detailed_results

# ============================================================================
# CONFIGURATION - LIVE THRESHOLDS (as deployed 2026-01-12)
# ============================================================================

MODELS = {
    'model3_cmf_macd_v4': {
        'path': 'models/model3_cmf_macd_v4.joblib',
        'threshold_long': 0.80,   # LIVE threshold
        'threshold_short': 0.20,  # LIVE threshold
        'expected_wr': 0.617,
        'expected_trades_per_day': 16.8,
    },
    'model1_high_conf': {
        'path': 'models/model1_high_conf.joblib',
        'threshold_long': 0.52,   # LIVE threshold
        'threshold_short': 0.48,  # LIVE threshold
        'expected_wr': 0.613,
        'expected_trades_per_day': 3.1,
    },
    'model_rf_v4': {
        'path': 'models/model_rf_v4.joblib',
        'threshold_long': 0.65,   # LIVE threshold
        'threshold_short': 0.35,  # LIVE threshold
        'expected_wr': 0.577,
        'expected_trades_per_day': 7.0,
    }
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_oos_data():
    """Load 2025 out-of-sample data for testing."""
    print("\n" + "="*80)
    print("LOADING 2025 OOS DATA")
    print("="*80)

    # Try parquet files first (have features)
    data_paths = [
        Path('data/features/xauusd_features_2020_2025_fixed.parquet'),
        Path('data/features/xauusd_features_2020_2025.parquet'),
        Path('data/features/xauusd_features_realistic.parquet'),
        Path('data/features/xauusd_features_latest.parquet'),
    ]

    df = None
    for path in data_paths:
        if path.exists():
            print(f"\n✓ Found data: {path}")
            df = pd.read_parquet(path)

            # Ensure timestamp is index
            if 'timestamp' in df.columns and df.index.name != 'timestamp':
                df = df.set_index('timestamp')

            # Filter to 2025 only
            df_2025 = df[df.index.year == 2025]
            if len(df_2025) > 0:
                print(f"  2025 data: {len(df_2025):,} rows from {df_2025.index[0]} to {df_2025.index[-1]}")
                return df_2025
            else:
                # Use last year of available data
                df_recent = df[df.index.year == df.index.year.max()]
                print(f"  Using {df.index.year.max()} data: {len(df_recent):,} rows")
                return df_recent

    raise FileNotFoundError("No data files found! Check data/features/ directory")


# ============================================================================
# PREDICTION ANALYSIS
# ============================================================================

def analyze_predictions(model_name, predictions, threshold_long, threshold_short):
    """Analyze prediction distribution and signal generation."""
    print(f"\n{model_name} - Prediction Analysis")
    print("-" * 60)

    # Distribution stats
    print(f"  Mean:   {predictions.mean():.4f}")
    print(f"  Median: {predictions.median():.4f}")
    print(f"  Std:    {predictions.std():.4f}")
    print(f"  Min:    {predictions.min():.4f}")
    print(f"  Max:    {predictions.max():.4f}")

    # Percentiles
    print(f"\n  Percentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        val = predictions.quantile(p/100)
        print(f"    {p:2d}%: {val:.4f}")

    # Signal distribution with LIVE thresholds
    signals_long = (predictions >= threshold_long).sum()
    signals_short = (predictions <= threshold_short).sum()
    signals_flat = len(predictions) - signals_long - signals_short

    print(f"\n  Signals (LIVE thresholds: {threshold_long:.2f}/{threshold_short:.2f}):")
    print(f"    LONG:  {signals_long:6,} ({signals_long/len(predictions)*100:5.2f}%)")
    print(f"    SHORT: {signals_short:6,} ({signals_short/len(predictions)*100:5.2f}%)")
    print(f"    FLAT:  {signals_flat:6,} ({signals_flat/len(predictions)*100:5.2f}%)")

    # Clustering check
    center_range = (predictions >= 0.45) & (predictions <= 0.55)
    clustering_pct = center_range.sum() / len(predictions) * 100
    print(f"\n  Clustering near 0.5 (0.45-0.55): {clustering_pct:.1f}%")
    if clustering_pct > 70:
        print("  ⚠️ WARNING: Predictions highly clustered near 0.5 (low confidence!)")

    return signals_long, signals_short, signals_flat


# ============================================================================
# MODEL BACKTESTING
# ============================================================================

def backtest_model(model_config, model_name, df):
    """Load model, generate predictions, and run backtest."""
    print("\n" + "="*80)
    print(f"BACKTESTING: {model_name}")
    print("="*80)

    # Load model
    model_path = Path(model_config['path'])
    if not model_path.exists():
        print(f"✗ Model not found: {model_path}")
        return None

    print(f"\n✓ Loading model from {model_path}")
    artifact = joblib.load(model_path)
    model = artifact['model']
    features = artifact['features']

    print(f"  Features: {len(features)}")
    print(f"  First 5: {features[:5]}")

    # Check if all features exist
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"\n✗ Missing features ({len(missing)}):")
        for f in missing[:10]:
            print(f"    - {f}")
        print("  Skipping this model.")
        return None

    # Generate predictions
    print(f"\n📊 Generating predictions...")
    X = df[features].values

    # Handle NaN
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"  ⚠️ Found {nan_count:,} NaN values, filling with 0")
        X = np.nan_to_num(X, nan=0.0)

    proba = model.predict_proba(X)

    # Get probability of UP (class 1)
    classes = model.classes_
    if 1 in classes:
        up_idx = list(classes).index(1)
    else:
        up_idx = -1

    predictions = pd.Series(proba[:, up_idx], index=df.index)

    # Analyze predictions
    signals_long, signals_short, signals_flat = analyze_predictions(
        model_name,
        predictions,
        model_config['threshold_long'],
        model_config['threshold_short']
    )

    # Convert to signals (1 = LONG, -1 = SHORT, 0 = FLAT)
    signals = pd.Series(0, index=df.index)
    signals[predictions >= model_config['threshold_long']] = 1
    signals[predictions <= model_config['threshold_short']] = -1

    # Run backtest
    print(f"\n🔄 Running backtest...")
    result = run_backtest(
        df=df,
        signals=signals,
        model_name=model_name,
        initial_balance=50000.0,
        position_size_lots=0.25,
        stop_atr_mult=1.0,
        target_atr_mult=1.5,
        max_bars_in_trade=30,
        verbose=False
    )

    # Calculate trades per day
    if len(result.trades) > 0:
        days = (df.index[-1] - df.index[0]).days
        trades_per_day = len(result.trades) / max(days, 1)
    else:
        trades_per_day = 0

    # Print results
    print_detailed_results(result)

    # Compare to expected
    print(f"\n📈 LIVE vs BACKTEST Comparison:")
    print(f"  Expected WR:         {model_config['expected_wr']*100:.1f}%")
    print(f"  Actual WR:           {result.win_rate*100:.1f}%")
    print(f"  Expected Trades/Day: {model_config['expected_trades_per_day']:.1f}")
    print(f"  Actual Trades/Day:   {trades_per_day:.1f}")

    # Red flags
    if trades_per_day == 0:
        print(f"\n⚠️ RED FLAG: 0 signals generated (model is DEAD with live thresholds)")
    elif trades_per_day > model_config['expected_trades_per_day'] * 3:
        print(f"\n⚠️ RED FLAG: Over-firing ({trades_per_day:.1f} vs expected {model_config['expected_trades_per_day']:.1f})")
    elif abs(result.win_rate - model_config['expected_wr']) > 0.15:
        print(f"\n⚠️ RED FLAG: Win rate mismatch ({result.win_rate*100:.1f}% vs expected {model_config['expected_wr']*100:.1f}%)")
    else:
        print(f"\n✅ Performance matches expectations")

    return {
        'model_name': model_name,
        'result': result,
        'trades_per_day': trades_per_day,
        'predictions': predictions,
        'signals_long': signals_long,
        'signals_short': signals_short,
        'signals_flat': signals_flat,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*80)
    print("HONEST BACKTEST - Live Thresholds")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTesting {len(MODELS)} models with LIVE production thresholds")

    # Load data
    df = load_oos_data()

    # Ensure required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\n✗ Missing OHLCV columns: {missing}")
        return

    # Run backtests
    results = {}
    for model_name, model_config in MODELS.items():
        result = backtest_model(model_config, model_name, df)
        if result:
            results[model_name] = result

    # Summary
    print("\n" + "="*80)
    print("SUMMARY - ALL MODELS")
    print("="*80)

    for model_name, data in results.items():
        result = data['result']
        print(f"\n{model_name}:")
        print(f"  Win Rate:     {result.win_rate*100:.1f}%")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Trades:        {len(result.trades):,}")
        print(f"  Trades/Day:    {data['trades_per_day']:.1f}")
        print(f"  Return:        {result.total_return_pct:+.2f}%")
        print(f"  Max DD:        {result.max_drawdown_pct:.2f}%")

    # Overall assessment
    print("\n" + "="*80)
    print("VERDICT")
    print("="*80)

    total_trades = sum(len(r['result'].trades) for r in results.values())
    avg_wr = np.mean([r['result'].win_rate for r in results.values()])

    print(f"\nTotal Trades (combined): {total_trades:,}")
    print(f"Average Win Rate: {avg_wr*100:.1f}%")

    if total_trades == 0:
        print("\n⚠️ CRITICAL ISSUE: NO SIGNALS GENERATED")
        print("Live thresholds are TOO STRICT - all predictions fall in dead zone!")
    elif total_trades < 10:
        print("\n⚠️ WARNING: Very few signals")
        print("Thresholds may be too conservative for live trading.")
    elif avg_wr < 0.50:
        print("\n⚠️ WARNING: Win rate below 50%")
        print("Models may not be profitable with these thresholds.")
    else:
        print("\n✅ Models producing reasonable signals")

    print("\n" + "="*80)
    print("Run complete. Check results above.")
    print("="*80)


if __name__ == "__main__":
    main()
