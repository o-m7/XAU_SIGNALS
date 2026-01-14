"""
Train Model 5 on 2014-2023 data, then test on 2024-2025 out-of-sample.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model5.config import Model5Config
from src.models.model5.features import build_all_features, get_feature_columns
from src.models.model5.validation import run_all_validations
from src.models.model5.backtest import run_backtest, BacktestResult
from src.models.model5.data_loader import load_all_data


def main():
    print("=" * 80)
    print("MODEL 5: TRAINING ON 2014-2023, TESTING ON 2024-2025")
    print("=" * 80)
    
    # Configuration
    config = Model5Config()
    
    # Data paths
    features_file = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
    
    if not features_file.exists():
        print(f"❌ Features file not found: {features_file}")
        print("   Please run feature engineering first")
        return
    
    # Load data
    print("\n[1] Loading data...")
    df = pd.read_parquet(features_file)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    df = df.sort_index()
    
    print(f"   Total rows: {len(df):,}")
    print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    # Check available date range
    available_start = df.index.min().date()
    available_end = df.index.max().date()
    
    print(f"\n   Available data: {available_start} to {available_end}")
    
    # Define periods
    train_start = "2014-01-01"
    train_end = "2023-12-31"
    test_start = "2024-01-01"
    test_end = "2025-12-31"
    
    # Adjust if data doesn't go back to 2014
    if pd.to_datetime(train_start).date() < available_start:
        train_start = str(available_start)
        print(f"   ⚠️  Adjusted train_start to {train_start} (data starts {available_start})")
    
    # Split data
    train_df = df[(df.index >= train_start) & (df.index <= train_end)].copy()
    test_df = df[(df.index >= test_start) & (df.index <= test_end)].copy()
    
    print(f"\n   Train period: {len(train_df):,} rows ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"   Test period:  {len(test_df):,} rows ({test_df.index.min().date()} to {test_df.index.max().date()})")
    
    if len(train_df) == 0:
        print("❌ No training data found")
        return
    
    if len(test_df) == 0:
        print("❌ No test data found")
        return
    
    # ========== BUILD FEATURES ==========
    print("\n[2] Building features...")
    
    # Resample to 15-minute if needed (check if already 15M)
    # For now, assume we need to resample from minute data
    # But if features file already has 15M, we can use it directly
    
    # Check if we have OHLCV columns
    required_cols = ['open', 'high', 'low', 'close']
    if not all(col in train_df.columns for col in required_cols):
        print("❌ Missing OHLCV columns. Need to resample from minute data.")
        print("   Please provide minute aggregate data file path")
        return
    
    # Build features on training data
    print("   Building features for training period...")
    train_features = build_all_features(train_df, None, None, config)
    
    # Drop rows with NaN in critical features
    critical_features = ['zscore_20', 'atr_14', 'variance_ratio_2']
    train_features = train_features.dropna(subset=critical_features)
    
    print(f"   Training features: {len(train_features):,} bars after cleanup")
    
    # Build features on test data
    print("   Building features for test period...")
    test_features = build_all_features(test_df, None, None, config)
    test_features = test_features.dropna(subset=critical_features)
    
    print(f"   Test features: {len(test_features):,} bars after cleanup")
    
    # ========== STATISTICAL VALIDATION (on training data) ==========
    print("\n[3] Statistical Validation (on training data)...")
    print("=" * 80)
    
    validation_results = run_all_validations(train_features, verbose=True)
    
    # Check if we should proceed
    vr_passed = validation_results.get('variance_ratio_2', {}).get('is_mean_reverting', False)
    zs_passed = validation_results.get('zscore_pred_2.0', {}).get('regression', {}).get('is_mean_reverting', False)
    
    if not (vr_passed or zs_passed):
        print("\n⚠️  WARNING: Statistical tests failed on training data!")
        print("    No significant mean reversion detected.")
        print("    Proceeding anyway for demonstration...")
    
    # ========== BACKTEST ON TRAINING DATA ==========
    print("\n" + "=" * 80)
    print("[4] BACKTEST ON TRAINING DATA (2014-2023)")
    print("=" * 80)
    
    train_result = run_backtest(train_features, config, verbose=True)
    
    # ========== OUT-OF-SAMPLE BACKTEST ==========
    print("\n" + "=" * 80)
    print("[5] OUT-OF-SAMPLE BACKTEST (2024-2025)")
    print("=" * 80)
    
    test_result = run_backtest(test_features, config, verbose=True)
    
    # ========== COMPARISON ==========
    print("\n" + "=" * 80)
    print("COMPARISON: TRAIN vs TEST")
    print("=" * 80)
    
    print(f"\n{'Metric':<25} {'Train (2014-2023)':<25} {'Test (2024-2025)':<25}")
    print("-" * 75)
    print(f"{'Trades':<25} {train_result.n_trades:<25,} {test_result.n_trades:<25,}")
    print(f"{'Win Rate':<25} {train_result.win_rate:<24.1%} {test_result.win_rate:<24.1%}")
    print(f"{'Profit Factor':<25} {train_result.profit_factor:<25.2f} {test_result.profit_factor:<25.2f}")
    print(f"{'Total PnL':<25} ${train_result.total_pnl:<24.2f} ${test_result.total_pnl:<24.2f}")
    print(f"{'Avg PnL':<25} ${train_result.avg_pnl:<24.2f} ${test_result.avg_pnl:<24.2f}")
    print(f"{'Sharpe':<25} {train_result.sharpe:<25.3f} {test_result.sharpe:<25.3f}")
    print(f"{'Max Drawdown':<25} ${train_result.max_drawdown:<24.2f} ${test_result.max_drawdown:<24.2f}")
    print(f"{'T-statistic':<25} {train_result.t_statistic:<25.2f} {test_result.t_statistic:<25.2f}")
    print(f"{'P-value':<25} {train_result.p_value:<25.4f} {test_result.p_value:<25.4f}")
    print(f"{'Significant':<25} {'YES' if train_result.p_value < 0.05 else 'NO':<25} {'YES' if test_result.p_value < 0.05 else 'NO':<25}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    if test_result.p_value < 0.05 and test_result.win_rate > 0.53:
        print("✅ Model 5 PASSED out-of-sample validation")
        print(f"   - Test win rate: {test_result.win_rate:.1%} (> 53%)")
        print(f"   - Test p-value: {test_result.p_value:.4f} (< 0.05)")
        print(f"   - Test Sharpe: {test_result.sharpe:.3f}")
    else:
        print("⚠️  Model 5 may not have edge on out-of-sample data")
        if test_result.p_value >= 0.05:
            print(f"   - Test p-value: {test_result.p_value:.4f} (not significant)")
        if test_result.win_rate <= 0.53:
            print(f"   - Test win rate: {test_result.win_rate:.1%} (<= 53%)")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    
    return {
        'train': {
            'n_trades': train_result.n_trades,
            'win_rate': train_result.win_rate,
            'profit_factor': train_result.profit_factor,
            'sharpe': train_result.sharpe,
            'p_value': train_result.p_value,
        },
        'test': {
            'n_trades': test_result.n_trades,
            'win_rate': test_result.win_rate,
            'profit_factor': test_result.profit_factor,
            'sharpe': test_result.sharpe,
            'p_value': test_result.p_value,
        },
        'validation': validation_results,
    }


if __name__ == "__main__":
    results = main()

