"""
train.py - Complete Training Pipeline

1. Load data
2. Build features
3. Run validation
4. Backtest
"""

import pandas as pd
from pathlib import Path
import argparse

from .config import Model5Config
from .features import build_all_features, get_feature_columns
from .validation import run_all_validations
from .labels import add_labels
from .backtest import run_backtest
from .data_loader import load_all_data


def run_pipeline(
    data_15m_path: str,
    data_quotes_path: str = None,
    data_seconds_path: str = None,
    config: Model5Config = None,
    output_dir: str = "models/model5"
) -> dict:
    """
    Complete training pipeline.
    
    Steps:
    1. Load all data
    2. Build features
    3. Statistical validation
    4. Backtest
    5. Save results
    """
    config = config or Model5Config()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== LOAD DATA ==========
    print("\n" + "="*60)
    print("STEP 1: Loading Data")
    print("="*60)
    
    df_15m, df_quotes, df_seconds = load_all_data(
        data_15m_path,
        data_quotes_path,
        data_seconds_path
    )
    
    print(f"15M data: {len(df_15m)} bars, {df_15m.index.min()} to {df_15m.index.max()}")
    
    if df_quotes is not None:
        print(f"Quotes: {len(df_quotes)} records")
    
    if df_seconds is not None:
        print(f"Seconds: {len(df_seconds)} bars")
    
    # ========== BUILD FEATURES ==========
    print("\n" + "="*60)
    print("STEP 2: Building Features")
    print("="*60)
    
    df = build_all_features(df_15m, df_quotes, df_seconds, config)
    
    # Drop rows with NaN in critical features
    critical_features = ['zscore_20', 'atr_14', 'variance_ratio_2']
    df = df.dropna(subset=critical_features)
    
    print(f"Features built: {len(df)} bars after cleanup")
    print(f"Feature columns: {len(get_feature_columns())}")
    
    # Save features
    df.to_parquet(output_dir / "features.parquet")
    
    # ========== STATISTICAL VALIDATION ==========
    print("\n" + "="*60)
    print("STEP 3: Statistical Validation")
    print("="*60)
    
    validation_results = run_all_validations(df, verbose=True)
    
    # Check if we should proceed
    vr_passed = validation_results.get('variance_ratio_2', {}).get('is_mean_reverting', False)
    zs_passed = validation_results.get('zscore_pred_2.0', {}).get('regression', {}).get('is_mean_reverting', False)
    
    if not (vr_passed or zs_passed):
        print("\n⚠️  WARNING: Statistical tests failed!")
        print("    No significant mean reversion detected.")
        print("    Strategy may not have edge.")
        proceed = input("    Proceed anyway? (y/n): ")
        if proceed.lower() != 'y':
            return {'status': 'aborted', 'reason': 'validation_failed'}
    
    # ========== BACKTEST ==========
    print("\n" + "="*60)
    print("STEP 4: Backtesting")
    print("="*60)
    
    result = run_backtest(df, config, verbose=True)
    
    # ========== SUMMARY ==========
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    
    return {
        'status': 'complete',
        'n_trades': result.n_trades,
        'win_rate': result.win_rate,
        'profit_factor': result.profit_factor,
        'sharpe': result.sharpe,
        'p_value': result.p_value,
        'validation': validation_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model 5 Training Pipeline")
    parser.add_argument("minute_path", help="Path to minute aggregate parquet file")
    parser.add_argument("--quotes", help="Path to quote data parquet file")
    parser.add_argument("--seconds", help="Path to second aggregate parquet file")
    parser.add_argument("--output", default="models/model5", help="Output directory")
    
    args = parser.parse_args()
    
    results = run_pipeline(
        args.minute_path,
        args.quotes,
        args.seconds,
        output_dir=args.output
    )
    
    print("\nResults:", results)

