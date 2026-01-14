#!/usr/bin/env python3
"""
Analyze Backtest Results

Read existing backtest results and provide summary.
"""

import sys
import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check for existing backtest results
print("=" * 80)
print("ANALYZING BACKTEST RESULTS")
print("=" * 80)

# Look for model paths
model_dir = PROJECT_ROOT / "models"
print(f"\nModel directory: {model_dir}")

existing_models = {
    "Model 1 (High Conf)": model_dir / "model1_high_conf.joblib",
    "Model 3 (CMF/MACD)": model_dir / "model3_cmf_macd" / "model3_cmf_macd_2014_2023_15min_balanced_fixed.joblib",
    "Model 5 (Range Reversion)": model_dir / "model5_range_reversion.joblib",
}

print("\nExisting Models:")
for name, path in existing_models.items():
    exists = path.exists() if path else False
    status = "  FOUND" if exists else "  NOT FOUND"
    print(f"  {name}: {status}")
    if exists:
        print(f"    Path: {path}")

# Check for existing backtest results
reports_dir = PROJECT_ROOT / "reports"
backtest_results = []

if reports_dir.exists():
    # Look for CSV files
    for csv_file in reports_dir.glob("**/*equity_curve*.csv"):
        print(f"\nFound backtest result: {csv_file}")
        try:
            df = pd.read_csv(csv_file)
            backtest_results.append({
                'file': str(csv_file),
                'rows': len(df),
                'start_date': df.index.min() if isinstance(df.index, pd.DatetimeIndex) else 'N/A',
                'end_date': df.index.max() if isinstance(df.index, pd.DatetimeIndex) else 'N/A',
            })
        except Exception as e:
            print(f"  Error reading: {e}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(f"Models checked: {len(existing_models)}")
print(f"Backtest results found: {len(backtest_results)}")

# Read logs from training
logs_dir = PROJECT_ROOT / "logs"
if logs_dir.exists():
    for log_file in logs_dir.glob("*.log"):
        print(f"\nLog file: {log_file}")
        # Check for completion markers
        with open(log_file, 'r') as f:
            content = f.read()
            if "TRAINING COMPLETE" in content:
                print(f"  Status: COMPLETED")
            elif "ERROR" in content:
                print(f"  Status: HAD ERRORS")
            else:
                print(f"  Status: INCOMPLETE/UNKNOWN")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print("\nRecommendations:")
print("1. Model 1 has high confidence filter (prob > 0.70)")
print("2. Model 3 has stricter CMF/MACD filters")
print("3. Model 5 has stricter range and volatility filters")
print("4. Model 6 (Order Flow) was created but needs data")
print("\nSuccess Criteria for Prop Challenge:")
print("  - Win Rate > 60%")
print("  - Max Drawdown < 5%")

