#!/usr/bin/env python3
"""
Run Funded Backtests on Existing Models

Analyze Model 1 and Model 5 performance against Prop Challenge criteria.
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from datetime import datetime
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from correct module - using direct imports
from scripts.backtest_model1_2024_2025 import FundedBacktestResult, run_funded_backtest as run_backtest_m1

# Account settings
ACCOUNT_SETTINGS = {
    'initial_balance': 50000.0,
    'leverage': 10.0,
    'fixed_position_size': 0.25,
    'profit_target_pct': 0.10,
    'max_drawdown_pct': 0.04,
    'spread_cost': 0.0002,
    'slippage_long': 0.0001,
    'slippage_short': 0.0001,
}

# Test period
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

# Success criteria
SUCCESS_CRITERIA = {
    'min_win_rate': 0.60,  # 60% minimum
    'max_drawdown': 0.05,     # 5% maximum
}


def evaluate_model(model_path, model_name, test_start, test_end):
    """Run funded backtest and check success criteria."""
    print(f"\n{'='*80}")
    print(f"BACKTESTING {model_name}")
    print(f"{'='*80}")
    print(f"Test period: {test_start} to {test_end}")
    
    if not model_path.exists():
        print(f"  Model not found: {model_path}")
        return None
    
    try:
        result = run_backtest_m1(
            model_path=model_path,
            test_start=test_start,
            test_end=test_end,
            verbose=False,
            **ACCOUNT_SETTINGS
        )
        
        print(f"\nResults:")
        print(f"  Win Rate: {result.win_rate:.1f}%")
        print(f"  Max DD: {result.max_drawdown_pct:.1f}%")
        print(f"  Total Return: {result.total_return_pct:.2f}%")
        print(f"  Trades: {result.n_trades}")
        print(f"  Final Balance: ${result.final_balance:,.2f}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        
        # Check success criteria
        passed = (
            result.win_rate >= SUCCESS_CRITERIA['min_win_rate']
            and result.max_drawdown_pct <= SUCCESS_CRITERIA['max_drawdown']
        )
        
        print(f"\nSuccess Criteria (Win Rate > {SUCCESS_CRITERIA['min_win_rate']*100:.0f}%, Max DD < {SUCCESS_CRITERIA['max_drawdown']*100:.0f}%):")
        if passed:
            print(f"  PASSED")
        else:
            if result.win_rate < SUCCESS_CRITERIA['min_win_rate']:
                print(f"  FAILED - Win rate too low ({result.win_rate:.1f}% < {SUCCESS_CRITERIA['min_win_rate']*100:.0f}%)")
            if result.max_drawdown_pct > SUCCESS_CRITERIA['max_drawdown']:
                print(f"  FAILED - Max DD too high ({result.max_drawdown_pct:.1f}% > {SUCCESS_CRITERIA['max_drawdown']*100:.0f}%)")
            else:
                print(f"  FAILED - Both criteria failed")
        
        return {
            'model_name': model_name,
            'win_rate': result.win_rate,
            'max_drawdown': result.max_drawdown_pct,
            'total_return': result.total_return_pct,
            'n_trades': result.n_trades,
            'final_balance': result.final_balance,
            'sharpe_ratio': result.sharpe_ratio,
            'profit_factor': result.profit_factor,
            'passed': passed,
        }
        
    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run evaluation on all models."""
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print(f"\nTest Period: {TEST_START} to {TEST_END}")
    print(f"\nSuccess Criteria:")
    print(f"  Win Rate > {SUCCESS_CRITERIA['min_win_rate']*100:.0f}%")
    print(f"  Max Drawdown < {SUCCESS_CRITERIA['max_drawdown']*100:.0f}%")
    
    results = {}
    
    # Evaluate Model 1
    print("\n" + "-" * 80)
    print("EVALUATING MODEL 1")
    print("-" * 80)
    model1_path = PROJECT_ROOT / "models" / "model1_quality_gate.joblib"
    results['model1'] = evaluate_model(
        model1_path, "Model 1 (Original)", TEST_START, TEST_END
    )
    
    # Evaluate Model 5
    print("\n" + "-" * 80)
    print("EVALUATING MODEL 5")
    print("-" * 80)
    model5_path = PROJECT_ROOT / "models" / "model5_range_reversion.joblib"
    results['model5'] = evaluate_model(
        model5_path, "Model 5 (Range Reversion)", TEST_START, TEST_END
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Model':<30} {'Status':<10} {'Win Rate %':<12} {'Max DD %':<12} {'Trades':<10} {'Return %':<10} {'Passed?':<10}")
    print("-" * 80)
    
    for model_name, result in results.items():
        if result:
            status = " PASSED" if result['passed'] else " FAILED "
            print(f"  {model_name:<30} {status:<10} {result['win_rate']:>7.1f}%     {result['max_drawdown']:>7.1f}%    {result['n_trades']:>10}   {result['total_return']:>8.2f}%    {status:<10}")
    
    # Check if any models pass
    any_passed = any(r['passed'] for r in results.values() if r)
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    if any_passed:
        print("\n✓ At least one model meets Prop Challenge criteria!")
    else:
        print("\nNo model meets Prop Challenge criteria")
    
    print("\nRECOMMENDATIONS:")
    print("1. Models need higher win rates (> 60%)")
    print("2. Models need lower max drawdowns (< 5%)")
    print("3. Model 1: Original - no high-confidence filter applied")
    print("4. Model 5: Range Reversion - uses statistical features")


if __name__ == "__main__":
    main()
