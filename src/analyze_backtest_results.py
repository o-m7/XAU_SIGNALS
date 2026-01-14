#!/usr/bin/env python3
"""
Funded Backtest Analysis for Existing Models

Analyze Model 1 and Model 5 performance against Prop Challenge criteria.
"""

import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Direct import from backtest script
from scripts.backtest_model1_2024_2025 import FundedBacktestResult, run_funded_backtest as run_backtest_m1

# Account settings
ACCOUNT_SETTINGS = {
    'initial_balance': 50000.0,
    'leverage': 10.0,
    'fixed_position_size': 0.25,
    'profit_target_pct': 0.10,  # Fixed typo
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

@dataclass
class BacktestMetrics:
    model_name: str
    win_rate: float
    max_drawdown: float
    total_return: float
    n_trades: int
    final_balance: float
    sharpe_ratio: float
    profit_factor: float
    passed: bool


def evaluate_model(model_path, model_name, test_start, test_end, verbose=True):
    """Run funded backtest and check success criteria."""
    if verbose:
        print(f"\n{'='*80}")
        print(f"BACKTESTING {model_name}")
        print(f"{'='*80}")
        print(f"Test period: {test_start} to {test_end}")
    
    if not model_path.exists():
        if verbose:
            print(f"  Model not found: {model_path}")
        return None
    
    try:
        result = run_backtest_m1(
            model_path=model_path,
            test_start=test_start,
            test_end=test_end,
            verbose=verbose,
            **ACCOUNT_SETTINGS
        )
        
        if verbose:
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
        
        if verbose:
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
        
        return BacktestMetrics(
            model_name=model_name,
            win_rate=result.win_rate,
            max_drawdown=result.max_drawdown_pct,
            total_return=result.total_return_pct,
            n_trades=result.n_trades,
            final_balance=result.final_balance,
            sharpe_ratio=result.sharpe_ratio,
            profit_factor=result.profit_factor,
            passed=passed,
        )
        
    except Exception as e:
        if verbose:
            print(f"\n  Error: {e}")
            import traceback
            traceback.print_exc()
        return None


def main():
    """Run evaluation on all models."""
    print("=" * 80)
    print("MODEL EVALUATION - FUNDED BACKTEST")
    print("=" * 80)
    print(f"\nTest Period: {TEST_START} to {TEST_END}")
    print(f"\nSuccess Criteria:")
    print(f"  Win Rate > {SUCCESS_CRITERIA['min_win_rate']*100:.0f}%")
    print(f"  Max Drawdown < {SUCCESS_CRITERIA['max_drawdown']*100:.0f}%")
    print(f"\nAccount Settings:")
    print(f"  Initial Balance: ${ACCOUNT_SETTINGS['initial_balance']:,.0f}")
    print(f"  Leverage: 1:{ACCOUNT_SETTINGS['leverage']:.0f}")
    print(f"  Position Size: {ACCOUNT_SETTINGS['fixed_position_size']:.2f} lots")
    print(f"  Profit Target: {ACCOUNT_SETTINGS['profit_target_pct']*100:.0f}%")
    print(f"  Max Drawdown: {ACCOUNT_SETTINGS['max_drawdown_pct']*100:.0f}%")
    
    results = {}
    
    # Check Model 1
    print("\n" + "-" * 80)
    print("EVALUATING MODEL 1 (Original - No High-Confidence Filter)")
    print("-" * 80)
    model1_path = PROJECT_ROOT / "models" / "model1_quality_gate.joblib"
    results['model1'] = evaluate_model(
        model1_path, "Model 1 (Original)", TEST_START, TEST_END, verbose=True
    )
    
    # Check Model 5
    print("\n" + "-" * 80)
    print("EVALUATING MODEL 5 (Range Reversion)")
    print("-" * 80)
    model5_path = PROJECT_ROOT / "models" / "model5_range_reversion.joblib"
    results['model5'] = evaluate_model(
        model5_path, "Model 5 (Range Reversion)", TEST_START, TEST_END, verbose=True
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Model':<30} {'Status':<10} {'Win Rate %':<12} {'Max DD %':<12} {'Trades':<10} {'Return %':<10} {'Profit Factor':<12} {'Passed?':<10}")
    print("-" * 80)
    
    for model_name, result in results.items():
        if result:
            status = "  PASSED  " if result.passed else "  FAILED  "
            print(f"  {model_name:<30} {status:<10} {result.win_rate:>7.1f}%     {result.max_drawdown:>7.1f}%    {result.n_trades:>10}   {result.total_return:>8.2f}%    {result.profit_factor:>7.2f}    {status:<10}")
    
    # Check if any models pass
    any_passed = any(r.passed for r in results.values() if r)
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    if any_passed:
        print("\n At least one model meets Prop Challenge criteria!")
    else:
        print("\n No model meets Prop Challenge criteria")
    
    # Save results to JSON
    report_path = PROJECT_ROOT / "reports" / f"backtest_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert results to serializable dict
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'test_period': f"{TEST_START} to {TEST_END}",
        'success_criteria': SUCCESS_CRITERIA,
        'account_settings': ACCOUNT_SETTINGS,
        'models': {}
    }
    
    for model_name, result in results.items():
        if result:
            report_data['models'][model_name] = {
                'win_rate': result.win_rate,
                'max_drawdown': result.max_drawdown,
                'total_return': result.total_return,
                'n_trades': result.n_trades,
                'passed': result.passed,
            }
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nEvaluation report saved to: {report_path}")
    
    print("\nRECOMMENDATIONS")
    print("=" * 80)
    print("For Prop Challenge Success:")
    print("1. Implement high-confidence filters (prob > 0.70)")
    print("2. Use stricter feature filters (CMF, range, volatility)")
    print("3. Apply dynamic position sizing based on confidence")
    print("4. Consider ensemble approaches combining multiple models")
    print("5. Focus on reducing max drawdown while maintaining win rate")
    print("\nModel-Specific Recommendations:")
    print("Model 1: Apply high-confidence filter to improve win rate > 60%")
    print("Model 5: Volatility filter shows promise, but needs tighter stops")
    print("Model 6: Order flow could provide microstructure edge")
    print("\nNext Steps:")
    print("1. Run complete retrain pipeline with all strict filters")
    print("2. Execute funded backtests on newly trained models")
    print("3. Generate feature importance analysis")
    print("4. Create ensemble model that combines Model 1 + Model 5")
    print("5. Validate against OOS data from 2026 Q1 onward")


if __name__ == "__main__":
    main()

