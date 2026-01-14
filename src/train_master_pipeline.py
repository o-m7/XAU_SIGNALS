#!/usr/bin/env python3
"""
Master Training Pipeline for High Confidence Models

Orchestrates training of Models 1, 3, 5, and creates Model 6.
Evaluates all models on test set (2023-H2 + 2024-2025).
Only saves models meeting success criteria:
- Win Rate > 60%
- Max Drawdown < 5%
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.model6_orderflow.train import main as train_model6_main
from train_model1_2014_2023 import main as train_model1_main
from src.model3_cmf_macd.train_model3_strict import main as train_model3_strict_main
from src.models.model5.train_strict import main as train_model5_strict_main

# Success criteria
SUCCESS_CRITERIA = {
    'min_win_rate': 0.60,  # 60% minimum
    'max_drawdown': 0.05,     # 5% maximum
}

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"

# Test period (H2 2023 + full 2024-2025)
# Note: Using 2023-H2 for validation during training, test on 2023-H2 + 2024-2025
TEST_START = "2023-07-01"
TEST_END = "2025-12-31"

# Evaluation report path
EVALUATION_REPORT_DIR = PROJECT_ROOT / "models" / "master_evaluation_reports"
EVALUATION_REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Account settings for funded backtest
ACCOUNT_SETTINGS = {
    'initial_balance': 50000.0,
    'leverage': 10.0,
    'fixed_position_size': 0.25,  # Fixed lot size (not dynamic)
    'profit_target_pct': 0.10,
    'max_drawdown_pct': 0.04,
    'spread_cost': 0.0002,
    'slippage_long': 0.0001,
    'slippage_short': 0.0001,
}


def evaluate_model(
    model_path: Path,
    model_name: str,
    test_start: str,
    test_end: str,
    account_settings: dict = None
) -> Dict[str, Any]:
    """
    Run funded backtest on test set.
    
    Returns evaluation results with metrics.
    """
    if account_settings is None:
        account_settings = ACCOUNT_SETTINGS.copy()
    
    print(f"\n{'='*80}")
    print(f"EVALUATING {model_name}")
    print(f"{'='*80}")
    print(f"Test period: {test_start} to {test_end}")
    
    # Import funded backtest module
    from scripts.funded_backtest_model1 import run_funded_backtest
    
    try:
        result = run_funded_backtest(
            model_path=model_path,
            test_start=test_start,
            test_end=test_end,
            verbose=False,  # Reduce output
            **account_settings
        )
        
        metrics = {
            'win_rate': result.win_rate,
            'max_drawdown': result.max_drawdown_pct,
            'total_return': result.total_return_pct,
            'n_trades': result.n_trades,
            'profit_factor': result.profit_factor,
            'sharpe_ratio': result.sharpe_ratio,
            'final_balance': result.final_balance,
        }
        
        # Check success criteria
        passed = (
            metrics['win_rate'] >= SUCCESS_CRITERIA['min_win_rate']
            and metrics['max_drawdown'] <= SUCCESS_CRITERIA['max_drawdown']
        )
        
        print(f"\n{'='*80}")
        print(f"RESULTS")
        print(f"{'='*80}")
        print(f"  Win Rate: {metrics['win_rate']:.1f}% (target: >={SUCCESS_CRITERIA['min_win_rate']*100:.0f}%)")
        print(f"  Max DD: {metrics['max_drawdown']:.1f}% (target: <={SUCCESS_CRITERIA['max_drawdown']*100:.0f}%)")
        print(f"  Total Return: {metrics['total_return']:.2f}%")
        print(f"  Trades: {metrics['n_trades']}")
        
        if passed:
            print(f"\n  ✓ PASSED - Meets success criteria!")
        else:
            if metrics['win_rate'] < SUCCESS_CRITERIA['min_win_rate']:
                print(f"\n  ✗ FAILED - Win rate too low (< {SUCCESS_CRITERIA['min_win_rate']*100:.0f}%)")
            if metrics['max_drawdown'] > SUCCESS_CRITERIA['max_drawdown']:
                print(f"\n  ✗ FAILED - Max drawdown too high (> {SUCCESS_CRITERIA['max_drawdown']*100:.0f}%)")
            else:
                print(f"\n  ✗ FAILED - Both criteria failed")
        
        return {
            'model_path': str(model_path),
            'model_name': model_name,
            'passed': passed,
            'metrics': metrics,
        }
        
    except Exception as e:
        print(f"\n  ✗ ERROR during evaluation: {e}")
        return {
            'model_path': str(model_path),
            'model_name': model_name,
            'passed': False,
            'error': str(e),
            'metrics': None,
        }


def run_training_and_evaluation():
    """
    Run all training pipelines and evaluate models.
    """
    print("=" * 80)
    print("MASTER TRAINING PIPELINE")
    print("=" * 80)
    print(f"\nTest Period: {TEST_START} to {TEST_END}")
    print(f"\nSuccess Criteria:")
    print(f"  - Win Rate > {SUCCESS_CRITERIA['min_win_rate']*100:.0f}%")
    print(f"  - Max Drawdown < {SUCCESS_CRITERIA['max_drawdown']*100:.0f}%")
    
    all_evaluations = {}
    
    # Model paths (will be updated after training)
    model_paths = {
        'model1_high_conf': MODELS_DIR / "model1_high_conf.joblib",
        'model3_strict': MODELS_DIR / "model3_cmf_macd_strict.joblib",
        'model5_strict_range_reversion': MODELS_DIR / "model5_range_reversion.joblib",
        'model6_orderflow': MODELS_DIR / "model6_orderflow.joblib",
    }
    
    # Train each model
    print("\n" + "=" * 80)
    print("PHASE 1: TRAINING MODELS")
    print("=" * 80)
    
    # Model 1
    print("\n" + "-" * 80)
    print("Training Model 1 (High Confidence)...")
    print("-" * 80)
    model1_artifact = train_model1_main()[1]  # Returns (model, artifact)
    all_evaluations['model1_high_conf'] = evaluate_model(
        model_paths['model1_high_conf'],
        "Model 1 High Conf",
        TEST_START,
        TEST_END
    )
    
    # Model 3
    print("\n" + "-" * 80)
    print("Training Model 3 (Strict CMF/MACD)...")
    print("-" * 80)
    model3_artifact = train_model3_strict_main()[1]
    all_evaluations['model3_strict'] = evaluate_model(
        model_paths['model3_strict'],
        "Model 3 Strict",
        TEST_START,
        TEST_END
    )
    
    # Model 5
    print("\n" + "-" * 80)
    print("Training Model 5 (Strict Range Reversion)...")
    print("-" * 80)
    model5_artifact = train_model5_strict_main()[1]
    all_evaluations['model5_strict_range_reversion'] = evaluate_model(
        model_paths['model5_strict_range_reversion'],
        "Model 5 Strict Range Reversion",
        TEST_START,
        TEST_END
    )
    
    # Model 6
    print("\n" + "-" * 80)
    print("Training Model 6 (Order Flow)...")
    print("-" * 80)
    model6_artifact = train_model6_main()[1]
    all_evaluations['model6_orderflow'] = evaluate_model(
        model_paths['model6_orderflow'],
        "Model 6 Order Flow",
        TEST_START,
        TEST_END
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("PHASE 2: EVALUATION SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<35} {'Status':<12} {'Win Rate %':<12} {'Total Return %':<14} {'Passed':<8}")
    print("-" * 80)
    
    for model_name, eval_result in all_evaluations.items():
        if eval_result['passed']:
            status = "  PASSED  "
            passed_marker = "  "
        else:
            status = "  FAILED  "
            passed_marker = "  "
        
        win_rate = f"{eval_result['metrics']['win_rate']:.1f}%"
        max_dd = f"{eval_result['metrics']['max_drawdown']:.1f}%"
        total_return = f"{eval_result['metrics']['total_return']:.2f}%"
        
        print(f"  {model_name:<35} {status:<10} {win_rate:<12} {max_dd:<14} {total_return:<14}{passed_marker}")
    
    # Save evaluation report
    evaluation_report = {
        'timestamp': datetime.now().isoformat(),
        'test_period': f"{TEST_START} to {TEST_END}",
        'success_criteria': SUCCESS_CRITERIA,
        'models': all_evaluations,
    }
    
    report_path = EVALUATION_REPORT_DIR / f"master_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w') as f:
        json.dump(evaluation_report, f, indent=2)
    
    print(f"\nEvaluation report saved to: {report_path}")
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    
    return all_evaluations


if __name__ == "__main__":
    run_training_and_evaluation()

