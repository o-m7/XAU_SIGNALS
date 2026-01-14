#!/usr/bin/env python3
"""
Run Funded Backtests on All Models with Realistic Market Costs
"""

import sys
from pathlib import Path
import subprocess
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project Root: {PROJECT_ROOT}")
print(f"Models Directory: {PROJECT_ROOT / 'models'}")

# Account settings
ACCOUNT_SETTINGS = {
    'initial_balance': 50000.0,
    'leverage': 10.0,
    'fixed_position_size': 0.25,
    'profit_target_pct': 0.10,
    'max_drawdown_pct': 0.04,
}

# Market costs per Standard Lot (100 oz gold)
MARKET_COSTS = {
    'spread_pips': 0.25,  # 0.25 pips = $2.50
    'slippage_pips': 0.10,  # 0.10 pips = $1.00
    'commission_per_side': 5.00,  # $5.00 per lot per side
}

# Success criteria
SUCCESS_CRITERIA = {
    'min_win_rate': 0.60,  # 60% minimum
    'max_drawdown': 0.05,  # 5% maximum
}


def run_backtest(model_path, test_start, test_end):
    """
    Run funded backtest for a specific model.
    
    Returns:
        dict: Backtest results including gross and net performance
    """
    print(f"\n{'='*80}")
    print(f"FUNDED BACKTEST: {model_path.name}")
    print(f"{'='*80}")
    print(f"Test Period: {test_start} to {test_end}")
    print(f"\nAccount Settings:")
    print(f"  Initial Balance: ${ACCOUNT_SETTINGS['initial_balance']:,.2f}")
    print(f"  Leverage: 1:{ACCOUNT_SETTINGS['leverage']:.0f}")
    print(f"  Position Size: {ACCOUNT_SETTINGS['fixed_position_size']} lots")
    print(f"  Profit Target: {ACCOUNT_SETTINGS['profit_target_pct']*100:.0f}%")
    print(f"  Max Drawdown Limit: {ACCOUNT_SETTINGS['max_drawdown_pct']*100:.0f}%")
    print(f"\nMarket Costs (per Standard Lot):")
    print(f"  Spread: {MARKET_COSTS['spread_pips']} pips (${MARKET_COSTS['spread_pips']*10:.2f})")
    print(f"  Slippage: {MARKET_COSTS['slippage_pips']} pips (${MARKET_COSTS['slippage_pips']*10:.2f})")
    print(f"  Commission: ${MARKET_COSTS['commission_per_side']:.2f} per side (${MARKET_COSTS['commission_per_side']*2:.2f} round-turn)")
    
    # Import backtest function
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from funded_backtest_model1 import run_funded_backtest
    
    # Run backtest
    try:
        result = run_funded_backtest(
            model_path=model_path,
            test_start=test_start,
            test_end=test_end,
            initial_balance=ACCOUNT_SETTINGS['initial_balance'],
            leverage=ACCOUNT_SETTINGS['leverage'],
            fixed_position_size=ACCOUNT_SETTINGS['fixed_position_size'],
            profit_target_pct=ACCOUNT_SETTINGS['profit_target_pct'],
            max_drawdown_pct=ACCOUNT_SETTINGS['max_drawdown_pct'],
        )
        
        return result
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run backtests on all models."""
    print("\n" + "="*80)
    print("COMPREHENSIVE FUNDED BACKTEST ANALYSIS")
    print("="*80)
    print("\nPurpose: Evaluate all models against Prop Challenge criteria with realistic market costs")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    models_dir = PROJECT_ROOT / "xauusd_signals" / "models"
    
    # Define models to test
    models_to_test = [
        {
            'name': 'Model 1 (Original)',
            'path': models_dir / "model1_quality_gate.joblib",
            'description': 'Microstructure features, original high-confidence filter',
        },
        {
            'name': 'Model 3 (CMF/MACD - Strict)',
            'path': models_dir / "model3_strict.joblib",
            'description': 'CMF > 0.15, MACD crossover confirmation, spread filter',
        },
        {
            'name': 'Model 5 (Range Reversion - Strict)',
            'path': models_dir / "model5_range_reversion.joblib",
            'description': 'ER < 0.2, SNR < 0.4, ATR volatility filter',
        },
    ]
    
    results = {}
    
    for model_info in models_to_test:
        model_name = model_info['name']
        model_path = model_info['path']
        description = model_info['description']
        
        print(f"\n{'#'*80}")
        print(f"# MODEL: {model_name}")
        print(f"# {description}")
        print(f"{'#'*80}")
        
        if not model_path.exists():
            print(f"\n⚠️  Model file not found: {model_path}")
            results[model_name] = None
            continue
        
        # Run backtest
        result = run_backtest(
            model_path=model_path,
            test_start="2024-01-01",
            test_end="2025-12-31"
        )
        
        if result:
            # Check success criteria
            passed = (
                result.win_rate >= SUCCESS_CRITERIA['min_win_rate']
                and result.max_drawdown_pct <= SUCCESS_CRITERIA['max_drawdown']
            )
            
            results[model_name] = {
                'path': str(model_path),
                'description': description,
                'n_trades': result.n_trades,
                'n_long': result.n_long,
                'n_short': result.n_short,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'sharpe_ratio': result.sharpe_ratio,
                'total_return_pct': result.total_return_pct,
                'max_drawdown_pct': result.max_drawdown_pct,
                'final_balance': result.final_balance,
                'initial_balance': result.initial_balance,
                'profit_target': result.profit_target,
                'max_drawdown_limit': result.max_drawdown_limit,
                'leverage': result.leverage,
                'position_size': result.position_size,
                'passed': passed,
                'account_status': result.account_status,
            }
        else:
            results[model_name] = None
    
    # Summary Report
    print("\n" + "="*80)
    print("COMPREHENSIVE BACKTEST SUMMARY")
    print("="*80)
    
    print(f"\n{'Model':<40} {'Trades':>8} {'Long':>6} {'Short':>6} {'Win %':>8} {'PF':>6} {'Sharpe':>7} {'Ret %':>8} {'DD %':>8} {'Status':<12}")
    print("-"*120)
    
    for model_name, result in results.items():
        if result:
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"{model_name:<40} {result['n_trades']:>8} {result['n_long']:>6} {result['n_short']:>6} {result['win_rate']*100:>7.1f}% {result['profit_factor']:>5.2f} {result['sharpe_ratio']:>7.2f} {result['total_return_pct']:>7.2f}% {result['max_drawdown_pct']*100:>7.2f}% {status:<12}")
        else:
            print(f"{model_name:<40} ERROR")
    
    # Detailed Results
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    for model_name, result in results.items():
        if result:
            print(f"\n{'-'*80}")
            print(f"{model_name}")
            print(f"{'-'*80}")
            print(f"\n📊 Trading Statistics:")
            print(f"  Total Trades: {result['n_trades']}")
            print(f"    - Long Trades: {result['n_long']}")
            print(f"    - Short Trades: {result['n_short']}")
            print(f"  Win Rate: {result['win_rate']*100:.2f}%")
            print(f"  Profit Factor: {result['profit_factor']:.2f}")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.3f}")
            
            print(f"\n💰 Performance Metrics:")
            print(f"  Initial Balance: ${result['initial_balance']:,.2f}")
            print(f"  Final Balance: ${result['final_balance']:,.2f}")
            print(f"  Total Return: {result['total_return_pct']:+.2f}%")
            print(f"  Max Drawdown: {result['max_drawdown_pct']*100:.2f}%")
            
            print(f"\n⚙️  Account Settings:")
            print(f"  Leverage: 1:{result['leverage']:.0f}")
            print(f"  Position Size: {result['position_size']} lots")
            print(f"  Profit Target: {result['profit_target']*100:.0f}%")
            print(f"  Max Drawdown Limit: {result['max_drawdown_limit']*100:.0f}%")
            
            print(f"\n🎯 Prop Challenge Status: {'✓ PASSED' if result['passed'] else '✗ FAILED'}")
            print(f"  Account Status: {result['account_status']}")
    
    # Overall Conclusion
    print("\n" + "="*80)
    print("OVERALL CONCLUSION")
    print("="*80)
    
    passed_models = [name for name, r in results.items() if r and r['passed']]
    failed_models = [name for name, r in results.items() if r and not r['passed']]
    error_models = [name for name, r in results.items() if r is None]
    
    if passed_models:
        print(f"\n✓ {len(passed_models)} model(s) meet Prop Challenge criteria:")
        for model in passed_models:
            result = results[model]
            print(f"    • {model}: {result['win_rate']*100:.1f}% win rate, {result['max_drawdown_pct']*100:.1f}% max DD")
    else:
        print("\n✗ No models meet Prop Challenge criteria")
    
    if failed_models:
        print(f"\n✗ {len(failed_models)} model(s) failed criteria:")
        for model in failed_models:
            result = results[model]
            fail_reason = []
            if result['win_rate'] < SUCCESS_CRITERIA['min_win_rate']:
                fail_reason.append(f"Win rate too low ({result['win_rate']*100:.1f}% < {SUCCESS_CRITERIA['min_win_rate']*100:.0f}%)")
            if result['max_drawdown_pct'] > SUCCESS_CRITERIA['max_drawdown']:
                fail_reason.append(f"Max drawdown too high ({result['max_drawdown_pct']*100:.1f}% > {SUCCESS_CRITERIA['max_drawdown']*100:.0f}%)")
            print(f"    • {model}: {'; '.join(fail_reason)}")
    
    if error_models:
        print(f"\n⚠️  {len(error_models)} model(s) encountered errors:")
        for model in error_models:
            print(f"    • {model}")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if passed_models:
        print("\n🎉 Congratulations! The following models are ready for Prop Challenge:")
        for model in passed_models:
            print(f"    • {model}")
        print("\nNext Steps:")
        print("1. Verify model ensemble strategy")
        print("2. Test on current market data (live paper trading)")
        print("3. Implement proper risk management")
        print("4. Prepare for live trading")
    else:
        print("\n💡 Model Optimization Required:")
        print("\nFor all models:")
        print("  • Apply high-confidence filters (predict_proba > 0.70)")
        print("  • Reduce position size during high volatility")
        print("  • Implement dynamic stop-loss based on ATR")
        print("\nModel-Specific:")
        print("Model 1: Fix syntax errors, retrain with balanced weights")
        print("Model 3: Ensure strong momentum confirmation before entry")
        print("Model 5: Optimize ER and SNR thresholds, use asymmetric R/R (1.5:1)")
        print("\nNext Steps:")
        print("1. Fix Model 1 syntax error in line 248")
        print("2. Retrain all models with corrected configurations")
        print("3. Run backtests again with realistic costs")
        print("4. Consider ensemble approach combining best signals")
    
    # Save results to JSON
    report_path = PROJECT_ROOT / "reports" / f"funded_backtest_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'test_period': "2024-01-01 to 2025-12-31",
        'account_settings': ACCOUNT_SETTINGS,
        'market_costs': MARKET_COSTS,
        'success_criteria': SUCCESS_CRITERIA,
        'models': {name: result for name, result in results.items() if result},
        'summary': {
            'total_models': len(models_to_test),
            'passed': len(passed_models),
            'failed': len(failed_models),
            'errors': len(error_models),
        }
    }
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n📄 Full report saved to: {report_path}")


if __name__ == "__main__":
    main()

