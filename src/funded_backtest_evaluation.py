#!/usr/bin/env python3
"""
Funded Backtest Analysis for Existing Models

Direct evaluation of trained models without retraining.
Simulates funded account with realistic market costs.
"""

import sys
from pathlib import Path
import json
import joblib
import pandas as pd
from datetime import datetime
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test settings
ACCOUNT_SETTINGS = {
    'initial_balance': 50000.0,
    'leverage': 10.0,
    'fixed_position_size': 0.25,
    'profit_target_pct': 0.10,
    'max_drawdown_pct': 0.04,
    'spread_cost': 0.0002,  # 0.25 pips = $2.50 per lot
    'slippage_long': 0.0001,  # 1 pip = $1.00 per lot
    'slippage_short': 0.0001,
}

# Success criteria
SUCCESS_CRITERIA = {
    'min_win_rate': 0.60,  # 60% minimum
    'max_drawdown': 0.05,     # 5% maximum
}


@dataclass
class BacktestResult:
    """Funded backtest result."""
    model_name: str
    win_rate: float
    max_drawdown: float
    total_return: float
    n_trades: int
    final_balance: float
    sharpe_ratio: float
    profit_factor: float
    trades: pd.DataFrame
    passed: bool


def load_model(model_path, model_name):
    """Load model and return metadata."""
    if not model_path.exists():
        return None
    
    try:
        artifact = joblib.load(model_path)
        return {
            'model_name': model_name,
            'features': artifact.get('features', []),
            'train_period': artifact.get('train_period', 'N/A'),
            'metrics': artifact.get('metrics', {}),
        }
    except Exception as e:
        return None


def run_funded_backtest_direct(model_path, test_start, test_end, verbose=False):
    """
    Run funded backtest with realistic costs.
    
    Simulates a prop funded account:
    - $50,000 starting capital
    - 1:30 leverage
    - 0.25 lot fixed position size (contract size)
    - 10% profit target
    - 4% max drawdown limit
    
    Applies realistic market costs:
    - Spread: 0.25 pips ($2.50 per standard lot)
    - Slippage: 0.1 pip ($1.00 per lot)
    - Commission: $5.00 per round-turn trade
    
    Returns Net Performance (after costs).
    """
    print(f"Running funded backtest on: {model_path.name}")
    print(f"Test period: {test_start} to {test_end}")
    
    # Load minute data for test period
    from src.data_loader import load_multi_year_data
    
    # Load data for test period
    years_to_load = list(range(int(test_start[:4]), int(test_end[:4]) + 1))
    print(f"Loading data for years: {years_to_load}...")
    df_bars = load_multi_year_data(minute_dir=str(PROJECT_ROOT.parent / "Data" / "ohlcv_minute"), quotes_dir=str(PROJECT_ROOT.parent / "Data" / "quotes"), years=years_to_load)
    
    if df_bars is None or len(df_bars) == 0:
        print("ERROR: No data loaded")
        return None
    
    # Load existing trained model
    model_info = load_model(model_path, model_path.name)
    if model_info is None:
        print(f"ERROR: Could not load model {model_path.name}")
        return None
    
    print(f"Model loaded: {model_info['model_name']}")
    print(f"Features: {len(model_info['features'])}")
    
    # Generate signals
    print(f"Generating signals for test period ({len(df_bars):,} bars)...")
    from src.signal_generator import generate_signals_from_model
    
    # Generate signals for entire period
    # This is inefficient for backtest but ensures correct signal generation
    signals = generate_signals_from_model(
        model_path=model_path,
        df=df_bars,
        verbose=False
    )
    
    if 'predictions' not in signals.columns:
        print("ERROR: No predictions in signal data")
        return None
    
    # Merge signals back to minute data
    print(f"Merging signals back to minute bars...")
    df_merged = pd.merge(signals[['timestamp', 'predictions']], df_bars, left_index=True, how='left')
    
    # Filter for test period
    df_test = df_merged[(df_merged.index >= test_start) & (df_merged.index <= test_end)].copy()
    
    # Add spread and volatility for cost calculation
    if 'spread_pct' not in df_test.columns:
        # Add realistic spread if not present
        df_test['spread_pct'] = df_test.get('spread_pct', 0.0002)  # 0.25 pips = $2.50 per lot
        print(f"Added spread to {len(df_test)} bars")
    
    if 'atr_14' not in df_test.columns:
        print("ERROR: No ATR in data - required for slippage calculation")
        return None
    
    # Calculate cost per trade
    # Fixed lot size = 1.0 standard lot (100 oz gold)
    # Contract value varies with price, but for simulation we use fixed multiplier
    
    # Entry Cost (spread + slippage)
    # Spread: 0.25 pips ($2.50 per lot)
    # Slippage: 0.1 pip ($1.00 per lot)
    df_test['entry_cost'] = df_test['spread_pct'] + 0.0001  # 0.35 pips = $3.50 per lot
    
    # Exit Cost (spread + slippage)
    df_test['exit_cost'] = df_test['spread_pct'] + 0.0001
    
    # Commission: $5.00 per round turn ($50.00 per lot)
    # Contract size: 0.25 lots = 25 oz gold
    # Contract value: ~25 oz * $2000/oz = $50,000
    # Commission per contract value: $5.00 / $50,000 = 0.0001 (0.01%)
    df_test['commission_pct'] = 0.0001
    
    # Total Cost (entry + exit + commission)
    df_test['total_cost_pct'] = df_test['entry_cost'] + df_test['exit_cost'] + df_test['commission_pct']
    
    # Calculate Net PnL (after costs)
    # We need to calculate returns for each trade and subtract costs
    df_test['gross_return'] = 0.0
    df_test['net_return'] = 0.0
    df_test['cumulative_gross'] = 0.0
    df_test['cumulative_net'] = 0.0
    
    print(f"Calculating trade execution (entry cost: {df_test['entry_cost'].mean():.4f}%, exit cost: {df_test['exit_cost'].mean():.4f}%, commission: {df_test['commission_pct'].mean():.4f}%)...")
    
    # For each prediction, simulate trade and calculate returns
    # This is a simplified backtest - in production you'd iterate through actual trades
    # For evaluation, we'll use predictions directly
    
    trades = []
    current_balance = ACCOUNT_SETTINGS['initial_balance']
    
    # Only evaluate when prediction is not neutral
    df_eval = df_test[df_test['predictions'] != 0].copy()
    
    print(f"Evaluating {len(df_eval):,} trading opportunities...")
    
    # Group by entry to simulate realistic trade batching
    # In funded account, multiple trades may be executed simultaneously
    # For simplicity, we'll process entries sequentially but account for batching
    for idx in range(0, len(df_eval), 10):  # Sample 1000 for speed
        if idx % 100 == 0:
            print(f"  Processing: {idx}/{len(df_eval)}...")
        
        row = df_eval.iloc[idx]
        prediction = row['predictions']  # -1 (SHORT), 1 (LONG)
        entry_price = row['close']
        
        # Simulate trade
        direction = 1 if prediction > 0 else -1
        
        # Skip if direction doesn't match prediction
        if direction * prediction <= 0:
            continue
        
        # Calculate PnL based on close price at bar idx + horizon
        horizon_minutes = 15  # Using 15-minute horizon (y_tb_15)
        
        # Find exit price at idx + horizon
        if idx + horizon >= len(df_eval):
            exit_price = row['close']  # Assume closed at end of test period
        else:
            exit_price = df_test['close'].iloc[idx + horizon]
        
        # Calculate gross PnL
        if direction == 1:  # LONG
            gross_return = (exit_price - entry_price) / entry_price
        else:  # SHORT
            gross_return = (entry_price - exit_price) / entry_price
        
        gross_return_pct = gross_return / entry_price
        
        # Apply costs
        net_return = gross_return - row['total_cost_pct'] * entry_price
        net_return_pct = net_return / entry_price
        
        # Update cumulative returns
        df_test.loc[df_test.index[idx], 'gross_return'] = gross_return_pct
        df_test.loc[df_test.index[idx], 'net_return'] = net_return_pct
        
        # Track trade
        trades.append({
            'entry_idx': idx,
            'entry_time': row.name,  # Assuming timestamp is index
            'entry_price': entry_price,
            'direction': direction,
            'exit_price': exit_price,
            'gross_return_pct': gross_return_pct,
            'net_return_pct': net_return_pct,
            'cost_pct': row['total_cost_pct'],
        })
        
        # Update balance
        if direction == 1:  # LONG
            current_balance += net_return * ACCOUNT_SETTINGS['fixed_position_size'] * ACCOUNT_SETTINGS['initial_balance']
        else:  # SHORT
            current_balance += net_return * ACCOUNT_SETTINGS['fixed_position_size'] * ACCOUNT_SETTINGS['initial_balance']
    
    # Create trades DataFrame
    df_trades = pd.DataFrame(trades)
    
    # Calculate metrics
    total_gross = df_test['gross_return'].sum()
    total_net = df_test['net_return'].sum()
    total_costs = df_test['cost_pct'].sum()
    final_balance = current_balance
    total_return = (final_balance - ACCOUNT_SETTINGS['initial_balance']) / ACCOUNT_SETTINGS['initial_balance']
    total_return_pct = total_return * 100
    
    n_wins = len(df_trades[df_trades['net_return'] > 0])
    n_losses = len(df_trades[df_trades['net_return'] <= 0])
    win_rate = n_wins / len(df_trades) if len(df_trades) > 0 else 0
    
    # Calculate max drawdown
    df_test['balance'] = ACCOUNT_SETTINGS['initial_balance'] + (df_test['net_return'] * ACCOUNT_SETTINGS['fixed_position_size'] * ACCOUNT_SETTINGS['initial_balance']).cumsum()
    peak = df_test['balance'].cummax()
    trough = peak * (1 - ACCOUNT_SETTINGS['max_drawdown_pct'])
    max_dd = (peak - trough) / peak
    
    # Calculate Sharpe ratio
    # Daily returns (assuming daily)
    daily_returns = df_test['net_return'].resample('1D').mean() / 252  # ~252 trading days per year
    sharpe = daily_returns.mean() / (daily_returns.std() + 1e-8) * (252**0.5)  # Annualized Sharpe
    
    profit_factor = abs(total_gross) / abs(total_net) if total_net != 0 else float('inf')
    
    print(f"Backtest completed!")
    print(f"  Trades: {n_wins} wins, {n_losses} losses")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Total Return: {total_return_pct:.2f}%")
    print(f"  Max Drawdown: {max_dd:.1f}%")
    print(f"  Final Balance: ${current_balance:,.2f}")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    print(f"  Profit Factor: {profit_factor:.2f}")
    
    # Check success criteria
    passed = (
        win_rate >= SUCCESS_CRITERIA['min_win_rate']
        and max_dd <= SUCCESS_CRITERIA['max_drawdown']
    )
    
    return BacktestResult(
        model_name=model_path.name,
        win_rate=win_rate,
        max_drawdown=max_dd,
        total_return=total_return_pct,
        n_trades=len(df_trades),
        final_balance=final_balance,
        sharpe_ratio=sharpe,
        profit_factor=profit_factor,
        trades=df_trades,
        passed=passed,
    )


def main():
    """Run evaluation on all models."""
    print("=" * 80)
    print("FUNDED BACKTEST ANALYSIS")
    print("=" * 80)
    print(f"\nTest Period: 2024-01-01 to 2025-12-31")
    print(f"\nAccount Settings:")
    print(f"  Initial Balance: ${ACCOUNT_SETTINGS['initial_balance']:,.2f}")
    print(f"  Leverage: 1:{ACCOUNT_SETTINGS['leverage']:.0f}")
    print(f"  Position Size: {ACCOUNT_SETTINGS['fixed_position_size']} lots (0.25 contract size)")
    print(f"  Profit Target: {ACCOUNT_SETTINGS['profit_target_pct']*100:.0f}%")
    print(f"  Max Drawdown Limit: {ACCOUNT_SETTINGS['max_drawdown_pct']*100:.0f}%")
    print(f"\nMarket Costs (per Standard Lot):")
    print(f"  Spread: {ACCOUNT_SETTINGS['spread_cost']*100:.2f} pips ($2.50)")
    print(f"  Slippage: {ACCOUNT_SETTINGS['slippage_long']*100:.1f} pips ($1.00)")
    print(f"  Commission: ${ACCOUNT_SETTINGS['spread_cost']*100:.1f} per round-turn ($5.00)")
    print(f"\nSuccess Criteria:")
    print(f"  Win Rate > {SUCCESS_CRITERIA['min_win_rate']*100:.0f}%")
    print(f"  Max Drawdown < {SUCCESS_CRITERIA['max_drawdown']*100:.0f}%")
    
    models_dir = PROJECT_ROOT / "models"
    
    results = {}
    
    # Evaluate each model
    model_paths = [
        ("Model 1 (Original - No High-Confidence Filter)", models_dir / "model1_quality_gate.joblib"),
        ("Model 5 (Range Reversion - Strict)", models_dir / "model5_range_reversion.joblib"),
    ]
    
    for model_name, model_path in model_paths:
        print("\n" + "-" * 80)
        print(f"EVALUATING {model_name}")
        print("-" * 80)
        
        result = run_funded_backtest_direct(
            model_path=model_path,
            test_start="2024-01-01",
            test_end="2025-12-31",
            verbose=False,
        )
        
        if result:
            results[model_name] = {
                'model_path': str(model_path),
                'win_rate': result.win_rate,
                'max_drawdown': result.max_drawdown,
                'total_return': result.total_return,
                'n_trades': result.n_trades,
                'final_balance': result.final_balance,
                'sharpe': result.sharpe_ratio,
                'profit_factor': result.profit_factor,
                'passed': result.passed,
            }
        else:
            results[model_name] = None
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Model':<30} {'Status':<10} {'Win Rate %':<12} {'Max DD %':<12} {'Trades':<10} {'Return %':<10} {'Passed?':<10}")
    print("-" * 80)
    
    for model_name, result in results.items():
        if result:
            status = "  PASSED  " if result.passed else "  FAILED  "
            print(f"  {model_name:<30} {status:<10} {result.win_rate:>7.1f}%     {result.max_drawdown:>7.1f}%    {result.n_trades:>10}   {result.total_return:>8.2f}%    {status:<10}")
        else:
            print(f"  {model_name:<30} ERROR")
    
    # Check if any models pass
    any_passed = any(r['passed'] for r in results.values() if r)
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    if any_passed:
        print("\n✓ At least one model meets Prop Challenge criteria!")
    else:
        print("\n✗ No model meets Prop Challenge criteria")
    
    # Save results to JSON
    report_path = PROJECT_ROOT / "reports" / f"backtest_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert results to serializable dict
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'test_period': "2024-01-01 to 2025-12-31",
        'success_criteria': SUCCESS_CRITERIA,
        'account_settings': ACCOUNT_SETTINGS,
        'models': {},
    }
    
    for model_name, result in results.items():
        if result:
            report_data['models'][model_name] = {
                'model_path': result['model_path'],
                'win_rate': result.win_rate,
                'max_drawdown': result.max_drawdown,
                'total_return': result.total_return,
                'n_trades': result.n_trades,
                'final_balance': result.final_balance,
                'sharpe': result.sharpe_ratio,
                'profit_factor': result.profit_factor,
                'passed': result.passed,
            }
    
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\nEvaluation report saved to: {report_path}")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("\nFor Prop Challenge Success:")
    print("1. Apply high-confidence filters (prob > 0.70) to improve Model 1 win rate")
    print("2. Implement tighter stops and take profits in Model 5 range reversion")
    print("3. Reduce max drawdown limit to 3-4% while maintaining win rate")
    print("4. Consider ensemble approach combining Model 1 (microstructure) + Model 5 (statistical)")
    print("\nModel-Specific Recommendations:")
    print("Model 1: Fix syntax errors, retrain with high-confidence threshold")
    print("Model 3: Ensure strong momentum signals, use regime filters for shorts")
    print("Model 5: Optimize ER and SNR thresholds, add asymmetric profit taking (1.5:1 R/R)")
    print("\nNext Steps:")
    print("1. Run funded backtests on all newly trained models")
    print("2. Generate feature importance analysis")
    print("3. Create ensemble model that combines best signals from Models 1, 3, 5")
    print("4. Validate against out-of-sample data (2026 Q1 onward)")


if __name__ == "__main__":
    main()

