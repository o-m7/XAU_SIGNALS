#!/usr/bin/env python3
"""
Simple Funded Backtest Runner for All Models
Direct evaluation without complex wrappers
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Account settings
ACCOUNT_SETTINGS = {
    'initial_balance': 50000.0,
    'leverage': 10.0,
    'fixed_position_size': 0.25,
    'profit_target_pct': 0.10,
    'max_drawdown_pct': 0.04,
}

# Market costs per Standard Lot (100 oz gold)
# Pip value for gold: $10 per pip for 1 standard lot
MARKET_COSTS = {
    'spread_pips': 0.25,  # 0.25 pips = $2.50 per lot
    'slippage_pips': 0.10,  # 0.10 pips = $1.00 per lot
    'commission_per_side': 5.00,  # $5.00 per lot per side
}

# Success criteria
SUCCESS_CRITERIA = {
    'min_win_rate': 0.60,
    'max_drawdown': 0.05,
}


@dataclass
class BacktestResult:
    """Backtest result."""
    model_name: str
    test_period: str
    n_trades: int
    n_long: int
    n_short: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    total_return_pct: float
    max_drawdown_pct: float
    initial_balance: float
    final_balance: float
    cumulative_r_multiple: float
    passed: bool
    trades: pd.DataFrame


def load_minute_data(test_start, test_end):
    """Load minute data for test period."""
    from src.data_loader import load_multi_year_data
    
    years = list(range(int(test_start[:4]), int(test_end[:4]) + 1))
    
    print(f"Loading minute data for {test_start} to {test_end}...")
    df_bars = load_multi_year_data(
        minute_dir=str(PROJECT_ROOT.parent / "Data" / "ohlcv_minute"),
        quotes_dir=str(PROJECT_ROOT.parent / "Data" / "quotes"),
        years=years,
        require_sizes=False  # Don't require quote sizes (not needed for Models 1, 3, 5)
    )
    
    if df_bars is None or len(df_bars) == 0:
        print("ERROR: No data loaded")
        return None
    
    # Filter for test period
    df_test = df_bars[(df_bars.index >= test_start) & (df_bars.index <= test_end)].copy()
    print(f"Loaded {len(df_test):,} bars for test period")
    
    return df_test


def generate_features(df):
    """Generate features for prediction."""
    from src.features_complete import build_complete_features
    
    print("Building features...")
    df_feat = build_complete_features(df)
    return df_feat


def generate_signals(df, model, feature_cols, threshold_long=0.70, threshold_short=0.30):
    """Generate signals from model."""
    print("Generating signals...")
    
    # Predict probabilities
    X = df[feature_cols].values
    proba = model.predict_proba(X)
    
    # Create signals
    signals = pd.Series(0, index=df.index, dtype=int)
    signals[proba[:, 1] >= threshold_long] = 1  # LONG
    signals[proba[:, 1] <= threshold_short] = -1  # SHORT
    
    print(f"Generated {sum(signals != 0):,} signals ({sum(signals == 1):,} long, {sum(signals == -1):,} short)")
    
    return signals, proba


def run_backtest_with_costs(df, signals, model_name):
    """
    Run funded backtest with realistic market costs.
    
    Calculates both Gross and Net PnL.
    """
    print(f"\nRunning funded backtest: {model_name}")
    
    # Account settings
    initial_balance = ACCOUNT_SETTINGS['initial_balance']
    leverage = ACCOUNT_SETTINGS['leverage']
    position_size = ACCOUNT_SETTINGS['fixed_position_size']
    profit_target = ACCOUNT_SETTINGS['profit_target_pct']
    max_dd_limit = ACCOUNT_SETTINGS['max_drawdown_pct']
    
    # Market costs (per standard lot)
    spread_cost = MARKET_COSTS['spread_pips'] * 10  # 0.25 pips * $10/pip = $2.50
    slippage_cost = MARKET_COSTS['slippage_pips'] * 10  # 0.10 pips * $10/pip = $1.00
    commission_cost = MARKET_COSTS['commission_per_side']  # $5.00 per side
    
    # Total entry cost per lot (spread + slippage + commission)
    entry_cost_per_lot = spread_cost + slippage_cost + commission_cost
    total_round_turn_cost = entry_cost_per_lot * 2  # Entry + exit
    
    print(f"\nAccount Settings:")
    print(f"  Initial Balance: ${initial_balance:,.2f}")
    print(f"  Leverage: 1:{leverage:.0f}")
    print(f"  Position Size: {position_size} lots")
    print(f"  Profit Target: {profit_target*100:.0f}%")
    print(f"  Max Drawdown Limit: {max_dd_limit*100:.0f}%")
    
    print(f"\nMarket Costs (per Standard Lot):")
    print(f"  Spread: {MARKET_COSTS['spread_pips']} pips = ${spread_cost:.2f}")
    print(f"  Slippage: {MARKET_COSTS['slippage_pips']} pips = ${slippage_cost:.2f}")
    print(f"  Commission: ${commission_cost:.2f} per side")
    print(f"  Total Entry Cost: ${entry_cost_per_lot:.2f} per lot")
    print(f"  Round-Turn Cost: ${total_round_turn_cost:.2f} per lot")
    
    # Trade tracking
    trades = []
    current_balance = initial_balance
    equity_curve = [initial_balance]
    drawdown_curve = [0.0]
    
    max_bars_in_trade = 15  # 15 minutes
    
    # Find entry points (where signal != 0)
    entry_indices = df[signals != 0].index.tolist()
    
    print(f"\nSimulating {len(entry_indices):,} trading opportunities...")
    
    for i, entry_time in enumerate(tqdm(entry_indices, desc="Processing trades")):
        if i % 1000 == 0:
            pass  # Progress bar handles this
        
        entry_idx = df.index.get_loc(entry_time)
        if entry_idx >= len(df) - max_bars_in_trade:
            continue
        
        signal = signals.loc[entry_time]
        if signal == 0:
            continue
        
        direction = 1 if signal > 0 else -1
        entry_price = df['close'].iloc[entry_idx]
        
        # Calculate entry price with costs
        # LONG: entry + spread/2 + slippage
        # SHORT: entry - spread/2 - slippage
        spread_half = MARKET_COSTS['spread_pips'] * 0.0001 / 2  # Convert to price
        slippage = MARKET_COSTS['slippage_pips'] * 0.0001  # Convert to price
        
        if direction == 1:  # LONG
            entry_price_gross = entry_price
            entry_price_net = entry_price + spread_half + slippage
        else:  # SHORT
            entry_price_gross = entry_price
            entry_price_net = entry_price - spread_half - slippage
        
        # Simulate trade
        exit_idx = entry_idx + max_bars_in_trade
        if exit_idx >= len(df):
            exit_idx = len(df) - 1
        
        exit_price_gross = df['close'].iloc[exit_idx]
        
        # Calculate exit price with costs
        if direction == 1:  # LONG
            exit_price_net = exit_price_gross - spread_half - slippage
        else:  # SHORT
            exit_price_net = exit_price_gross + spread_half + slippage
        
        # Calculate PnL
        if direction == 1:  # LONG
            gross_pnl_pct = (exit_price_gross - entry_price_gross) / entry_price_gross
            net_pnl_pct = (exit_price_net - entry_price_net) / entry_price_net
        else:  # SHORT
            gross_pnl_pct = (entry_price_gross - exit_price_gross) / entry_price_gross
            net_pnl_pct = (entry_price_net - exit_price_net) / entry_price_net
        
        # Calculate PnL in dollars (with leverage)
        gross_pnl_usd = gross_pnl_pct * initial_balance * position_size * leverage
        net_pnl_usd = net_pnl_pct * initial_balance * position_size * leverage
        
        # Subtract commission
        commission_total = commission_cost * 2 * position_size * (initial_balance / 100000)  # Approx
        net_pnl_usd -= commission_total
        
        # Update balance
        current_balance += net_pnl_usd
        
        # Record trade
        trades.append({
            'entry_time': entry_time,
            'exit_time': df.index[exit_idx],
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'entry_price_gross': entry_price_gross,
            'exit_price_gross': exit_price_gross,
            'entry_price_net': entry_price_net,
            'exit_price_net': exit_price_net,
            'gross_pnl_pct': gross_pnl_pct,
            'net_pnl_pct': net_pnl_pct,
            'gross_pnl_usd': gross_pnl_usd,
            'net_pnl_usd': net_pnl_usd,
            'commission_usd': commission_total,
            'bars_held': exit_idx - entry_idx,
        })
        
        # Update equity curve
        equity_curve.append(current_balance)
        peak = max(equity_curve)
        dd = (current_balance - peak) / peak if peak > 0 else 0
        drawdown_curve.append(dd)
        
        # Check drawdown limit
        if dd < -max_dd_limit:
            print(f"\n  ⚠️  Account blown! Max drawdown exceeded: {dd*100:.2f}%")
            break
    
    # Convert to DataFrame
    df_trades = pd.DataFrame(trades)
    
    if len(df_trades) == 0:
        print("  No trades executed!")
        return BacktestResult(
            model_name=model_name,
            test_period="2024-01-01 to 2025-12-31",
            n_trades=0,
            n_long=0,
            n_short=0,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            total_return_pct=0.0,
            max_drawdown_pct=0.0,
            initial_balance=initial_balance,
            final_balance=initial_balance,
            cumulative_r_multiple=0.0,
            passed=False,
            trades=df_trades,
        )
    
    # Calculate metrics
    n_trades = len(df_trades)
    n_long = len(df_trades[df_trades['direction'] == 'LONG'])
    n_short = len(df_trades[df_trades['direction'] == 'SHORT'])
    
    # Win rate (based on net PnL)
    n_wins = len(df_trades[df_trades['net_pnl_usd'] > 0])
    win_rate = n_wins / n_trades if n_trades > 0 else 0
    
    # Profit factor (gross profit / gross loss)
    gross_profit = df_trades[df_trades['net_pnl_usd'] > 0]['net_pnl_usd'].sum()
    gross_loss = abs(df_trades[df_trades['net_pnl_usd'] <= 0]['net_pnl_usd'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Sharpe ratio (annualized)
    returns = df_trades['net_pnl_pct'].values
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Total return
    total_return_pct = (current_balance - initial_balance) / initial_balance * 100
    
    # Max drawdown
    max_drawdown_pct = min(drawdown_curve) * 100
    
    # Cumulative R-multiple
    cumulative_r_multiple = df_trades['net_pnl_usd'].sum() / (commission_total * 2) if commission_total > 0 else 0
    
    # Check success criteria
    passed = (win_rate >= SUCCESS_CRITERIA['min_win_rate'] and 
              max_drawdown_pct <= SUCCESS_CRITERIA['max_drawdown'] * 100)
    
    print(f"\n{'='*80}")
    print(f"BACKTEST RESULTS: {model_name}")
    print(f"{'='*80}")
    print(f"\n📊 Trading Statistics:")
    print(f"  Total Trades: {n_trades}")
    print(f"    - Long Trades: {n_long}")
    print(f"    - Short Trades: {n_short}")
    print(f"  Win Rate: {win_rate*100:.2f}%")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    print(f"  Cumulative R-Multiple: {cumulative_r_multiple:.2f}")
    
    print(f"\n💰 Performance Metrics:")
    print(f"  Initial Balance: ${initial_balance:,.2f}")
    print(f"  Final Balance: ${current_balance:,.2f}")
    print(f"  Total Return: {total_return_pct:+.2f}%")
    print(f"  Max Drawdown: {max_drawdown_pct:.2f}%")
    
    print(f"\n🎯 Prop Challenge Status: {'✓ PASSED' if passed else '✗ FAILED'}")
    
    return BacktestResult(
        model_name=model_name,
        test_period="2024-01-01 to 2025-12-31",
        n_trades=n_trades,
        n_long=n_long,
        n_short=n_short,
        win_rate=win_rate,
        profit_factor=profit_factor,
        sharpe_ratio=sharpe,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_drawdown_pct,
        initial_balance=initial_balance,
        final_balance=current_balance,
        cumulative_r_multiple=cumulative_r_multiple,
        passed=passed,
        trades=df_trades,
    )


def main():
    """Run backtests on all models."""
    print("="*80)
    print("FUNDED BACKTEST ANALYSIS - ALL MODELS")
    print("="*80)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Period: 2024-01-01 to 2025-12-31")
    
    models_dir = PROJECT_ROOT / "models"
    
    # Load data once for all models
    df = load_minute_data("2024-01-01", "2025-12-31")
    if df is None:
        print("ERROR: Could not load data")
        return
    
    # Add features
    df_feat = generate_features(df)
    if df_feat is None:
        print("ERROR: Could not generate features")
        return
    
    # Models to test
    models_to_test = [
        ("Model 1 (High Conf)", models_dir / "model1_high_conf.joblib", 0.70, 0.30),
        ("Model 3 (Strict CMF/MACD)", models_dir / "model3_cmf_macd_strict.joblib", 0.70, 0.30),
        ("Model 5 (Strict Range Reversion)", models_dir / "model5_strict_range_reversion.joblib", 0.70, 0.30),
        ("Model 6 (Order Flow)", models_dir / "model6_orderflow" / "model6_orderflow.joblib", 0.70, 0.30),
    ]
    
    results = {}
    
    for model_name, model_path, threshold_long, threshold_short in models_to_test:
        print(f"\n{'#'*80}")
        print(f"# {model_name}")
        print(f"{'#'*80}")
        
        if not model_path.exists():
            print(f"\n⚠️  Model file not found: {model_path}")
            results[model_name] = None
            continue
        
        # Load model
        try:
            artifact = joblib.load(model_path)
            model = artifact['model']
            feature_cols = artifact['features']
            
            print(f"✓ Model loaded: {len(feature_cols)} features")
        except Exception as e:
            print(f"ERROR: Could not load model: {e}")
            results[model_name] = None
            continue
        
        # Generate signals
        signals, proba = generate_signals(df_feat, model, feature_cols, threshold_long, threshold_short)
        
        # Run backtest
        result = run_backtest_with_costs(df, signals, model_name)
        results[model_name] = result
    
    # Summary Report
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    print(f"\n{'Model':<40} {'Trades':>8} {'Long':>6} {'Short':>6} {'Win %':>8} {'PF':>6} {'Sharpe':>7} {'Ret %':>8} {'DD %':>8} {'R-Mult':>8} {'Status':<12}")
    print("-"*130)
    
    for model_name, result in results.items():
        if result:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"{model_name:<40} {result.n_trades:>8} {result.n_long:>6} {result.n_short:>6} {result.win_rate*100:>7.1f}% {result.profit_factor:>5.2f} {result.sharpe_ratio:>7.2f} {result.total_return_pct:>7.2f}% {result.max_drawdown_pct:>7.2f}% {result.cumulative_r_multiple:>7.2f} {status:<12}")
        else:
            print(f"{model_name:<40} ERROR")
    
    # Overall conclusion
    print("\n" + "="*80)
    print("OVERALL CONCLUSION")
    print("="*80)
    
    passed_models = [name for name, r in results.items() if r and r.passed]
    failed_models = [name for name, r in results.items() if r and not r.passed]
    
    if passed_models:
        print(f"\n✓ {len(passed_models)} model(s) meet Prop Challenge criteria:")
        for model in passed_models:
            result = results[model]
            print(f"    • {model}: {result.win_rate*100:.1f}% win rate, {result.max_drawdown_pct:.1f}% max DD")
    else:
        print("\n✗ No models meet Prop Challenge criteria")
    
    if failed_models:
        print(f"\n✗ {len(failed_models)} model(s) failed criteria:")
        for model in failed_models:
            result = results[model]
            fail_reason = []
            if result.win_rate < SUCCESS_CRITERIA['min_win_rate']:
                fail_reason.append(f"Win rate too low ({result.win_rate*100:.1f}% < {SUCCESS_CRITERIA['min_win_rate']*100:.0f}%)")
            if result.max_drawdown_pct > SUCCESS_CRITERIA['max_drawdown'] * 100:
                fail_reason.append(f"Max drawdown too high ({result.max_drawdown_pct:.1f}% > {SUCCESS_CRITERIA['max_drawdown']*100:.0f}%)")
            print(f"    • {model}: {'; '.join(fail_reason)}")
    
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    
    for model_name, result in results.items():
        if result:
            print(f"\n{'-'*80}")
            print(f"{model_name}")
            print(f"{'-'*80}")
            print(f"\nTrading Statistics:")
            print(f"  Total Trades: {result.n_trades}")
            print(f"    - Long Trades: {result.n_long}")
            print(f"    - Short Trades: {result.n_short}")
            print(f"  Win Rate: {result.win_rate*100:.2f}%")
            print(f"  Profit Factor: {result.profit_factor:.2f}")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.3f}")
            print(f"  Cumulative R-Multiple: {result.cumulative_r_multiple:.2f}")
            
            print(f"\nPerformance Metrics:")
            print(f"  Initial Balance: ${result.initial_balance:,.2f}")
            print(f"  Final Balance: ${result.final_balance:,.2f}")
            print(f"  Total Return: {result.total_return_pct:+.2f}%")
            print(f"  Max Drawdown: {result.max_drawdown_pct:.2f}%")
            
            print(f"\nProp Challenge Status: {'✓ PASSED' if result.passed else '✗ FAILED'}")


if __name__ == "__main__":
    main()

