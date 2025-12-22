"""
Backtesting module for the XAUUSD Signal Engine.

Dr. Chen Style Implementation:
==============================
This backtest evaluates the TWO-STAGE signal system:
1. Environment classification quality (is the model identifying good setups?)
2. Direction accuracy (is microstructure correctly determining direction?)

The backtest separately tracks:
- Environment accuracy: When model says "good env", was there movement?
- Direction accuracy: When we took a direction, was it correct?
- Combined performance: Overall P&L

This separation helps debug where the system is failing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .signal_generator import (
    Signal, HORIZON_PARAMS, FILTER_PARAMS,
    load_environment_models, batch_generate_signals
)


def vectorized_backtest_environment(
    df: pd.DataFrame,
    signals_df: pd.DataFrame,
    horizon: str,
    horizon_minutes: int,
    k1: float,
    k2: float
) -> pd.DataFrame:
    """
    Vectorized backtest that evaluates environment + direction separately.
    
    Returns DataFrame with trade results including:
    - Environment prediction quality
    - Direction accuracy
    - P&L
    """
    # Get arrays
    mid = df["mid"].values
    sigma = df["sigma"].values if "sigma" in df.columns else np.ones(len(df)) * 0.0001
    spread_pct = df["spread_pct"].values
    
    env_score = signals_df[f"env_score_{horizon}"].values
    env_good = signals_df[f"env_good_{horizon}"].values
    signal = signals_df[f"signal_{horizon}"].values
    direction = signals_df["direction"].values
    filter_passed = signals_df["filter_passed"].values
    
    n = len(df)
    
    # Pre-allocate results
    results = {
        "timestamp": [],
        "mid": [],
        "sigma": [],
        "spread_pct": [],
        "env_score": [],
        "env_good": [],
        "signal": [],
        "direction": [],
        "exit_price": [],
        "actual_return": [],
        "hit_tp": [],
        "hit_sl": [],
        "timed_out": [],
        "env_was_correct": [],  # Did price actually move?
        "direction_was_correct": [],  # Was direction right?
        "pnl_ret": [],
        "r_multiple": [],
    }
    
    valid_end = n - horizon_minutes - 1
    
    for i in range(valid_end):
        if signal[i] == 0:
            continue
        
        entry_mid = mid[i]
        entry_sigma = sigma[i]
        entry_spread_pct = spread_pct[i]
        
        if np.isnan(entry_mid) or np.isnan(entry_sigma) or entry_sigma <= 0:
            continue
        
        # Entry at next bar's mid (no look-ahead)
        if i + 1 >= n:
            continue
        entry_price = mid[i + 1]
        if np.isnan(entry_price):
            continue
        
        # Compute SL/TP
        sl_ret = k1 * entry_sigma
        tp_ret = k2 * entry_sigma + entry_spread_pct
        
        trade_direction = signal[i]
        
        if trade_direction == 1:  # LONG
            sl_price = entry_price * (1 - sl_ret)
            tp_price = entry_price * (1 + tp_ret)
        else:  # SHORT
            sl_price = entry_price * (1 + sl_ret)
            tp_price = entry_price * (1 - tp_ret)
        
        # Simulate forward path
        exit_price = None
        hit_tp = False
        hit_sl = False
        timed_out = False
        
        for j in range(i + 2, min(i + 2 + horizon_minutes, n)):
            future_mid = mid[j]
            if np.isnan(future_mid):
                continue
            
            if trade_direction == 1:  # LONG
                if future_mid <= sl_price:
                    exit_price = sl_price
                    hit_sl = True
                    break
                elif future_mid >= tp_price:
                    exit_price = tp_price
                    hit_tp = True
                    break
            else:  # SHORT
                if future_mid >= sl_price:
                    exit_price = sl_price
                    hit_sl = True
                    break
                elif future_mid <= tp_price:
                    exit_price = tp_price
                    hit_tp = True
                    break
        
        # Timeout: exit at horizon end
        if exit_price is None:
            exit_idx = min(i + 1 + horizon_minutes, n - 1)
            exit_price = mid[exit_idx]
            timed_out = True
        
        if np.isnan(exit_price):
            continue
        
        # Calculate returns
        if trade_direction == 1:  # LONG
            actual_return = (exit_price / entry_price) - 1.0
        else:  # SHORT
            actual_return = (entry_price / exit_price) - 1.0
        
        # Net return after spread cost
        pnl_ret = actual_return - entry_spread_pct
        
        # R-multiple (return relative to risk)
        r_multiple = pnl_ret / sl_ret if sl_ret > 0 else 0
        
        # Was environment actually good? (did price move significantly?)
        forward_returns = (mid[i+1:i+1+horizon_minutes] / entry_mid) - 1.0
        forward_returns = forward_returns[~np.isnan(forward_returns)]
        if len(forward_returns) > 0:
            max_move = max(np.max(forward_returns), abs(np.min(forward_returns)))
            env_was_correct = max_move > (entry_spread_pct * 1.5)  # Movement exceeded cost
        else:
            env_was_correct = False
        
        # Was direction correct?
        direction_was_correct = pnl_ret > 0
        
        # Store results
        results["timestamp"].append(df.index[i])
        results["mid"].append(entry_mid)
        results["sigma"].append(entry_sigma)
        results["spread_pct"].append(entry_spread_pct)
        results["env_score"].append(env_score[i])
        results["env_good"].append(env_good[i])
        results["signal"].append(trade_direction)
        results["direction"].append(direction[i])
        results["exit_price"].append(exit_price)
        results["actual_return"].append(actual_return)
        results["hit_tp"].append(hit_tp)
        results["hit_sl"].append(hit_sl)
        results["timed_out"].append(timed_out)
        results["env_was_correct"].append(env_was_correct)
        results["direction_was_correct"].append(direction_was_correct)
        results["pnl_ret"].append(pnl_ret)
        results["r_multiple"].append(r_multiple)
    
    return pd.DataFrame(results)


def compute_backtest_metrics(trades_df: pd.DataFrame) -> Dict:
    """
    Compute comprehensive backtest metrics separating environment and direction.
    """
    if len(trades_df) == 0:
        return {
            "n_trades": 0,
            "error": "No trades generated"
        }
    
    n_trades = len(trades_df)
    
    # Basic stats
    total_return = trades_df["pnl_ret"].sum()
    avg_return = trades_df["pnl_ret"].mean()
    std_return = trades_df["pnl_ret"].std()
    
    # Win/loss
    winners = trades_df["pnl_ret"] > 0
    win_rate = winners.mean()
    
    # R-multiples
    total_r = trades_df["r_multiple"].sum()
    avg_r = trades_df["r_multiple"].mean()
    
    # Direction breakdown
    long_trades = trades_df[trades_df["signal"] == 1]
    short_trades = trades_df[trades_df["signal"] == -1]
    
    long_count = len(long_trades)
    short_count = len(short_trades)
    
    long_win_rate = (long_trades["pnl_ret"] > 0).mean() if long_count > 0 else 0
    short_win_rate = (short_trades["pnl_ret"] > 0).mean() if short_count > 0 else 0
    
    long_avg_r = long_trades["r_multiple"].mean() if long_count > 0 else 0
    short_avg_r = short_trades["r_multiple"].mean() if short_count > 0 else 0
    
    # Exit type breakdown
    hit_tp_rate = trades_df["hit_tp"].mean()
    hit_sl_rate = trades_df["hit_sl"].mean()
    timeout_rate = trades_df["timed_out"].mean()
    
    # Environment accuracy
    env_accuracy = trades_df["env_was_correct"].mean()
    
    # Direction accuracy
    direction_accuracy = trades_df["direction_was_correct"].mean()
    
    # Sharpe (annualized assuming minute bars)
    if std_return > 0:
        sharpe = (avg_return / std_return) * np.sqrt(252 * 1440)
    else:
        sharpe = 0
    
    # Expectancy
    if win_rate > 0 and win_rate < 1:
        avg_win = trades_df.loc[winners, "pnl_ret"].mean()
        avg_loss = abs(trades_df.loc[~winners, "pnl_ret"].mean())
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    else:
        avg_win = avg_loss = expectancy = 0
    
    # Max drawdown
    cumulative = trades_df["pnl_ret"].cumsum()
    running_max = cumulative.cummax()
    drawdowns = running_max - cumulative
    max_drawdown = drawdowns.max()
    
    return {
        # Overall
        "n_trades": n_trades,
        "total_return_pct": total_return * 100,
        "avg_return_pct": avg_return * 100,
        "win_rate": win_rate,
        
        # R-multiples
        "total_r": total_r,
        "avg_r": avg_r,
        "expectancy_r": avg_r * n_trades,
        
        # Direction balance (CRITICAL - should be closer to 50/50)
        "long_count": long_count,
        "short_count": short_count,
        "long_pct": long_count / n_trades * 100 if n_trades > 0 else 0,
        "short_pct": short_count / n_trades * 100 if n_trades > 0 else 0,
        
        # Direction-specific performance
        "long_win_rate": long_win_rate,
        "short_win_rate": short_win_rate,
        "long_avg_r": long_avg_r,
        "short_avg_r": short_avg_r,
        
        # Exit types
        "hit_tp_rate": hit_tp_rate,
        "hit_sl_rate": hit_sl_rate,
        "timeout_rate": timeout_rate,
        
        # Accuracy decomposition (CRITICAL for debugging)
        "env_accuracy": env_accuracy,  # Was environment actually tradeable?
        "direction_accuracy": direction_accuracy,  # Was direction correct?
        
        # Risk metrics
        "sharpe": sharpe,
        "avg_win_pct": avg_win * 100 if winners.any() else 0,
        "avg_loss_pct": avg_loss * 100 if (~winners).any() else 0,
        "expectancy_pct": expectancy * 100,
        "max_drawdown_pct": max_drawdown * 100,
    }


def run_backtest_all_horizons(
    df: pd.DataFrame,
    model_dir: str,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Run backtest for all horizons using two-stage signal generation.
    """
    # Generate signals for all data
    if verbose:
        print("Generating signals (two-stage: environment → direction)...")
    
    signals_df = batch_generate_signals(df, model_dir)
    
    # Merge with original data
    full_df = pd.concat([df, signals_df], axis=1)
    
    results = {}
    
    for horizon in ["5m", "15m", "30m"]:
        horizon_minutes = int(horizon.replace("m", ""))
        hp = HORIZON_PARAMS.get(horizon, {"k1": 1.5, "k2": 2.0})
        
        if verbose:
            print(f"\nBacktesting {horizon} horizon...")
        
        # Run backtest
        trades = vectorized_backtest_environment(
            df=full_df,
            signals_df=signals_df,
            horizon=horizon,
            horizon_minutes=horizon_minutes,
            k1=hp["k1"],
            k2=hp["k2"]
        )
        
        # Compute metrics
        metrics = compute_backtest_metrics(trades)
        
        results[horizon] = {
            "trades": trades,
            "metrics": metrics,
        }
        
        if verbose:
            m = metrics
            print(f"  Trades: {m['n_trades']:,}")
            print(f"  Long/Short: {m['long_count']:,} / {m['short_count']:,} "
                  f"({m['long_pct']:.1f}% / {m['short_pct']:.1f}%)")
            print(f"  Win Rate: {m['win_rate']:.1%}")
            print(f"  Avg R: {m['avg_r']:.3f}")
            print(f"  Total R: {m['total_r']:.2f}")
            print(f"  Direction Accuracy: {m['direction_accuracy']:.1%}")
            print(f"  Environment Accuracy: {m['env_accuracy']:.1%}")
    
    return results


def format_backtest_report(results: Dict[str, Dict]) -> str:
    """Format comprehensive backtest report."""
    lines = [
        "=" * 70,
        "BACKTEST REPORT - Dr. Chen Environment-Based System",
        "=" * 70,
        "",
        "System Design:",
        "  - Model predicts ENVIRONMENT quality (tradeable vs not)",
        "  - Direction is determined by MICROSTRUCTURE (imbalance, momentum)",
        "  - This separation prevents directional collapse",
        "",
    ]
    
    for horizon, data in results.items():
        m = data["metrics"]
        
        lines.extend([
            f"{'='*70}",
            f"HORIZON: {horizon}",
            f"{'='*70}",
            "",
            f"TRADE COUNTS:",
            f"  Total Trades:     {m['n_trades']:,}",
            f"  Long Trades:      {m['long_count']:,} ({m['long_pct']:.1f}%)",
            f"  Short Trades:     {m['short_count']:,} ({m['short_pct']:.1f}%)",
            f"  ← Balance Check: {'✓ BALANCED' if 30 < m['long_pct'] < 70 else '⚠ IMBALANCED'}",
            "",
            f"PERFORMANCE:",
            f"  Win Rate:         {m['win_rate']:.1%}",
            f"  Long Win Rate:    {m['long_win_rate']:.1%}",
            f"  Short Win Rate:   {m['short_win_rate']:.1%}",
            "",
            f"R-MULTIPLES (Risk-Adjusted):",
            f"  Average R:        {m['avg_r']:.3f}",
            f"  Total R:          {m['total_r']:.2f}",
            f"  Long Avg R:       {m['long_avg_r']:.3f}",
            f"  Short Avg R:      {m['short_avg_r']:.3f}",
            "",
            f"SYSTEM ACCURACY:",
            f"  Environment:      {m['env_accuracy']:.1%} (did price actually move?)",
            f"  Direction:        {m['direction_accuracy']:.1%} (was direction correct?)",
            "",
            f"EXIT ANALYSIS:",
            f"  Hit TP:           {m['hit_tp_rate']:.1%}",
            f"  Hit SL:           {m['hit_sl_rate']:.1%}",
            f"  Timeout:          {m['timeout_rate']:.1%}",
            "",
            f"RISK METRICS:",
            f"  Sharpe Ratio:     {m['sharpe']:.2f}",
            f"  Max Drawdown:     {m['max_drawdown_pct']:.2f}%",
            f"  Avg Win:          {m['avg_win_pct']:.3f}%",
            f"  Avg Loss:         {m['avg_loss_pct']:.3f}%",
            f"  Expectancy:       {m['expectancy_pct']:.4f}%",
            "",
        ])
    
    # Summary
    lines.extend([
        "=" * 70,
        "SUMMARY",
        "=" * 70,
    ])
    
    for horizon, data in results.items():
        m = data["metrics"]
        status = "✓" if m['avg_r'] > 0 and 30 < m['long_pct'] < 70 else "⚠"
        lines.append(
            f"  {horizon}: {status} Total R = {m['total_r']:+.2f}, "
            f"WR = {m['win_rate']:.1%}, "
            f"Balance = {m['long_pct']:.0f}%L / {m['short_pct']:.0f}%S"
        )
    
    lines.append("")
    
    return "\n".join(lines)


# Legacy compatibility
def run_backtest_for_horizon(*args, **kwargs):
    """Legacy function - use run_backtest_all_horizons instead."""
    raise NotImplementedError(
        "Use run_backtest_all_horizons() with the new two-stage system"
    )
