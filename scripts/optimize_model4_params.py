#!/usr/bin/env python3
"""
Optimize Model #4 TP/SL parameters.

Tests different combinations to maximize Profit Factor.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from itertools import product
import joblib

# Import training script functions
from train_and_backtest_model4 import (
    load_5min_data,
    compute_model4_features,
    create_forward_return_labels,
    compute_news_gate_sync,
    split_train_val,
    train_xgboost,
    Model4Backtest,
    MODEL4_FEATURES,
)

print("="*80)
print("MODEL #4 PARAMETER OPTIMIZATION")
print("="*80)

# Load and prepare data (reuse from previous run)
df = load_5min_data(weeks=260)
df = compute_model4_features(df, inplace=True)
df = create_forward_return_labels(df)
df = compute_news_gate_sync(df)
df = df.dropna(subset=MODEL4_FEATURES + ['target', 'news_gate_mult'])

train_df, val_df = split_train_val(df, val_weeks=12)

# Train model once
X_train = train_df[MODEL4_FEATURES]
y_train = train_df['target']
X_val = val_df[MODEL4_FEATURES]
y_val = val_df['target']

model = train_xgboost(X_train, y_train, X_val, y_val)

# Parameter grid
tp_multipliers = [1.0, 1.5, 2.0, 2.5]
sl_multipliers = [0.8, 1.0, 1.2, 1.5]
entry_thresholds = [0.4, 0.5, 0.6]

print(f"\n{'='*80}")
print("TESTING PARAMETER COMBINATIONS")
print(f"{'='*80}")
print(f"TP multipliers: {tp_multipliers}")
print(f"SL multipliers: {sl_multipliers}")
print(f"Entry thresholds: {entry_thresholds}")
print(f"Total combinations: {len(tp_multipliers) * len(sl_multipliers) * len(entry_thresholds)}")
print("")

results = []

for tp_mult, sl_mult, entry_thresh in product(tp_multipliers, sl_multipliers, entry_thresholds):
    backtest = Model4Backtest(
        model,
        atr_target_mult=tp_mult,
        atr_stop_mult=sl_mult,
        entry_threshold=entry_thresh
    )

    trades_df = backtest.run(val_df)

    if len(trades_df) == 0:
        continue

    # Calculate metrics
    n_trades = len(trades_df)
    n_wins = (trades_df['pnl_dollars'] > 0).sum()
    win_rate = n_wins / n_trades

    gross_profit = trades_df[trades_df['pnl_dollars'] > 0]['pnl_dollars'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_dollars'] < 0]['pnl_dollars'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    total_pnl = trades_df['pnl_dollars'].sum()
    total_return = total_pnl / backtest.initial_capital

    results.append({
        'tp_mult': tp_mult,
        'sl_mult': sl_mult,
        'entry_thresh': entry_thresh,
        'rr_ratio': tp_mult / sl_mult,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_return': total_return,
        'total_pnl': total_pnl,
    })

    print(f"TP={tp_mult:.1f}, SL={sl_mult:.1f}, Thresh={entry_thresh:.1f} → "
          f"Trades={n_trades:3d}, WR={win_rate*100:5.1f}%, PF={profit_factor:5.2f}, Return={total_return*100:+6.2f}%")

# Convert to DataFrame and sort
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('profit_factor', ascending=False)

print(f"\n{'='*80}")
print("TOP 10 PARAMETER COMBINATIONS (by Profit Factor)")
print(f"{'='*80}")
print(results_df.head(10).to_string(index=False))

print(f"\n{'='*80}")
print("TOP 10 BY TOTAL RETURN")
print(f"{'='*80}")
results_by_return = results_df.sort_values('total_return', ascending=False)
print(results_by_return.head(10).to_string(index=False))

# Find best that meets criteria
valid = results_df[(results_df['win_rate'] >= 0.52) & (results_df['profit_factor'] >= 1.6)]

if len(valid) > 0:
    print(f"\n{'='*80}")
    print("✅ CONFIGURATIONS MEETING DEPLOYMENT CRITERIA (WR≥52%, PF≥1.6)")
    print(f"{'='*80}")
    print(valid.to_string(index=False))

    best = valid.iloc[0]
    print(f"\n🏆 RECOMMENDED CONFIGURATION:")
    print(f"   TP Multiplier: {best['tp_mult']}")
    print(f"   SL Multiplier: {best['sl_mult']}")
    print(f"   Entry Threshold: {best['entry_thresh']}")
    print(f"   R:R Ratio: {best['rr_ratio']:.2f}:1")
    print(f"   Win Rate: {best['win_rate']*100:.1f}%")
    print(f"   Profit Factor: {best['profit_factor']:.2f}")
    print(f"   Total Return: {best['total_return']*100:+.2f}%")
else:
    print(f"\n⚠️  NO CONFIGURATION MEETS DEPLOYMENT CRITERIA")
    print(f"   Best Profit Factor: {results_df.iloc[0]['profit_factor']:.2f}")
    print(f"   Best Win Rate: {results_df['win_rate'].max()*100:.1f}%")

print(f"\n{'='*80}")
print("COMPLETE")
print(f"{'='*80}")
