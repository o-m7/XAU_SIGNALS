#!/usr/bin/env python3
"""Fast 2025 out-of-sample test using vectorized operations."""

import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

print("="*70)
print("XAUUSD 2025 OUT-OF-SAMPLE TEST (Vectorized - Fast)")
print("="*70)

# 1. Load data
print("\n[1] Loading 2025 data...")
from src.data_loader import get_combined_dataset
from src.feature_engineering import build_feature_matrix

df = get_combined_dataset(
    'Data/ohlcv_minute/XAUUSD_minute_2025.parquet',
    'Data/quotes/XAUUSD_quotes_2025.parquet'
)
df = build_feature_matrix(df)
print(f"    Rows: {len(df):,}")
print(f"    Range: {df.index.min().date()} to {df.index.max().date()}")

# 2. Load models and predict ALL at once (vectorized)
print("\n[2] Generating signals (vectorized batch prediction)...")
models = {}
for h in ['5m', '15m', '30m']:
    data = joblib.load(f'xauusd_signals/models/model_{h}.pkl')
    models[h] = {'model': data['model'], 'features': data['features']}

# Get features for all rows at once
feature_cols = models['5m']['features']
X = df[feature_cols].values

# Predict all rows at once for each horizon
predictions = {}
for h, m in models.items():
    probs = m['model'].predict_proba(X)  # Vectorized!
    preds = np.argmax(probs, axis=1) - 1  # Convert 0,1,2 to -1,0,1
    max_probs = np.max(probs, axis=1)
    predictions[h] = {'pred': preds, 'conf': max_probs}
    
print("    Done - all signals generated")

# 3. Compute results per horizon
print("\n[3] Computing backtest results...")

HORIZONS = {
    '5m': {'minutes': 5, 'k1': 1.0, 'k2': 1.5},
    '15m': {'minutes': 15, 'k1': 1.0, 'k2': 2.0},
    '30m': {'minutes': 30, 'k1': 1.0, 'k2': 2.5},
}
THRESHOLD = 0.6

for h, params in HORIZONS.items():
    mins = params['minutes']
    k1, k2 = params['k1'], params['k2']
    
    pred = predictions[h]['pred']
    conf = predictions[h]['conf']
    
    # Filter by confidence threshold
    signals = np.where(conf >= THRESHOLD, pred, 0)
    
    # Get signal indices (non-zero signals)
    signal_mask = signals != 0
    signal_indices = np.where(signal_mask)[0]
    
    # Remove signals too close to end
    valid_indices = signal_indices[signal_indices < len(df) - mins - 1]
    
    if len(valid_indices) == 0:
        print(f"\n{h}: No trades with confidence >= {THRESHOLD}")
        continue
    
    # Compute trade results
    trades = []
    mid = df['mid'].values
    sigma = df['sigma'].values
    spread_pct = df['spread_pct'].values
    high = df['high'].values
    low = df['low'].values
    
    i = 0
    while i < len(valid_indices):
        idx = valid_indices[i]
        direction = signals[idx]
        
        entry_idx = idx + 1
        entry_price = mid[entry_idx]
        
        # SL/TP based on volatility at signal time
        sl_ret = -k1 * sigma[idx]
        tp_ret = k2 * sigma[idx] + spread_pct[idx]
        
        if direction == 1:  # Long
            sl_price = entry_price * (1 + sl_ret)
            tp_price = entry_price * (1 + tp_ret)
        else:  # Short
            sl_price = entry_price * (1 - sl_ret)
            tp_price = entry_price * (1 - tp_ret)
        
        # Simulate trade
        exit_idx = entry_idx + 1
        exit_price = entry_price
        exit_reason = 'timeout'
        
        for j in range(entry_idx + 1, min(entry_idx + mins + 1, len(df))):
            if direction == 1:  # Long
                if low[j] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    exit_idx = j
                    break
                if high[j] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    exit_idx = j
                    break
            else:  # Short
                if high[j] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    exit_idx = j
                    break
                if low[j] <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    exit_idx = j
                    break
            exit_idx = j
            exit_price = mid[j]
        
        # P&L
        if direction == 1:
            pnl = (exit_price - entry_price) / entry_price - spread_pct[idx]
        else:
            pnl = (entry_price - exit_price) / entry_price - spread_pct[idx]
        
        # Risk (R)
        risk = abs(sl_ret)
        pnl_R = pnl / risk if risk > 0 else 0
        
        trades.append({
            'direction': direction,
            'pnl': pnl,
            'pnl_R': pnl_R,
            'exit_reason': exit_reason
        })
        
        # Skip to after this trade exits
        i += 1
        while i < len(valid_indices) and valid_indices[i] <= exit_idx:
            i += 1
    
    # Results
    trades_df = pd.DataFrame(trades)
    n = len(trades_df)
    
    if n == 0:
        print(f"\n{h}: No completed trades")
        continue
    
    wins = trades_df['pnl'] > 0
    longs = trades_df['direction'] == 1
    shorts = trades_df['direction'] == -1
    
    print(f"\n{'='*60}")
    print(f"{h.upper()} HORIZON RESULTS")
    print(f"{'='*60}")
    print(f"Total Trades:      {n}")
    print(f"  Long:            {longs.sum()}")
    print(f"  Short:           {shorts.sum()}")
    print(f"\nPERFORMANCE:")
    print(f"  Win Rate:        {wins.mean():.1%}")
    print(f"  Avg Return:      {trades_df['pnl'].mean()*100:.4f}%")
    print(f"  Total Return:    {trades_df['pnl'].sum()*100:.2f}%")
    print(f"  Avg R:           {trades_df['pnl_R'].mean():.2f}R")
    print(f"  Total R:         {trades_df['pnl_R'].sum():.1f}R")
    print(f"  Best Trade:      {trades_df['pnl'].max()*100:.4f}% ({trades_df['pnl_R'].max():.2f}R)")
    print(f"  Worst Trade:     {trades_df['pnl'].min()*100:.4f}% ({trades_df['pnl_R'].min():.2f}R)")
    
    # By direction
    if longs.sum() > 0:
        long_df = trades_df[longs]
        print(f"\nLONG TRADES ({len(long_df)}):")
        print(f"  Win Rate:        {(long_df['pnl'] > 0).mean():.1%}")
        print(f"  Avg R:           {long_df['pnl_R'].mean():.2f}R")
    
    if shorts.sum() > 0:
        short_df = trades_df[shorts]
        print(f"\nSHORT TRADES ({len(short_df)}):")
        print(f"  Win Rate:        {(short_df['pnl'] > 0).mean():.1%}")
        print(f"  Avg R:           {short_df['pnl_R'].mean():.2f}R")
    
    # Exit reasons
    print(f"\nEXIT REASONS:")
    for reason in ['tp', 'sl', 'timeout']:
        cnt = (trades_df['exit_reason'] == reason).sum()
        if cnt > 0:
            subset = trades_df[trades_df['exit_reason'] == reason]
            print(f"  {reason.upper():8s}: {cnt:4d} ({100*cnt/n:.1f}%) | WR: {(subset['pnl']>0).mean():.1%} | Avg R: {subset['pnl_R'].mean():.2f}R")
    
    # Equity stats
    cum = trades_df['pnl'].cumsum()
    dd = (np.maximum.accumulate(cum) - cum).max()
    print(f"\nRISK METRICS:")
    print(f"  Max Drawdown:    {dd*100:.2f}%")
    print(f"  Expectancy:      {trades_df['pnl_R'].mean():.3f}R per trade")

print("\n" + "="*70)
print("✅ COMPLETE")
print("="*70)

