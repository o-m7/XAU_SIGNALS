#!/usr/bin/env python3
"""
Complete training and testing pipeline for XAUUSD Signal Engine.

Walk-forward structure:
- Train: 2020-2023
- Validate: 2024-H1
- Test: 2024-H2 to 2025

Outputs comprehensive metrics including R-multiples.
"""

import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import time

print("="*70)
print("XAUUSD Signal Engine - Training & Testing Pipeline")
print("="*70)

# =============================================================================
# 1. LOAD ALL DATA (2020-2025)
# =============================================================================
print("\n[1/5] Loading data from 2020-2025...")
start = time.time()

from src.data_loader import get_combined_dataset_multi_year

years = [2020, 2021, 2022, 2023, 2024, 2025]
df = get_combined_dataset_multi_year(
    minute_dir='Data/ohlcv_minute',
    quotes_dir='Data/quotes',
    years=years,
    symbol='XAUUSD'
)
print(f"Loaded {len(df):,} rows in {time.time()-start:.1f}s")
print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")

# =============================================================================
# 2. BUILD FEATURES
# =============================================================================
print("\n[2/5] Building features...")
start = time.time()

from src.feature_engineering import build_feature_matrix, verify_no_lookahead

df = build_feature_matrix(df)
print(f"Feature matrix: {df.shape} in {time.time()-start:.1f}s")

# Verify no look-ahead bias
from src.config import FEATURE_COLUMNS
available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
verification = verify_no_lookahead(df, available_features)
if verification["warnings"]:
    print("\n⚠️  LOOK-AHEAD WARNINGS:")
    for w in verification["warnings"]:
        print(f"  {w}")
else:
    print("✓ No look-ahead bias detected")

# =============================================================================
# 3. GENERATE LABELS
# =============================================================================
print("\n[3/5] Generating labels...")
start = time.time()

from src.labeling import generate_labels_for_all_horizons, get_label_statistics

df = generate_labels_for_all_horizons(df, verbose=False)
print(f"Labels generated in {time.time()-start:.1f}s")

# Show label statistics
stats = get_label_statistics(df)
print("\nLabel Distribution:")
for label, s in stats.items():
    long_pct = 100 * s['long_count'] / s['count'] if s['count'] > 0 else 0
    short_pct = 100 * s['short_count'] / s['count'] if s['count'] > 0 else 0
    flat_pct = 100 * s['flat_count'] / s['count'] if s['count'] > 0 else 0
    print(f"  {label}: Long={long_pct:.1f}%, Short={short_pct:.1f}%, Flat={flat_pct:.1f}%")

# =============================================================================
# 4. TRAIN MODELS (Walk-Forward)
# =============================================================================
print("\n[4/5] Training models...")
start = time.time()

from src.model_training import train_all_horizon_models

results = train_all_horizon_models(
    df=df,
    model_dir='xauusd_signals/models/',
    train_end='2023-12-31',
    val_end='2024-06-30',
    use_class_weights=True,
    verbose=True
)
print(f"\nTraining completed in {time.time()-start:.1f}s")

# =============================================================================
# 5. TEST ON 2024-H2 TO 2025 (Out-of-Sample)
# =============================================================================
print("\n[5/5] Testing on out-of-sample data (2024-H2 to 2025)...")
start = time.time()

# Load models
models = {}
for h in ['5m', '15m', '30m']:
    data = joblib.load(f'xauusd_signals/models/model_{h}.pkl')
    models[h] = {'model': data['model'], 'features': data['features']}

# Get test data (after validation period)
test_df = df[df.index > '2024-06-30']
print(f"Test period: {test_df.index.min().date()} to {test_df.index.max().date()}")
print(f"Test rows: {len(test_df):,}")

# Horizon parameters
HORIZONS = {
    '5m': {'minutes': 5, 'k1': 1.5, 'k2': 2.0},
    '15m': {'minutes': 15, 'k1': 1.5, 'k2': 2.5},
    '30m': {'minutes': 30, 'k1': 1.5, 'k2': 3.0},
}

THRESHOLD = 0.65
FILTER_PARAMS = {
    'min_sigma': 0.00005,
    'max_sigma': 0.002,
    'max_spread_pct': 0.0005,
    'max_vol_regime': 2.0,
}

for h, params in HORIZONS.items():
    mins = params['minutes']
    k1, k2 = params['k1'], params['k2']
    
    model = models[h]['model']
    features = models[h]['features']
    
    # Get feature matrix
    X = test_df[features].values
    
    # Batch predict
    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=1) - 1  # Convert to -1, 0, 1
    confs = np.max(probs, axis=1)
    
    # Apply confidence threshold
    signals = np.where(confs >= THRESHOLD, preds, 0)
    
    # Apply filters
    sigma = test_df['sigma'].values
    spread_pct = test_df['spread_pct'].values
    vol_regime = test_df['vol_regime'].values if 'vol_regime' in test_df.columns else np.ones(len(test_df))
    
    filter_mask = (
        (sigma >= FILTER_PARAMS['min_sigma']) &
        (sigma <= FILTER_PARAMS['max_sigma']) &
        (spread_pct <= FILTER_PARAMS['max_spread_pct']) &
        (vol_regime <= FILTER_PARAMS['max_vol_regime']) &
        (~np.isnan(sigma)) &
        (~np.isnan(spread_pct))
    )
    
    # Apply filter to signals
    signals = np.where(filter_mask, signals, 0)
    
    # Get signal indices
    signal_mask = signals != 0
    signal_indices = np.where(signal_mask)[0]
    
    # Remove signals too close to end
    valid_indices = signal_indices[signal_indices < len(test_df) - mins - 1]
    
    if len(valid_indices) == 0:
        print(f"\n{h}: No signals after filtering")
        continue
    
    # Simulate trades
    trades = []
    mid = test_df['mid'].values
    high = test_df['high'].values
    low = test_df['low'].values
    
    i = 0
    while i < len(valid_indices):
        idx = valid_indices[i]
        direction = signals[idx]
        
        entry_idx = idx + 1
        if entry_idx >= len(test_df):
            break
            
        entry_price = mid[entry_idx]
        
        # SL/TP based on volatility
        sig = sigma[idx]
        sp = spread_pct[idx]
        
        sl_ret = -k1 * sig
        tp_ret = k2 * sig + sp
        
        if direction == 1:  # Long
            sl_price = entry_price * (1 + sl_ret)
            tp_price = entry_price * (1 + tp_ret)
        else:  # Short
            sl_price = entry_price * (1 - sl_ret)
            tp_price = entry_price * (1 - tp_ret)
        
        # Simulate trade
        exit_price = entry_price
        exit_reason = 'timeout'
        
        for j in range(entry_idx + 1, min(entry_idx + mins + 1, len(test_df))):
            if direction == 1:
                if low[j] <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break
                if high[j] >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break
            else:
                if high[j] >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break
                if low[j] <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break
            exit_price = mid[j]
        
        # P&L
        if direction == 1:
            pnl = (exit_price - entry_price) / entry_price - sp
        else:
            pnl = (entry_price - exit_price) / entry_price - sp
        
        # R-multiple
        risk = abs(sl_ret)
        pnl_R = pnl / risk if risk > 0 else 0
        
        trades.append({
            'direction': direction,
            'pnl': pnl,
            'pnl_R': pnl_R,
            'exit_reason': exit_reason,
            'confidence': confs[idx]
        })
        
        # Skip to after trade
        i += 1
        while i < len(valid_indices) and valid_indices[i] <= idx + mins:
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
    print(f"{h.upper()} HORIZON - OUT-OF-SAMPLE RESULTS")
    print(f"{'='*60}")
    print(f"Total Trades:      {n}")
    print(f"  Long:            {longs.sum()} ({100*longs.sum()/n:.1f}%)")
    print(f"  Short:           {shorts.sum()} ({100*shorts.sum()/n:.1f}%)")
    print(f"\nPERFORMANCE:")
    print(f"  Win Rate:        {wins.mean():.1%}")
    print(f"  Avg Return:      {trades_df['pnl'].mean()*100:.4f}%")
    print(f"  Total Return:    {trades_df['pnl'].sum()*100:.2f}%")
    print(f"  Avg R:           {trades_df['pnl_R'].mean():.3f}R")
    print(f"  Total R:         {trades_df['pnl_R'].sum():.1f}R")
    print(f"  Expectancy:      {trades_df['pnl_R'].mean():.3f}R per trade")
    
    # By direction
    if longs.sum() > 0:
        long_df = trades_df[longs]
        print(f"\nLONG ({len(long_df)}):")
        print(f"  Win Rate:        {(long_df['pnl'] > 0).mean():.1%}")
        print(f"  Avg R:           {long_df['pnl_R'].mean():.3f}R")
    
    if shorts.sum() > 0:
        short_df = trades_df[shorts]
        print(f"\nSHORT ({len(short_df)}):")
        print(f"  Win Rate:        {(short_df['pnl'] > 0).mean():.1%}")
        print(f"  Avg R:           {short_df['pnl_R'].mean():.3f}R")
    
    # Exit reasons
    print(f"\nEXIT REASONS:")
    for reason in ['tp', 'sl', 'timeout']:
        subset = trades_df[trades_df['exit_reason'] == reason]
        if len(subset) > 0:
            print(f"  {reason.upper():8s}: {len(subset):5d} ({100*len(subset)/n:5.1f}%) | "
                  f"WR: {(subset['pnl']>0).mean():.1%} | Avg R: {subset['pnl_R'].mean():+.3f}R")
    
    # Risk metrics
    cum = trades_df['pnl'].cumsum()
    dd = (np.maximum.accumulate(cum) - cum).max()
    print(f"\nRISK:")
    print(f"  Max Drawdown:    {dd*100:.2f}%")
    print(f"  Best Trade:      {trades_df['pnl_R'].max():.2f}R")
    print(f"  Worst Trade:     {trades_df['pnl_R'].min():.2f}R")

print(f"\n{'='*70}")
print(f"Pipeline completed in {time.time()-start:.1f}s")
print("="*70)

