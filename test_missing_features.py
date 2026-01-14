#!/usr/bin/env python3
"""
Test script to verify all missing features are now present.

Specifically checks for:
- mid, spread, spread_pct, close_mid_diff
- momentum_5, momentum_10
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test the fixed feature engineering
from src.features import build_features

print("=" * 80)
print("TESTING MISSING FEATURES FIX")
print("=" * 80)

# Create synthetic test data (500 bars)
np.random.seed(42)
n_bars = 500
base_time = datetime.now()
timestamps = [base_time + timedelta(minutes=i) for i in range(n_bars)]

# Create realistic OHLCV data
close_prices = 2650 + np.cumsum(np.random.randn(n_bars) * 0.5)
high_prices = close_prices + np.abs(np.random.randn(n_bars)) * 2
low_prices = close_prices - np.abs(np.random.randn(n_bars)) * 2
open_prices = close_prices + np.random.randn(n_bars) * 1
volumes = np.abs(np.random.randn(n_bars)) * 1000 + 5000

bars = pd.DataFrame({
    'timestamp': timestamps,
    'open': open_prices,
    'high': high_prices,
    'low': low_prices,
    'close': close_prices,
    'volume': volumes
}).set_index('timestamp')

# Create quotes data (bid/ask)
quotes = pd.DataFrame({
    'timestamp': timestamps,
    'bid_price': close_prices - np.random.rand(n_bars) * 0.5,
    'ask_price': close_prices + np.random.rand(n_bars) * 0.5
}).set_index('timestamp')

print(f"Created test data: {len(bars)} bars, {len(quotes)} quotes")
print(f"Price range: {bars['close'].min():.2f} - {bars['close'].max():.2f}")
print()

# Test feature engineering
try:
    print("Running feature engineering with 'all' feature set...")
    features = build_features(bars, quotes, feature_set="all")

    print()
    print("=" * 80)
    print("✓ SUCCESS: Feature engineering completed without errors!")
    print("=" * 80)
    print()

    # Check for previously missing features
    previously_missing = ['mid', 'spread', 'spread_pct', 'close_mid_diff',
                         'momentum_5', 'momentum_10']

    print("Checking previously missing features:")
    all_present = True
    for feat in previously_missing:
        if feat in features.columns:
            nan_count = features[feat].isna().sum()
            nan_pct = nan_count / len(features) * 100
            print(f"  ✓ {feat:20s}: Present ({nan_count} NaNs = {nan_pct:.1f}%)")
        else:
            print(f"  ✗ {feat:20s}: MISSING!")
            all_present = False

    print()

    if all_present:
        print("=" * 80)
        print("✅ ALL MISSING FEATURES ARE NOW PRESENT!")
        print("=" * 80)
        print()

        # Show sample values
        print("Sample of last 5 rows:")
        print(features.tail(5)[previously_missing])
        print()

        # Check if values are reasonable
        print("Value ranges:")
        for feat in previously_missing:
            min_val = features[feat].min()
            max_val = features[feat].max()
            mean_val = features[feat].mean()
            print(f"  {feat:20s}: min={min_val:8.4f}, max={max_val:8.4f}, mean={mean_val:8.4f}")
    else:
        print("=" * 80)
        print("❌ SOME FEATURES STILL MISSING!")
        print("=" * 80)
        exit(1)

except Exception as e:
    print()
    print("=" * 80)
    print("✗ FAILURE: Feature engineering failed!")
    print("=" * 80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
