#!/usr/bin/env python3
"""
Test script to verify the regime feature bug fix.

This tests that feature engineering completes without errors,
specifically testing the atr_percentile computation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Test the fixed feature engineering
from src.features import build_features

print("=" * 80)
print("TESTING REGIME FEATURE FIX")
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
    print(f"Features shape: {features.shape}")
    print(f"Number of features: {features.shape[1]}")
    print(f"Number of rows: {features.shape[0]}")
    print()

    # Check for regime features specifically
    regime_features = ['adx', 'adx_slope', 'efficiency_ratio', 'atr_percentile',
                      'regime_trending', 'regime_ranging', 'regime_volatile']

    print("Regime features check:")
    for feat in regime_features:
        if feat in features.columns:
            nan_count = features[feat].isna().sum()
            print(f"  ✓ {feat:20s}: {nan_count} NaNs (out of {len(features)})")
        else:
            print(f"  ✗ {feat:20s}: MISSING!")

    print()
    print("Sample of last 5 rows:")
    print(features.tail(5)[regime_features])

    print()
    print("=" * 80)
    print("TEST PASSED: The bug fix is working correctly!")
    print("=" * 80)

except Exception as e:
    print()
    print("=" * 80)
    print("✗ FAILURE: Feature engineering failed!")
    print("=" * 80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
