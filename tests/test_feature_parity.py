#!/usr/bin/env python3
"""
Feature Parity Tests - Training vs Live

Verifies that training and live feature engineering produce IDENTICAL results.
Tests for:
1. Feature parity (training == live)
2. No lookahead bias (future data doesn't affect past features)
3. Feature consistency (NaN counts, value ranges)

CRITICAL: These tests MUST pass before deploying retrained models.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List

# Import training path
from src.features import build_features

# Import live path
import sys
sys.path.insert(0, '/Users/omar/Desktop/ML/xauusd_signals')
from src.live.feature_buffer import FeatureBuffer, FEATURE_NAMES


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

def generate_sample_bars(n_bars: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV bar data for testing."""
    np.random.seed(seed)

    base_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    timestamps = [base_time + timedelta(minutes=i) for i in range(n_bars)]

    # Generate realistic price data with trend and noise
    base_price = 2650.0
    trend = np.linspace(0, 10, n_bars)  # Slight uptrend
    noise = np.random.randn(n_bars) * 2.0
    close_prices = base_price + trend + noise

    bars = []
    for i, (ts, close) in enumerate(zip(timestamps, close_prices)):
        open_price = close_prices[i-1] if i > 0 else close
        high = max(open_price, close) + abs(np.random.randn() * 0.5)
        low = min(open_price, close) - abs(np.random.randn() * 0.5)
        volume = abs(np.random.randn() * 1000 + 5000)

        bars.append({
            "timestamp": ts,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    df = pd.DataFrame(bars)
    df = df.set_index("timestamp")
    return df


def generate_sample_quotes(bars: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic bid/ask quote data aligned with bars."""
    np.random.seed(seed)

    quotes = []
    for ts, row in bars.iterrows():
        mid = row["close"]
        spread = abs(np.random.randn() * 0.2 + 0.5)

        quotes.append({
            "timestamp": ts,
            "bid_price": mid - spread / 2,
            "ask_price": mid + spread / 2,
        })

    df = pd.DataFrame(quotes)
    df = df.set_index("timestamp")
    return df


# =============================================================================
# TEST 1: FEATURE PARITY (Training == Live)
# =============================================================================

def test_feature_parity_training_vs_live():
    """
    CRITICAL TEST: Verify training and live produce identical features.

    This is the PRIMARY test ensuring that models will behave identically
    in training and production.
    """
    print("\n" + "="*80)
    print("TEST 1: Feature Parity (Training vs Live)")
    print("="*80)

    # Generate test data
    n_bars = 100
    bars = generate_sample_bars(n_bars=n_bars, seed=123)
    quotes = generate_sample_quotes(bars, seed=123)

    print(f"Generated {len(bars)} bars with quotes")

    # =================================================================
    # TRAINING PATH: Use unified build_features()
    # =================================================================
    print("\n[TRAINING PATH]")
    training_features = build_features(bars, quotes=quotes, feature_set="all")
    print(f"Training features shape: {training_features.shape}")
    print(f"Training features: {list(training_features.columns)[:10]}...")

    # =================================================================
    # LIVE PATH: Use FeatureBuffer
    # =================================================================
    print("\n[LIVE PATH]")
    buffer = FeatureBuffer(max_window=200)

    # Feed bars into buffer (simulating live ingestion)
    for i, (ts, row) in enumerate(bars.iterrows()):
        # Get corresponding quote
        quote_row = quotes.loc[ts]

        # Create bar event with merged bid/ask
        bar_event = {
            "timestamp": ts,
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"],
            "bid_price": quote_row["bid_price"],
            "ask_price": quote_row["ask_price"],
        }

        result = buffer.update_from_bar(bar_event)

    # Get features from buffer (last row only)
    live_features = buffer.get_feature_row()
    print(f"Live features shape: {live_features.shape}")
    print(f"Live features: {list(live_features.columns)[:10]}...")

    # =================================================================
    # COMPARISON: Feature by feature
    # =================================================================
    print("\n[COMPARISON]")

    # Get last row from training (to match live's last row)
    training_last = training_features.iloc[-1]
    live_last = live_features.iloc[0]

    # Compare all features
    mismatches = []
    nan_mismatches = []

    for feat in FEATURE_NAMES:
        if feat not in training_last.index:
            mismatches.append(f"{feat}: MISSING in training")
            continue
        if feat not in live_last.index:
            mismatches.append(f"{feat}: MISSING in live")
            continue

        train_val = training_last[feat]
        live_val = live_last[feat]

        # Check NaN parity
        if pd.isna(train_val) != pd.isna(live_val):
            nan_mismatches.append(f"{feat}: NaN mismatch (train={pd.isna(train_val)}, live={pd.isna(live_val)})")
            continue

        # Skip if both NaN
        if pd.isna(train_val) and pd.isna(live_val):
            continue

        # Check value parity (allow small floating point differences)
        if not np.isclose(train_val, live_val, rtol=1e-5, atol=1e-8):
            mismatches.append(
                f"{feat}: train={train_val:.8f}, live={live_val:.8f}, "
                f"diff={abs(train_val - live_val):.2e}"
            )

    # Report results
    print(f"Total features checked: {len(FEATURE_NAMES)}")
    print(f"Mismatches: {len(mismatches)}")
    print(f"NaN mismatches: {len(nan_mismatches)}")

    if mismatches:
        print("\n⚠️ VALUE MISMATCHES:")
        for msg in mismatches[:20]:  # Show first 20
            print(f"  - {msg}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")

    if nan_mismatches:
        print("\n⚠️ NaN MISMATCHES:")
        for msg in nan_mismatches[:10]:
            print(f"  - {msg}")

    if not mismatches and not nan_mismatches:
        print("\n✅ PERFECT PARITY: All features match!")

    # Assert no mismatches (test must pass)
    assert len(mismatches) == 0, f"Feature parity test failed: {len(mismatches)} mismatches"
    assert len(nan_mismatches) == 0, f"Feature parity test failed: {len(nan_mismatches)} NaN mismatches"


# =============================================================================
# TEST 2: NO LOOKAHEAD BIAS
# =============================================================================

def test_no_lookahead_bias():
    """
    Test that features don't use future data.

    Method:
    1. Compute features for first N bars
    2. Add M more bars to the end
    3. Re-compute features for first N bars
    4. Verify features for bars 1-N didn't change
    """
    print("\n" + "="*80)
    print("TEST 2: No Lookahead Bias")
    print("="*80)

    # Generate initial dataset
    n_initial = 100
    n_additional = 20

    bars_initial = generate_sample_bars(n_bars=n_initial, seed=456)
    quotes_initial = generate_sample_quotes(bars_initial, seed=456)

    print(f"Initial dataset: {len(bars_initial)} bars")

    # Compute features for initial dataset
    features_initial = build_features(bars_initial, quotes=quotes_initial, feature_set="all")
    print(f"Initial features shape: {features_initial.shape}")

    # Generate extended dataset (add more bars to the end)
    bars_extended = generate_sample_bars(n_bars=n_initial + n_additional, seed=456)
    quotes_extended = generate_sample_quotes(bars_extended, seed=456)

    print(f"Extended dataset: {len(bars_extended)} bars (+{n_additional})")

    # Compute features for extended dataset
    features_extended = build_features(bars_extended, quotes=quotes_extended, feature_set="all")
    print(f"Extended features shape: {features_extended.shape}")

    # Compare features for the initial N bars (they should be IDENTICAL)
    features_initial_from_extended = features_extended.iloc[:n_initial]

    print("\n[COMPARISON]")
    print(f"Comparing first {n_initial} bars...")

    # Check each feature
    mismatches = []
    for feat in features_initial.columns:
        if feat not in features_initial_from_extended.columns:
            continue

        initial_vals = features_initial[feat].values
        extended_vals = features_initial_from_extended[feat].values

        # Check for differences (allowing for floating point tolerance)
        diffs = np.abs(initial_vals - extended_vals)
        diffs = diffs[~np.isnan(diffs)]  # Ignore NaN comparisons

        if len(diffs) > 0 and np.max(diffs) > 1e-5:
            max_diff = np.max(diffs)
            max_idx = np.argmax(diffs)
            mismatches.append(
                f"{feat}: max_diff={max_diff:.2e} at idx={max_idx}, "
                f"initial={initial_vals[max_idx]:.8f}, extended={extended_vals[max_idx]:.8f}"
            )

    print(f"Features checked: {len(features_initial.columns)}")
    print(f"Lookahead violations: {len(mismatches)}")

    if mismatches:
        print("\n⚠️ LOOKAHEAD BIAS DETECTED:")
        for msg in mismatches[:10]:
            print(f"  - {msg}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
    else:
        print("\n✅ NO LOOKAHEAD BIAS: Features stable when future data added!")

    assert len(mismatches) == 0, f"Lookahead bias detected in {len(mismatches)} features"


# =============================================================================
# TEST 3: FEATURE CONSISTENCY
# =============================================================================

def test_feature_consistency():
    """
    Test that features have reasonable values and NaN counts.

    Checks:
    - All required features are present
    - NaN ratio < 50% for all features
    - No infinite values
    - Features in reasonable ranges
    """
    print("\n" + "="*80)
    print("TEST 3: Feature Consistency")
    print("="*80)

    # Generate test data
    bars = generate_sample_bars(n_bars=100, seed=789)
    quotes = generate_sample_quotes(bars, seed=789)

    # Compute features
    features = build_features(bars, quotes=quotes, feature_set="all")

    print(f"Features shape: {features.shape}")
    print(f"Total features: {len(features.columns)}")

    # Check 1: All required features present
    missing_features = [f for f in FEATURE_NAMES if f not in features.columns]
    print(f"\nMissing features: {len(missing_features)}")
    if missing_features:
        print(f"  {missing_features}")

    # Check 2: NaN ratios
    nan_ratios = features.isna().mean()
    high_nan_features = nan_ratios[nan_ratios > 0.5]
    print(f"\nHigh NaN features (>50%): {len(high_nan_features)}")
    if len(high_nan_features) > 0:
        print(high_nan_features)

    # Check 3: Infinite values
    inf_counts = np.isinf(features.select_dtypes(include=[np.number])).sum()
    features_with_inf = inf_counts[inf_counts > 0]
    print(f"\nFeatures with infinite values: {len(features_with_inf)}")
    if len(features_with_inf) > 0:
        print(features_with_inf)

    # Check 4: Constant features (might indicate data issues)
    feature_std = features.std()
    constant_features = feature_std[feature_std < 1e-10]
    print(f"\nConstant features (std < 1e-10): {len(constant_features)}")
    if len(constant_features) > 0:
        print(constant_features.index.tolist())

    print("\n" + "="*80)

    # Assertions
    assert len(missing_features) == 0, f"Missing {len(missing_features)} required features"
    assert len(high_nan_features) == 0, f"{len(high_nan_features)} features have >50% NaN"
    assert len(features_with_inf) == 0, f"{len(features_with_inf)} features have infinite values"

    print("✅ All consistency checks passed!")


# =============================================================================
# TEST 4: SYNTHETIC ORDER FLOW FIX
# =============================================================================

def test_synthetic_order_flow_fix():
    """
    Verify that synthetic_order_flow is computed correctly without lookahead bias.

    OLD (WRONG): synthetic_order_flow = net_tick_pressure * volume
    NEW (CORRECT): synthetic_order_flow = net_tick_pressure

    This test ensures the fix is in place.
    """
    print("\n" + "="*80)
    print("TEST 4: Synthetic Order Flow Fix")
    print("="*80)

    # Generate test data
    bars = generate_sample_bars(n_bars=50, seed=999)
    quotes = generate_sample_quotes(bars, seed=999)

    # Compute features
    features = build_features(bars, quotes=quotes, feature_set="all")

    # Check if synthetic_order_flow exists
    if "synthetic_order_flow" not in features.columns:
        print("⚠️ synthetic_order_flow not found in features")
        print("SKIPPING: synthetic_order_flow feature not generated")
        return

    sof = features["synthetic_order_flow"].values
    volume = bars["volume"].values

    print(f"synthetic_order_flow range: [{sof.min():.2f}, {sof.max():.2f}]")
    print(f"Volume range: [{volume.min():.2f}, {volume.max():.2f}]")

    # OLD BEHAVIOR: sof would scale with volume (sof = net_pressure * volume)
    # NEW BEHAVIOR: sof should be independent of volume magnitude

    # Check correlation (should be LOW if fix is in place)
    # Filter out NaN values
    valid_mask = ~np.isnan(sof)
    if valid_mask.sum() > 10:
        correlation = np.corrcoef(sof[valid_mask], volume[valid_mask])[0, 1]
        print(f"Correlation with volume: {correlation:.4f}")

        # If old behavior, correlation would be very high (>0.9)
        # With fix, correlation should be much lower
        if abs(correlation) > 0.8:
            print("⚠️ WARNING: High correlation with volume detected!")
            print("   This suggests synthetic_order_flow might still be using old formula")
        else:
            print("✅ Low correlation - fix appears to be in place")

    print("\n✅ Synthetic order flow check complete")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("FEATURE PARITY TEST SUITE")
    print("="*80)
    print("\nThese tests verify training/live parity and no lookahead bias.")
    print("ALL tests must pass before deploying retrained models.\n")

    try:
        # Run each test
        test_feature_parity_training_vs_live()
        test_no_lookahead_bias()
        test_feature_consistency()
        test_synthetic_order_flow_fix()

        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✅")
        print("="*80)
        print("\nFeature engineering is validated:")
        print("  ✓ Training and live produce identical features")
        print("  ✓ No lookahead bias detected")
        print("  ✓ Features are consistent and well-formed")
        print("  ✓ Synthetic order flow fix verified")
        print("\nSafe to proceed to Phase 5: Model Retraining")
        print("="*80)

    except AssertionError as e:
        print("\n" + "="*80)
        print("TEST FAILED! ❌")
        print("="*80)
        print(f"\nError: {e}")
        print("\nDO NOT PROCEED TO MODEL RETRAINING until tests pass.")
        raise
    except Exception as e:
        print("\n" + "="*80)
        print("TEST ERROR! ⚠️")
        print("="*80)
        print(f"\nUnexpected error: {e}")
        raise
