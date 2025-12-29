#!/usr/bin/env python3
"""
Diagnostic script to check if features are updating correctly.

This script helps diagnose why P(up) might be stuck at 0.4-0.5
even when the market moves significantly.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from live.feature_buffer import FeatureBuffer
from live.signal_engine import SignalEngine
from datetime import datetime, timezone, timedelta

def test_feature_updates():
    """Test if features update when price changes."""
    
    print("=" * 60)
    print("FEATURE UPDATE DIAGNOSTIC")
    print("=" * 60)
    
    # Initialize buffer
    buffer = FeatureBuffer(max_window=200)
    
    # Initialize signal engine
    model_path = Path(__file__).parent / "models" / "y_tb_60_hgb_tuned.joblib"
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    engine = SignalEngine(
        model_path=str(model_path),
        threshold_long=0.60,
        threshold_short=0.40
    )
    
    # Simulate price crash: start at 2650, crash to 2430 (8% drop)
    base_time = datetime.now(timezone.utc)
    prices = [2650.0]
    
    # Generate 100 bars with gradual crash
    for i in range(100):
        # Crash from 2650 to 2430 over 100 bars
        price = 2650.0 - (2650.0 - 2430.0) * (i / 100.0)
        prices.append(price)
    
    print(f"\nSimulating {len(prices)} bars:")
    print(f"  Start price: {prices[0]:.2f}")
    print(f"  End price: {prices[-1]:.2f}")
    print(f"  Total drop: {(prices[0] - prices[-1]) / prices[0] * 100:.2f}%")
    
    # Feed bars to buffer
    proba_history = []
    feature_history = []
    
    for i, price in enumerate(prices):
        timestamp = base_time + timedelta(minutes=i)
        
        # Create bar event
        bar = {
            "type": "bar",
            "timestamp": timestamp,
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 1000,
        }
        
        feature_row = buffer.update_from_bar(bar)
        
        if feature_row is not None and buffer.is_ready():
            # Get prediction
            result = engine.generate_signal(feature_row, price)
            proba_up = result.get("proba_up", 0.5)
            proba_history.append(proba_up)
            
            # Store key features
            key_feats = ['ret_1', 'ret_5', 'ret_10', 'vol_10', 'ATR_14']
            feat_vals = {f: feature_row[f].iloc[0] if f in feature_row.columns else np.nan 
                        for f in key_feats}
            feat_vals['price'] = price
            feat_vals['proba_up'] = proba_up
            feature_history.append(feat_vals)
            
            # Log every 10 bars
            if len(proba_history) % 10 == 0:
                print(f"\nBar {len(proba_history)}: Price={price:.2f}, P(up)={proba_up:.4f}")
                print(f"  Features: ret_1={feat_vals.get('ret_1', 'N/A'):.6f}, "
                      f"ret_5={feat_vals.get('ret_5', 'N/A'):.6f}, "
                      f"ATR={feat_vals.get('ATR_14', 'N/A'):.2f}")
    
    # Analyze results
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    
    if len(proba_history) < 2:
        print("❌ Not enough data collected")
        return
    
    proba_array = np.array(proba_history)
    proba_std = proba_array.std()
    proba_range = proba_array.max() - proba_array.min()
    
    print(f"\nP(up) Statistics:")
    print(f"  Mean: {proba_array.mean():.4f}")
    print(f"  Std: {proba_std:.4f}")
    print(f"  Min: {proba_array.min():.4f}")
    print(f"  Max: {proba_array.max():.4f}")
    print(f"  Range: {proba_range:.4f}")
    
    if proba_std < 0.01:
        print("\n⚠️  WARNING: P(up) has very low variance (< 0.01)")
        print("   This suggests features are NOT updating properly!")
    elif proba_range < 0.05:
        print("\n⚠️  WARNING: P(up) range is very small (< 0.05)")
        print("   Features may be updating but model predictions are stuck")
    else:
        print("\n✅ P(up) is varying - features appear to be updating")
    
    # Check feature changes
    if len(feature_history) >= 2:
        print(f"\nFeature Changes:")
        first = feature_history[0]
        last = feature_history[-1]
        
        for feat in ['ret_1', 'ret_5', 'ret_10', 'ATR_14']:
            if feat in first and feat in last:
                first_val = first[feat]
                last_val = last[feat]
                if not (np.isnan(first_val) or np.isnan(last_val)):
                    change = abs(last_val - first_val)
                    print(f"  {feat}: {first_val:.6f} → {last_val:.6f} (Δ={change:.6f})")
        
        price_change = abs(last['price'] - first['price'])
        proba_change = abs(last['proba_up'] - first['proba_up'])
        print(f"\nPrice change: {price_change:.2f} ({price_change/first['price']*100:.2f}%)")
        print(f"P(up) change: {proba_change:.4f} ({proba_change*100:.2f}%)")
        
        if price_change > 50 and proba_change < 0.05:
            print("\n❌ CRITICAL: Price moved significantly but P(up) barely changed!")
            print("   This indicates features are NOT responding to price changes")
            print("   Possible causes:")
            print("   1. Features are stale/cached")
            print("   2. Feature calculation has bugs")
            print("   3. Model is seeing same feature values repeatedly")

if __name__ == "__main__":
    test_feature_updates()
