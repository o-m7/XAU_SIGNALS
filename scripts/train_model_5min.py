#!/usr/bin/env python3
"""
Train 5-Minute Short-Term Model.

This model predicts short-term price direction over a 5-minute horizon.
Uses tighter ATR multipliers suitable for scalping/short-term trading.

Labels:
- +1: Price hits TP (entry + 0.5*ATR) before SL within 5 bars
- -1: Price hits SL (entry - 0.5*ATR) before TP within 5 bars
- 0: Neither hit (filtered out for training)

Usage:
    python scripts/train_model_5min.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import TimeSeriesSplit

# Data path (pre-computed features)
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"

print("=" * 80)
print("TRAIN 5-MINUTE SHORT-TERM MODEL")
print("=" * 80)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Model settings
MODEL_NAME = "model_5min_scalp_v2"
MODEL_PATH = PROJECT_ROOT / "models" / f"{MODEL_NAME}.joblib"

# Label parameters (ASYMMETRIC to balance classes)
HORIZON = 5  # 5 bars (5 minutes)

# LONG-biased asymmetry to counteract bearish data bias:
# - Smaller upside target (easier to hit) + wider downside stop (harder to stop out)
# - This increases LONG wins (+1 labels)
TP_MULT_LONG = 0.4   # LONG take profit at 0.4 * ATR (tighter target)
SL_MULT_LONG = 0.6   # LONG stop loss at 0.6 * ATR (wider stop)

# For SHORT labels (inverse):
TP_MULT_SHORT = 0.6  # SHORT take profit at 0.6 * ATR (wider target)
SL_MULT_SHORT = 0.4  # SHORT stop loss at 0.4 * ATR (tighter stop)

# Features to use (reactive, short-term focused)
FEATURES = [
    # Very short-term returns
    'ret_1', 'ret_3', 'ret_5',
    'log_ret_1',

    # Short-term volatility
    'vol_10', 'hl_range', 'norm_range', 'ATR_14',

    # Order flow (most important for short-term)
    'synthetic_order_flow', 'flow_cvd_60',
    'micro_velocity', 'effort_ratio',

    # Microstructure
    'spread_zscore', 'close_mid_diff',

    # Candlestick
    'body_pct', 'upper_wick_pct', 'lower_wick_pct',

    # Very short-term momentum
    'momentum_5',

    # Regime (to avoid bad conditions)
    'regime_volatile', 'atr_percentile',

    # MACD for momentum confirmation
    'macd', 'macd_histogram', 'macd_momentum',

    # RSI for overbought/oversold
    'rsi',
]

# Training split
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"


# =============================================================================
# LABEL CREATION
# =============================================================================

def create_5min_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create 5-minute triple barrier labels with ASYMMETRIC thresholds.

    Uses different TP/SL for LONG vs SHORT to balance class distribution:
    - LONG: 0.4*ATR TP (tighter), 0.6*ATR SL (wider) → easier to win longs
    - SHORT: 0.6*ATR TP (wider), 0.4*ATR SL (tighter) → harder to win shorts

    This counteracts the natural bearish bias in gold price action.
    """
    # Calculate ATR if not present
    if 'ATR_14' not in df.columns:
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=7).mean()
    else:
        atr = df['ATR_14'].copy()

    # Ensure minimum ATR
    min_atr = df['close'] * 0.0005  # 0.05% minimum
    atr = atr.clip(lower=min_atr)

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    atr_vals = atr.values

    n = len(df)
    labels = np.zeros(n)

    for i in range(n - HORIZON):
        entry_price = close[i]
        current_atr = atr_vals[i]

        if np.isnan(current_atr) or current_atr <= 0:
            labels[i] = 0
            continue

        # ASYMMETRIC barriers for LONG position
        # Upside target (TP for long) - tighter to hit more often
        long_tp_price = entry_price + (TP_MULT_LONG * current_atr)
        # Downside stop (SL for long) - wider to get stopped less
        long_sl_price = entry_price - (SL_MULT_LONG * current_atr)

        long_tp_hit = None
        long_sl_hit = None

        for j in range(1, HORIZON + 1):
            if i + j >= n:
                break

            bar_high = high[i + j]
            bar_low = low[i + j]

            # Check LONG TP (price goes UP)
            if bar_high >= long_tp_price and long_tp_hit is None:
                long_tp_hit = j

            # Check LONG SL (price goes DOWN)
            if bar_low <= long_sl_price and long_sl_hit is None:
                long_sl_hit = j

            if long_tp_hit is not None and long_sl_hit is not None:
                break

        # Determine label based on which barrier was hit first
        if long_tp_hit is not None and long_sl_hit is None:
            # LONG TP hit, SL not hit → LONG win
            labels[i] = 1
        elif long_sl_hit is not None and long_tp_hit is None:
            # LONG SL hit, TP not hit → SHORT win
            labels[i] = -1
        elif long_tp_hit is not None and long_sl_hit is not None:
            # Both hit - which was first?
            if long_tp_hit < long_sl_hit:
                labels[i] = 1   # LONG win (TP hit first)
            else:
                labels[i] = -1  # SHORT win (SL hit first)
        else:
            labels[i] = 0  # Neither hit within horizon

    return pd.Series(labels, index=df.index, name='y_tb_5')


# =============================================================================
# MAIN TRAINING
# =============================================================================

def main():
    # Load data
    print("\n[1/5] Loading data...")
    df = pd.read_parquet(FEATURES_PATH)

    # Ensure datetime index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    print(f"  Loaded {len(df):,} rows from {df.index.min()} to {df.index.max()}")

    # Features already loaded from parquet
    print("\n[2/5] Features already computed in parquet file...")

    # Create labels
    print("\n[3/5] Creating 5-minute labels...")
    df['y_tb_5'] = create_5min_labels(df)

    # Label distribution
    label_counts = df['y_tb_5'].value_counts()
    print(f"  Label distribution:")
    print(f"    +1 (TP hit): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(df)*100:.1f}%)")
    print(f"    -1 (SL hit): {label_counts.get(-1, 0):,} ({label_counts.get(-1, 0)/len(df)*100:.1f}%)")
    print(f"     0 (neither): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(df)*100:.1f}%)")

    # Filter out neutral labels and NaN
    df_train = df[df['y_tb_5'] != 0].copy()

    # Check which features exist
    available_features = [f for f in FEATURES if f in df_train.columns]
    missing_features = [f for f in FEATURES if f not in df_train.columns]

    if missing_features:
        print(f"  Warning: Missing features: {missing_features}")

    print(f"  Using {len(available_features)} features")

    # Drop NaN rows
    df_train = df_train.dropna(subset=available_features + ['y_tb_5'])
    print(f"  Training samples after filtering: {len(df_train):,}")

    # Time-based split
    print("\n[4/5] Splitting data...")
    train_mask = df_train.index < TRAIN_END
    val_mask = df_train.index >= VAL_START

    X_train = df_train.loc[train_mask, available_features].values
    y_train = df_train.loc[train_mask, 'y_tb_5'].values
    X_val = df_train.loc[val_mask, available_features].values
    y_val = df_train.loc[val_mask, 'y_tb_5'].values

    print(f"  Training: {len(X_train):,} samples ({df_train[train_mask].index.min()} to {df_train[train_mask].index.max()})")
    print(f"  Validation: {len(X_val):,} samples ({df_train[val_mask].index.min()} to {df_train[val_mask].index.max()})")

    # Train model
    print("\n[5/5] Training model...")
    model = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        class_weight='balanced',
    )

    model.fit(X_train, y_train)

    # Evaluate
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Training metrics
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\nTraining Accuracy: {train_acc:.4f}")

    # Validation metrics
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {val_acc:.4f}")

    print("\nValidation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=['DOWN (-1)', 'UP (+1)']))

    # Probability calibration check
    y_val_proba = model.predict_proba(X_val)
    up_idx = list(model.classes_).index(1) if 1 in model.classes_ else -1
    proba_up = y_val_proba[:, up_idx]

    print(f"\nProbability Distribution:")
    print(f"  Mean p(up): {proba_up.mean():.3f}")
    print(f"  Std p(up):  {proba_up.std():.3f}")
    print(f"  Min p(up):  {proba_up.min():.3f}")
    print(f"  Max p(up):  {proba_up.max():.3f}")

    # Win rate by probability bucket
    print("\nWin Rate by Probability Bucket:")
    for low, high in [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]:
        mask = (proba_up >= low) & (proba_up < high)
        if mask.sum() > 0:
            win_rate = (y_val[mask] == 1).mean()
            print(f"  p(up) [{low:.1f}-{high:.1f}]: {mask.sum():,} samples, win rate = {win_rate:.1%}")

    # Save model
    print("\n" + "=" * 80)
    print("SAVING MODEL")
    print("=" * 80)

    model_data = {
        'model': model,
        'features': available_features,
        'horizon': HORIZON,
        # Asymmetric TP/SL multipliers
        'tp_mult_long': TP_MULT_LONG,
        'sl_mult_long': SL_MULT_LONG,
        'tp_mult_short': TP_MULT_SHORT,
        'sl_mult_short': SL_MULT_SHORT,
        # For backward compatibility (use LONG values as default)
        'tp_mult': TP_MULT_LONG,
        'sl_mult': SL_MULT_LONG,
        'train_end': TRAIN_END,
        'val_start': VAL_START,
        'train_samples': len(X_train),
        'val_accuracy': val_acc,
        'created': datetime.now().isoformat(),
    }

    joblib.dump(model_data, MODEL_PATH)
    print(f"  Saved to: {MODEL_PATH}")

    # Recommended thresholds
    print("\n" + "=" * 80)
    print("RECOMMENDED THRESHOLDS")
    print("=" * 80)
    print(f"  LONG signal:  p(up) >= 0.60")
    print(f"  SHORT signal: p(up) <= 0.40")
    print(f"  FLAT zone:    0.40 < p(up) < 0.60")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
