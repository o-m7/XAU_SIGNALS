#!/usr/bin/env python3
"""
Train 60-Minute Models with BALANCED Labels.

Uses asymmetric TP/SL to counteract data bias and achieve balanced class distribution.
This solves the short-bias problem where models were predicting SHORT too often.

Trains three models:
- Model 1 HGB: Microstructure features (HistGradientBoostingClassifier)
- Model 3 CMF/MACD: Technical indicator features (XGBoost)
- Model RF: Random Forest ensemble

Each model uses asymmetric TP/SL:
- LONG: 0.8×ATR TP (tighter), 1.2×ATR SL (wider) → easier to win longs
- SHORT: 1.2×ATR TP (wider), 0.8×ATR SL (tighter) → harder to win shorts

Usage:
    python scripts/train_60min_balanced.py
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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from xgboost import XGBClassifier

# Import feature sets from single source of truth
from src.features.model_feature_sets import (
    MODEL1_MICROSTRUCTURE_FEATURES,
    MODEL3_TECHNICAL_FEATURES,
    MODEL_RF_FEATURES,
    get_features_for_model
)

# Data path
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
OUTPUT_DIR = PROJECT_ROOT / "models" / "balanced"

print("=" * 80)
print("TRAIN 60-MINUTE BALANCED MODELS")
print("=" * 80)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Label parameters - ASYMMETRIC to balance classes
HORIZON = 60  # 60 bars (60 minutes)

# LONG-biased asymmetry to counteract bearish data bias:
# - Smaller upside target (easier to hit) + wider downside stop (harder to stop out)
# - This increases LONG wins (+1 labels)
TP_MULT_LONG = 0.8   # LONG take profit at 0.8 * ATR (tighter target)
SL_MULT_LONG = 1.2   # LONG stop loss at 1.2 * ATR (wider stop)

# For SHORT labels (inverse):
TP_MULT_SHORT = 1.2  # SHORT take profit at 1.2 * ATR (wider target)
SL_MULT_SHORT = 0.8  # SHORT stop loss at 0.8 * ATR (tighter stop)

# Training split
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"


# =============================================================================
# LABEL CREATION
# =============================================================================

def create_balanced_60min_labels(df: pd.DataFrame) -> pd.Series:
    """
    Create 60-minute triple barrier labels with ASYMMETRIC thresholds.

    Uses different TP/SL for LONG vs SHORT to balance class distribution:
    - LONG: 0.8*ATR TP (tighter), 1.2*ATR SL (wider) → easier to win longs
    - SHORT: 1.2*ATR TP (wider), 0.8*ATR SL (tighter) → harder to win shorts

    This counteracts natural data bias.
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
        long_tp_price = entry_price + (TP_MULT_LONG * current_atr)
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
            labels[i] = 1  # LONG win
        elif long_sl_hit is not None and long_tp_hit is None:
            labels[i] = -1  # SHORT win
        elif long_tp_hit is not None and long_sl_hit is not None:
            if long_tp_hit < long_sl_hit:
                labels[i] = 1   # LONG win (TP hit first)
            else:
                labels[i] = -1  # SHORT win (SL hit first)
        else:
            labels[i] = 0  # Neither hit within horizon

    return pd.Series(labels, index=df.index, name='y_tb_60')


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def train_model1_hgb(X_train, y_train, X_val, y_val, feature_names):
    """Train Model 1 using HistGradientBoostingClassifier."""
    print("\n  Training Model 1 (HGB)...")

    model = HistGradientBoostingClassifier(
        max_iter=300,
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
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)
    up_idx = list(model.classes_).index(1) if 1 in model.classes_ else -1
    proba_up = y_val_proba[:, up_idx]

    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, proba_up)

    print(f"    Validation Accuracy: {val_acc:.4f}")
    print(f"    Validation AUC: {val_auc:.4f}")

    return model, val_acc, val_auc, proba_up


def train_model3_xgb(X_train, y_train, X_val, y_val, feature_names):
    """Train Model 3 using XGBoost."""
    print("\n  Training Model 3 (XGBoost)...")

    # Calculate scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    model = XGBClassifier(
        max_depth=5,
        learning_rate=0.05,
        n_estimators=300,
        min_child_weight=100,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluate
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)

    print(f"    Validation Accuracy: {val_acc:.4f}")
    print(f"    Validation AUC: {val_auc:.4f}")

    return model, val_acc, val_auc, y_val_proba


def train_model_rf(X_train, y_train, X_val, y_val, feature_names):
    """Train Random Forest model."""
    print("\n  Training Model RF (RandomForest)...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=50,
        min_samples_split=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)

    print(f"    Validation Accuracy: {val_acc:.4f}")
    print(f"    Validation AUC: {val_auc:.4f}")

    return model, val_acc, val_auc, y_val_proba


def print_probability_analysis(y_val, proba_up, model_name):
    """Print probability distribution and win rates by bucket."""
    print(f"\n  {model_name} Probability Analysis:")
    print(f"    Mean p(up): {proba_up.mean():.3f}")
    print(f"    Std p(up):  {proba_up.std():.3f}")

    print("\n    Win Rate by Probability Bucket:")
    for low, high in [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]:
        mask = (proba_up >= low) & (proba_up < high)
        if mask.sum() > 0:
            win_rate = (y_val[mask] == 1).mean()
            print(f"      p(up) [{low:.1f}-{high:.1f}]: {mask.sum():,} samples, win rate = {win_rate:.1%}")


# =============================================================================
# MAIN TRAINING
# =============================================================================

def main():
    # Load data
    print("\n[1/6] Loading data...")
    df = pd.read_parquet(FEATURES_PATH)

    # Ensure datetime index
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    print(f"  Loaded {len(df):,} rows from {df.index.min()} to {df.index.max()}")

    # Create balanced labels
    print("\n[2/6] Creating balanced 60-minute labels...")
    print(f"  Asymmetric TP/SL: LONG={TP_MULT_LONG}/{SL_MULT_LONG}, SHORT={TP_MULT_SHORT}/{SL_MULT_SHORT}")

    df['y_tb_60'] = create_balanced_60min_labels(df)

    # Label distribution
    label_counts = df['y_tb_60'].value_counts()
    total = len(df)
    print(f"\n  Label distribution:")
    print(f"    +1 (LONG win):  {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/total*100:.1f}%)")
    print(f"    -1 (SHORT win): {label_counts.get(-1, 0):,} ({label_counts.get(-1, 0)/total*100:.1f}%)")
    print(f"     0 (neither):   {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/total*100:.1f}%)")

    # Filter out neutral labels
    df_train = df[df['y_tb_60'] != 0].copy()

    # Time-based split
    print("\n[3/6] Splitting data...")
    train_mask = df_train.index < TRAIN_END
    val_mask = df_train.index >= VAL_START

    train_df = df_train[train_mask]
    val_df = df_train[val_mask]

    print(f"  Training period: {train_df.index.min()} to {train_df.index.max()}")
    print(f"  Validation period: {val_df.index.min()} to {val_df.index.max()}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # TRAIN MODEL 1 (HGB) - Microstructure features
    # =========================================================================
    print("\n" + "=" * 80)
    print("[4/6] TRAINING MODEL 1 HGB (Microstructure)")
    print("=" * 80)

    model1_features = MODEL1_MICROSTRUCTURE_FEATURES.copy()
    available_m1 = [f for f in model1_features if f in df_train.columns]
    missing_m1 = [f for f in model1_features if f not in df_train.columns]

    if missing_m1:
        print(f"  Warning: Missing {len(missing_m1)} features: {missing_m1[:5]}...")

    print(f"  Using {len(available_m1)} features")

    # Prepare data
    train_m1 = train_df.dropna(subset=available_m1 + ['y_tb_60'])
    val_m1 = val_df.dropna(subset=available_m1 + ['y_tb_60'])

    X_train_m1 = train_m1[available_m1].values
    y_train_m1 = (train_m1['y_tb_60'] == 1).astype(int).values
    X_val_m1 = val_m1[available_m1].values
    y_val_m1 = (val_m1['y_tb_60'] == 1).astype(int).values

    print(f"  Train: {len(X_train_m1):,} samples (UP={y_train_m1.mean():.1%})")
    print(f"  Val: {len(X_val_m1):,} samples")

    model1, m1_acc, m1_auc, m1_proba = train_model1_hgb(
        X_train_m1, y_train_m1, X_val_m1, y_val_m1, available_m1
    )

    print_probability_analysis(y_val_m1, m1_proba, "Model 1 HGB")

    # Save Model 1
    m1_artifact = {
        'model': model1,
        'features': available_m1,
        'horizon': HORIZON,
        'tp_mult_long': TP_MULT_LONG,
        'sl_mult_long': SL_MULT_LONG,
        'tp_mult_short': TP_MULT_SHORT,
        'sl_mult_short': SL_MULT_SHORT,
        'tp_mult': TP_MULT_LONG,  # Backward compatibility
        'sl_mult': SL_MULT_LONG,
        'threshold_long': 0.60,
        'threshold_short': 0.40,
        'training_date': datetime.now().isoformat(),
        'train_end': TRAIN_END,
        'val_start': VAL_START,
        'val_accuracy': m1_acc,
        'val_auc': m1_auc,
        'training_samples': len(X_train_m1),
        'balanced_labels': True,
    }

    m1_path = OUTPUT_DIR / "model1_hgb_balanced.joblib"
    joblib.dump(m1_artifact, m1_path)
    print(f"\n  Saved: {m1_path}")

    # =========================================================================
    # TRAIN MODEL 3 (XGBoost) - CMF/MACD features
    # =========================================================================
    print("\n" + "=" * 80)
    print("[5/6] TRAINING MODEL 3 CMF/MACD (XGBoost)")
    print("=" * 80)

    model3_features = MODEL3_TECHNICAL_FEATURES.copy()
    available_m3 = [f for f in model3_features if f in df_train.columns]
    missing_m3 = [f for f in model3_features if f not in df_train.columns]

    if missing_m3:
        print(f"  Warning: Missing {len(missing_m3)} features: {missing_m3[:5]}...")

    print(f"  Using {len(available_m3)} features")

    # Prepare data
    train_m3 = train_df.dropna(subset=available_m3 + ['y_tb_60'])
    val_m3 = val_df.dropna(subset=available_m3 + ['y_tb_60'])

    X_train_m3 = train_m3[available_m3].values
    y_train_m3 = (train_m3['y_tb_60'] == 1).astype(int).values
    X_val_m3 = val_m3[available_m3].values
    y_val_m3 = (val_m3['y_tb_60'] == 1).astype(int).values

    print(f"  Train: {len(X_train_m3):,} samples (UP={y_train_m3.mean():.1%})")
    print(f"  Val: {len(X_val_m3):,} samples")

    model3, m3_acc, m3_auc, m3_proba = train_model3_xgb(
        X_train_m3, y_train_m3, X_val_m3, y_val_m3, available_m3
    )

    print_probability_analysis(y_val_m3, m3_proba, "Model 3 CMF/MACD")

    # Save Model 3
    m3_artifact = {
        'model': model3,
        'features': available_m3,
        'horizon': HORIZON,
        'tp_mult_long': TP_MULT_LONG,
        'sl_mult_long': SL_MULT_LONG,
        'tp_mult_short': TP_MULT_SHORT,
        'sl_mult_short': SL_MULT_SHORT,
        'tp_mult': TP_MULT_LONG,
        'sl_mult': SL_MULT_LONG,
        'threshold_long': 0.60,
        'threshold_short': 0.40,
        'training_date': datetime.now().isoformat(),
        'train_end': TRAIN_END,
        'val_start': VAL_START,
        'val_accuracy': m3_acc,
        'val_auc': m3_auc,
        'training_samples': len(X_train_m3),
        'balanced_labels': True,
    }

    m3_path = OUTPUT_DIR / "model3_cmf_macd_balanced.joblib"
    joblib.dump(m3_artifact, m3_path)
    print(f"\n  Saved: {m3_path}")

    # =========================================================================
    # TRAIN MODEL RF (RandomForest) - Same features as Model 1
    # =========================================================================
    print("\n" + "=" * 80)
    print("[6/6] TRAINING MODEL RF (RandomForest)")
    print("=" * 80)

    # Use same features as Model 1
    print(f"  Using {len(available_m1)} features (same as Model 1)")

    model_rf, rf_acc, rf_auc, rf_proba = train_model_rf(
        X_train_m1, y_train_m1, X_val_m1, y_val_m1, available_m1
    )

    print_probability_analysis(y_val_m1, rf_proba, "Model RF")

    # Save Model RF
    rf_artifact = {
        'model': model_rf,
        'features': available_m1,
        'horizon': HORIZON,
        'tp_mult_long': TP_MULT_LONG,
        'sl_mult_long': SL_MULT_LONG,
        'tp_mult_short': TP_MULT_SHORT,
        'sl_mult_short': SL_MULT_SHORT,
        'tp_mult': TP_MULT_LONG,
        'sl_mult': SL_MULT_LONG,
        'threshold_long': 0.60,
        'threshold_short': 0.40,
        'training_date': datetime.now().isoformat(),
        'train_end': TRAIN_END,
        'val_start': VAL_START,
        'val_accuracy': rf_acc,
        'val_auc': rf_auc,
        'training_samples': len(X_train_m1),
        'balanced_labels': True,
    }

    rf_path = OUTPUT_DIR / "model_rf_balanced.joblib"
    joblib.dump(rf_artifact, rf_path)
    print(f"\n  Saved: {rf_path}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"\nLabel Configuration (Balanced):")
    print(f"  LONG:  TP={TP_MULT_LONG}×ATR, SL={SL_MULT_LONG}×ATR")
    print(f"  SHORT: TP={TP_MULT_SHORT}×ATR, SL={SL_MULT_SHORT}×ATR")
    print(f"\nModels saved to: {OUTPUT_DIR}")
    print(f"\n  Model 1 HGB:     Accuracy={m1_acc:.1%}, AUC={m1_auc:.3f}")
    print(f"  Model 3 CMF/MACD: Accuracy={m3_acc:.1%}, AUC={m3_auc:.3f}")
    print(f"  Model RF:        Accuracy={rf_acc:.1%}, AUC={rf_auc:.3f}")
    print(f"\nRecommended Thresholds:")
    print(f"  LONG signal:  p(up) >= 0.60")
    print(f"  SHORT signal: p(up) <= 0.40")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
