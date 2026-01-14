#!/usr/bin/env python3
"""
Model 1 v2: High Confidence Trend Following

Improvements:
- 15-bar labels (shorter horizon)
- class_weight='balanced' for class imbalance
- Regime detection as features
- Proper ATR validation
- Feature importance logging
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.labeling import create_triple_barrier_labels, calculate_atr
from src.regime_v2 import add_regime_features

# Paths
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "model1_v2.joblib"

# Date splits
TRAIN_START = "2014-01-01"
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2025-12-31"

# Model config
LABEL_HORIZON = 15  # 15 minutes
TP_MULT = 1.5       # Take profit: 1.5 ATR
SL_MULT = 1.0       # Stop loss: 1.0 ATR
HIGH_CONF_THRESHOLD = 0.60


def main():
    print("=" * 80)
    print("MODEL 1 v2: HIGH CONFIDENCE TREND FOLLOWING")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Train Period: {TRAIN_START} to {TRAIN_END}")
    print(f"Validation Period: {VAL_START} to {VAL_END}")
    print(f"\nConfiguration:")
    print(f"  Label Horizon: {LABEL_HORIZON} bars")
    print(f"  TP Mult: {TP_MULT} ATR")
    print(f"  SL Mult: {SL_MULT} ATR")
    print(f"  Class Weighting: balanced")
    
    # Load features
    print(f"\n[1] Loading features...")
    if not FEATURES_PATH.exists():
        print(f"ERROR: Features file not found: {FEATURES_PATH}")
        return
    
    df = pd.read_parquet(FEATURES_PATH)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    # Add ATR if missing
    print("\n[2] Validating ATR...")
    if 'atr_14' not in df.columns:
        print("  Calculating ATR...")
        df['atr_14'] = calculate_atr(df)
    else:
        # Ensure positive ATR
        min_atr = df['close'] * 0.001
        df['atr_14'] = df['atr_14'].clip(lower=min_atr)
    print(f"  ATR range: {df['atr_14'].min():.4f} - {df['atr_14'].max():.4f}")
    
    # Add regime features
    print("\n[3] Adding regime detection...")
    df = add_regime_features(df)
    
    # Create improved labels
    print(f"\n[4] Creating {LABEL_HORIZON}-bar labels...")
    df['y_label'] = create_triple_barrier_labels(
        df, horizon=LABEL_HORIZON, tp_mult=TP_MULT, sl_mult=SL_MULT
    )
    
    label_dist = df['y_label'].value_counts(normalize=True).sort_index()
    print(f"  Label distribution:")
    print(f"    Down (-1): {label_dist.get(-1, 0):.1%}")
    print(f"    Neutral (0): {label_dist.get(0, 0):.1%}")
    print(f"    Up (+1): {label_dist.get(1, 0):.1%}")
    
    # Split data
    print("\n[5] Splitting data...")
    train_df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    val_df = df[(df.index >= VAL_START) & (df.index <= VAL_END)].copy()
    
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Validation: {len(val_df):,} rows")
    
    # Select features
    exclude_cols = ['y_label', 'y_tb_15', 'y_tb_30', 'y_tb_60', 'y_tb_120', 
                    'y_dir_15', 'timestamp', 'date', 'regime']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols 
                    and not c.startswith('y_') 
                    and train_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    print(f"\n[6] Preparing training data...")
    print(f"  Using {len(feature_cols)} features")
    
    # Clean data - remove neutral labels and NaN
    train_clean = train_df.dropna(subset=['y_label'] + feature_cols[:20])
    train_clean = train_clean[train_clean['y_label'] != 0]
    
    X_train = train_clean[feature_cols].values
    y_train = (train_clean['y_label'] > 0).astype(int).values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    val_clean = val_df.dropna(subset=['y_label'] + feature_cols[:20])
    val_clean = val_clean[val_clean['y_label'] != 0]
    
    X_val = val_clean[feature_cols].values
    y_val = (val_clean['y_label'] > 0).astype(int).values
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Train class balance: up={y_train.mean():.1%}, down={1-y_train.mean():.1%}")
    print(f"  Validation samples: {len(X_val):,}")
    
    # Hyperparameter search with class_weight='balanced'
    print("\n[7] Hyperparameter search (with class_weight='balanced')...")
    param_configs = [
        {'max_depth': 3, 'learning_rate': 0.05, 'min_samples_leaf': 500, 'l2_regularization': 0.3},
        {'max_depth': 4, 'learning_rate': 0.03, 'min_samples_leaf': 300, 'l2_regularization': 0.5},
        {'max_depth': 3, 'learning_rate': 0.01, 'min_samples_leaf': 1000, 'l2_regularization': 0.5},
        {'max_depth': 2, 'learning_rate': 0.05, 'min_samples_leaf': 500, 'l2_regularization': 0.4},
    ]
    
    best_auc = 0
    best_params = None
    best_model = None
    
    for i, params in enumerate(param_configs):
        model = HistGradientBoostingClassifier(
            class_weight='balanced',  # KEY FIX
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            min_samples_leaf=params['min_samples_leaf'],
            l2_regularization=params['l2_regularization'],
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        val_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_proba)
        print(f"  Config {i+1}/{len(param_configs)}: AUC={auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_model = model
    
    print(f"\n  Best config: {best_params}")
    print(f"  Best AUC: {best_auc:.4f}")
    
    # Final metrics
    print("\n[8] Final metrics...")
    
    train_pred = best_model.predict(X_train)
    train_proba = best_model.predict_proba(X_train)[:, 1]
    val_pred = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)[:, 1]
    
    print(f"\n  Training Set:")
    print(f"    Accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"    ROC-AUC:  {roc_auc_score(y_train, train_proba):.4f}")
    print(f"    F1 (up):  {f1_score(y_train, train_pred, pos_label=1):.4f}")
    print(f"    F1 (down): {f1_score(y_train, train_pred, pos_label=0):.4f}")
    
    print(f"\n  Validation Set:")
    print(f"    Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"    ROC-AUC:  {roc_auc_score(y_val, val_proba):.4f}")
    print(f"    F1 (up):  {f1_score(y_val, val_pred, pos_label=1):.4f}")
    print(f"    F1 (down): {f1_score(y_val, val_pred, pos_label=0):.4f}")
    
    # Signal distribution with balanced model
    print(f"\n[9] Signal distribution check...")
    high_conf_long = val_proba >= HIGH_CONF_THRESHOLD
    high_conf_short = val_proba <= (1.0 - HIGH_CONF_THRESHOLD)
    neutral = ~high_conf_long & ~high_conf_short
    
    print(f"  LONG signals (proba >= {HIGH_CONF_THRESHOLD}): {high_conf_long.sum():,} ({high_conf_long.mean():.1%})")
    print(f"  SHORT signals (proba <= {1-HIGH_CONF_THRESHOLD}): {high_conf_short.sum():,} ({high_conf_short.mean():.1%})")
    print(f"  NEUTRAL (no trade): {neutral.sum():,} ({neutral.mean():.1%})")
    
    # Feature importance
    print("\n[10] Top 20 feature importances...")
    if hasattr(best_model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in importances.head(20).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    print(f"\n[11] Saving model to {MODEL_OUTPUT_PATH}...")
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        'model': best_model,
        'features': feature_cols,
        'label': 'y_label',
        'label_horizon': LABEL_HORIZON,
        'tp_mult': TP_MULT,
        'sl_mult': SL_MULT,
        'train_period': f"{TRAIN_START} to {TRAIN_END}",
        'val_period': f"{VAL_START} to {VAL_END}",
        'best_params': best_params,
        'val_auc': best_auc,
        'class_weight': 'balanced',
        'high_conf_threshold': HIGH_CONF_THRESHOLD,
        'saved_at': datetime.now().isoformat(),
    }
    
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    print(f"  Model saved successfully!")
    
    print("\n" + "=" * 80)
    print("MODEL 1 v2 TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

