#!/usr/bin/env python3
"""
SNIPER MODELS - High Precision, Few Trades

Strategy: Only take the BEST trades with multiple confirmations
- High probability threshold (0.75+)
- Multiple indicator agreement
- Quality over quantity
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
BACKUP_PATH = PROJECT_ROOT / "models" / "backups" / "y_tb_60_hgb_tuned_20251230_175110.joblib"

TRAIN_START = "2020-01-01"
TRAIN_END = "2023-06-30"
VAL_START = "2023-07-01"
VAL_END = "2024-06-30"


def create_sniper_labels(df: pd.DataFrame, horizon: int = 30):
    """
    Create labels for SNIPER trades only.
    Label = 1 only for STRONG moves (top 20% of moves)
    """
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    
    # Calculate thresholds for strong moves
    up_threshold = future_return.quantile(0.80)  # Top 20% up moves
    down_threshold = future_return.quantile(0.20)  # Top 20% down moves
    
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[future_return >= up_threshold] = 1   # Strong UP
    labels[future_return <= down_threshold] = -1  # Strong DOWN
    
    return labels, up_threshold, down_threshold


def main():
    print("=" * 80)
    print("SNIPER MODELS - HIGH PRECISION TRADING")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now()}")
    
    # Load backup model features (proven to work)
    print("\n[1] Loading proven features from backup model...")
    backup = joblib.load(BACKUP_PATH)
    feature_cols = backup['features']
    print(f"  Using {len(feature_cols)} features")
    
    # Load data
    print("\n[2] Loading data...")
    df = pd.read_parquet(FEATURES_PATH)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    print(f"  Loaded {len(df):,} rows")
    
    # Create SNIPER labels (only strong moves)
    print("\n[3] Creating SNIPER labels (only strong moves)...")
    df['label'], up_thresh, down_thresh = create_sniper_labels(df, horizon=30)
    print(f"  UP threshold: {up_thresh:.4%}")
    print(f"  DOWN threshold: {down_thresh:.4%}")
    
    label_dist = df['label'].value_counts(normalize=True)
    print(f"  Label distribution:")
    print(f"    Strong UP (+1):  {label_dist.get(1, 0):.1%}")
    print(f"    Neutral (0):     {label_dist.get(0, 0):.1%}")
    print(f"    Strong DOWN (-1): {label_dist.get(-1, 0):.1%}")
    
    # Split data
    print("\n[4] Splitting data...")
    train_df = df[TRAIN_START:TRAIN_END].copy()
    val_df = df[VAL_START:VAL_END].copy()
    
    available = [f for f in feature_cols if f in train_df.columns]
    print(f"  Available features: {len(available)}")
    
    # Prepare data - exclude neutral labels
    train_clean = train_df.dropna(subset=['label'] + available)
    train_clean = train_clean[train_clean['label'] != 0]
    
    X_train = train_clean[available].values
    y_train = (train_clean['label'] == 1).astype(int).values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    val_clean = val_df.dropna(subset=['label'] + available)
    val_clean = val_clean[val_clean['label'] != 0]
    
    X_val = val_clean[available].values
    y_val = (val_clean['label'] == 1).astype(int).values
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Train: {len(X_train):,} samples (up={y_train.mean():.1%})")
    print(f"  Val: {len(X_val):,} samples")
    
    # Train SNIPER XGBoost
    print("\n[5] Training SNIPER XGBoost...")
    
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos = neg / pos if pos > 0 else 1.0
    
    model = XGBClassifier(
        max_depth=6,
        learning_rate=0.02,
        n_estimators=500,
        min_child_weight=50,
        reg_lambda=1.0,
        reg_alpha=0.5,
        scale_pos_weight=scale_pos,
        eval_metric='auc',
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Evaluate
    val_proba = model.predict_proba(X_val)[:, 1]
    val_pred = model.predict(X_val)
    
    auc = roc_auc_score(y_val, val_proba)
    acc = accuracy_score(y_val, val_pred)
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  ROC-AUC:  {auc:.4f}")
    
    # Test different SNIPER thresholds
    print("\n[6] Testing SNIPER thresholds...")
    for thresh in [0.70, 0.75, 0.80, 0.85]:
        long_signals = (val_proba >= thresh).sum()
        short_signals = (val_proba <= 1 - thresh).sum()
        
        # Calculate expected win rate at this threshold
        if long_signals > 0:
            long_wr = y_val[val_proba >= thresh].mean()
        else:
            long_wr = 0
        if short_signals > 0:
            short_wr = 1 - y_val[val_proba <= 1 - thresh].mean()
        else:
            short_wr = 0
        
        total_signals = long_signals + short_signals
        avg_wr = (long_wr * long_signals + short_wr * short_signals) / (total_signals + 1e-10)
        
        print(f"  Threshold {thresh}: LONG={long_signals:,}, SHORT={short_signals:,}, Expected WR={avg_wr:.1%}")
    
    # Save with high threshold
    print("\n[7] Saving SNIPER model...")
    artifact = {
        'model': model,
        'model_type': 'XGBoost_Sniper',
        'features': available,
        'threshold': 0.75,  # SNIPER threshold
        'val_auc': auc,
        'label_type': 'strong_moves_only',
        'saved_at': datetime.now().isoformat()
    }
    
    save_path = MODELS_DIR / "sniper_xgb.joblib"
    joblib.dump(artifact, save_path)
    print(f"  Saved to: {save_path}")
    
    print("\n" + "=" * 80)
    print("SNIPER MODEL TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

