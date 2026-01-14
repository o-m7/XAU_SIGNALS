#!/usr/bin/env python3
"""
SNIPER v2 - Use exact same labels as backup model (y_tb_60)
Train high-precision models that only take the BEST trades
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
BACKUP_PATH = PROJECT_ROOT / "models" / "backups" / "y_tb_60_hgb_tuned_20251230_175110.joblib"

# Use same training period as backup
TRAIN_START = "2024-01-01"
TRAIN_END = "2024-09-30"
VAL_START = "2024-10-01"
VAL_END = "2024-12-31"


def main():
    print("=" * 80)
    print("SNIPER v2 - USING y_tb_60 LABELS (same as backup)")
    print("=" * 80)
    
    # Load backup model features
    print("\n[1] Loading proven features...")
    backup = joblib.load(BACKUP_PATH)
    feature_cols = backup['features']
    print(f"  Features: {len(feature_cols)}")
    
    # Load data with pre-built labels
    print("\n[2] Loading data with y_tb_60 labels...")
    df = pd.read_parquet(FEATURES_PATH)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    print(f"  Total rows: {len(df):,}")
    print(f"  y_tb_60 distribution:")
    print(df['y_tb_60'].value_counts(normalize=True))
    
    # Split
    print("\n[3] Splitting data...")
    train_df = df[TRAIN_START:TRAIN_END].copy()
    val_df = df[VAL_START:VAL_END].copy()
    
    available = [f for f in feature_cols if f in train_df.columns]
    print(f"  Train: {len(train_df):,}")
    print(f"  Val: {len(val_df):,}")
    print(f"  Features: {len(available)}")
    
    # Prepare data - use y_tb_60
    train_clean = train_df.dropna(subset=['y_tb_60'] + available)
    train_clean = train_clean[train_clean['y_tb_60'] != 0]
    
    X_train = train_clean[available].values
    y_train = (train_clean['y_tb_60'] == 1).astype(int).values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    val_clean = val_df.dropna(subset=['y_tb_60'] + available)
    val_clean = val_clean[val_clean['y_tb_60'] != 0]
    
    X_val = val_clean[available].values
    y_val = (val_clean['y_tb_60'] == 1).astype(int).values
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Train samples: {len(X_train):,} (up={y_train.mean():.1%})")
    print(f"  Val samples: {len(X_val):,}")
    
    # Train HistGradientBoosting (same as backup)
    print("\n[4] Training HistGradientBoosting (same as backup)...")
    
    model_hgb = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.03,
        max_iter=400,
        min_samples_leaf=200,
        l2_regularization=0.1,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=42
    )
    
    model_hgb.fit(X_train, y_train)
    
    val_proba_hgb = model_hgb.predict_proba(X_val)[:, 1]
    auc_hgb = roc_auc_score(y_val, val_proba_hgb)
    acc_hgb = accuracy_score(y_val, model_hgb.predict(X_val))
    
    print(f"  HGB Accuracy: {acc_hgb:.4f}")
    print(f"  HGB ROC-AUC:  {auc_hgb:.4f}")
    
    # Train XGBoost for comparison
    print("\n[5] Training XGBoost...")
    
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos = neg / pos
    
    model_xgb = XGBClassifier(
        max_depth=4,
        learning_rate=0.03,
        n_estimators=400,
        min_child_weight=200,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos,
        eval_metric='auc',
        early_stopping_rounds=10,
        random_state=42,
        n_jobs=-1
    )
    
    model_xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    val_proba_xgb = model_xgb.predict_proba(X_val)[:, 1]
    auc_xgb = roc_auc_score(y_val, val_proba_xgb)
    acc_xgb = accuracy_score(y_val, model_xgb.predict(X_val))
    
    print(f"  XGB Accuracy: {acc_xgb:.4f}")
    print(f"  XGB ROC-AUC:  {auc_xgb:.4f}")
    
    # Test thresholds for SNIPER signals
    print("\n[6] Testing SNIPER thresholds (high confidence only)...")
    
    for name, proba in [("HGB", val_proba_hgb), ("XGB", val_proba_xgb)]:
        print(f"\n  {name}:")
        for thresh in [0.65, 0.70, 0.75, 0.80]:
            long_mask = proba >= thresh
            short_mask = proba <= 1 - thresh
            
            long_cnt = long_mask.sum()
            short_cnt = short_mask.sum()
            
            if long_cnt > 0:
                long_wr = y_val[long_mask].mean()
            else:
                long_wr = 0
            if short_cnt > 0:
                short_wr = 1 - y_val[short_mask].mean()
            else:
                short_wr = 0
            
            total = long_cnt + short_cnt
            if total > 0:
                avg_wr = (long_wr * long_cnt + short_wr * short_cnt) / total
            else:
                avg_wr = 0
            
            print(f"    {thresh}: L={long_cnt:,} S={short_cnt:,} WR={avg_wr:.1%}")
    
    # Save both models
    print("\n[7] Saving models...")
    
    for name, model, auc in [("sniper_hgb", model_hgb, auc_hgb), ("sniper_xgb", model_xgb, auc_xgb)]:
        artifact = {
            'model': model,
            'model_type': 'HistGradientBoosting' if 'hgb' in name else 'XGBoost',
            'features': available,
            'threshold': 0.70,
            'val_auc': auc,
            'label': 'y_tb_60',
            'saved_at': datetime.now().isoformat()
        }
        
        save_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(artifact, save_path)
        print(f"  Saved: {save_path}")
    
    print("\n" + "=" * 80)
    print("SNIPER TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

