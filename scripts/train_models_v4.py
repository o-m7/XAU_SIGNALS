#!/usr/bin/env python3
"""
Train Models v4 - Using PROVEN features from backup model

The backup model achieved 60%+ win rate using the pre-computed features.
This script trains multiple model types using those same features.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
MODELS_DIR = PROJECT_ROOT / "models"

# Training config
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-06-30"
VAL_START = "2023-07-01"
VAL_END = "2024-06-30"

# Load backup model to get its features
BACKUP_PATH = PROJECT_ROOT / "models" / "backups" / "y_tb_60_hgb_tuned_20251230_175110.joblib"


def create_labels(df: pd.DataFrame, horizon: int = 30, min_move: float = 0.0005):
    """Create direction labels matching backup model approach."""
    future_return = df['close'].shift(-horizon) / df['close'] - 1
    labels = pd.Series(0, index=df.index, dtype=int)
    labels[future_return > min_move] = 1   # UP
    labels[future_return < -min_move] = -1  # DOWN
    return labels


def main():
    print("=" * 80)
    print("TRAINING MODELS v4 - USING PROVEN FEATURES")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now()}")
    
    # Load backup model features
    print("\n[1] Loading backup model features...")
    backup = joblib.load(BACKUP_PATH)
    feature_cols = backup['features']
    print(f"  Using {len(feature_cols)} proven features")
    
    # Load data
    print("\n[2] Loading data...")
    df = pd.read_parquet(FEATURES_PATH)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    print(f"  Loaded {len(df):,} rows")
    
    # Create labels
    print("\n[3] Creating labels...")
    df['label'] = create_labels(df, horizon=30)
    
    # Split data
    print("\n[4] Splitting data...")
    train_df = df[TRAIN_START:TRAIN_END].copy()
    val_df = df[VAL_START:VAL_END].copy()
    
    print(f"  Train: {len(train_df):,}")
    print(f"  Val: {len(val_df):,}")
    
    # Filter to available features
    available = [f for f in feature_cols if f in train_df.columns]
    print(f"  Available features: {len(available)}/{len(feature_cols)}")
    
    # Prepare training data
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
    
    print(f"  Train samples: {len(X_train):,} (up={y_train.mean():.1%})")
    print(f"  Val samples: {len(X_val):,}")
    
    # Class weight for imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"  Scale pos weight: {scale_pos_weight:.2f}")
    
    # Train multiple model types
    models_to_train = {
        "model_xgb_v4": {
            "type": "XGBoost",
            "model": XGBClassifier(
                max_depth=5,
                learning_rate=0.03,
                n_estimators=500,
                min_child_weight=100,
                reg_lambda=0.5,
                scale_pos_weight=scale_pos_weight,
                eval_metric='auc',
                early_stopping_rounds=30,
                random_state=42,
                n_jobs=-1
            )
        },
        "model_rf_v4": {
            "type": "RandomForest",
            "model": RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_leaf=50,
                min_samples_split=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        },
        "model_nn_v4": {
            "type": "NeuralNetwork",
            "model": MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=256,
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=42
            ),
            "needs_scaling": True
        }
    }
    
    results = {}
    
    for model_name, config in models_to_train.items():
        print(f"\n{'='*60}")
        print(f"Training: {model_name} ({config['type']})")
        print(f"{'='*60}")
        
        model = config['model']
        scaler = None
        
        X_train_use = X_train
        X_val_use = X_val
        
        # Scale for neural network
        if config.get('needs_scaling'):
            scaler = StandardScaler()
            X_train_use = scaler.fit_transform(X_train)
            X_val_use = scaler.transform(X_val)
        
        # Train
        if config['type'] == 'XGBoost':
            model.fit(X_train_use, y_train, eval_set=[(X_val_use, y_val)], verbose=False)
        else:
            model.fit(X_train_use, y_train)
        
        # Evaluate
        val_proba = model.predict_proba(X_val_use)[:, 1]
        val_pred = model.predict(X_val_use)
        
        acc = accuracy_score(y_val, val_pred)
        auc = roc_auc_score(y_val, val_proba)
        
        print(f"  Accuracy: {acc:.4f}")
        print(f"  ROC-AUC:  {auc:.4f}")
        
        # Signal distribution
        threshold = 0.55
        long_signals = (val_proba >= threshold).sum()
        short_signals = (val_proba <= 1 - threshold).sum()
        print(f"  LONG signals (>={threshold}): {long_signals:,}")
        print(f"  SHORT signals (<={1-threshold}): {short_signals:,}")
        
        # Save
        artifact = {
            'model': model,
            'model_type': config['type'],
            'features': available,
            'threshold': 0.70,  # Higher threshold for quality signals
            'val_auc': auc,
            'scaler': scaler,
            'saved_at': datetime.now().isoformat()
        }
        
        save_path = MODELS_DIR / f"{model_name}.joblib"
        joblib.dump(artifact, save_path)
        print(f"  Saved to: {save_path}")
        
        results[model_name] = {'auc': auc, 'acc': acc}
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    for name, r in results.items():
        print(f"  {name}: AUC={r['auc']:.4f}, Acc={r['acc']:.4f}")
    
    print("\nDone! Run backtest to verify.")


if __name__ == "__main__":
    main()

