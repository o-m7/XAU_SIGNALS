#!/usr/bin/env python3
"""
Model 6 v2: Order Flow / Microstructure

Improvements:
- 15-bar labels (shorter horizon)
- class_weight='balanced' for class imbalance
- Volume and microstructure features
- Regime detection as features
- Proper ATR validation
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.labeling import create_triple_barrier_labels, calculate_atr
from src.regime_v2 import add_regime_features

# Paths
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "model6_orderflow"
MODEL_OUTPUT_PATH = MODEL_OUTPUT_DIR / "model6_v2.joblib"

# Date splits
TRAIN_START = "2014-01-01"
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2025-12-31"

# Model config
LABEL_HORIZON = 15
TP_MULT = 1.5
SL_MULT = 1.0
HIGH_CONF_THRESHOLD = 0.60


def build_orderflow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build order flow / microstructure features."""
    df = df.copy()
    
    # Volume analysis
    if 'volume' in df.columns:
        print("  Building volume features...")
        df['volume_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma20'] + 1e-10)
        df['volume_spike'] = (df['volume'] > df['volume_ma20'] * 2).astype(int)
        
        # Volume trend
        df['volume_trend'] = df['volume_ma5'] / (df['volume_ma20'] + 1e-10)
        
        # Accumulation/Distribution
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        df['ad_line'] = (clv * df['volume']).cumsum()
        df['ad_momentum'] = df['ad_line'].diff(5)
    
    # Price momentum (micro-level)
    if 'close' in df.columns:
        print("  Building micro momentum...")
        for period in [3, 5, 10]:
            df[f'momentum_{period}'] = df['close'].pct_change(period) * 100
            df[f'momentum_{period}_smooth'] = df[f'momentum_{period}'].rolling(3).mean()
        
        # Velocity and acceleration
        df['price_velocity'] = df['close'].diff()
        df['price_acceleration'] = df['price_velocity'].diff()
        
        # Micro trend strength
        df['micro_trend'] = df['close'].rolling(10).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 10 else 0
        )
    
    # High-Low range dynamics
    if all(c in df.columns for c in ['high', 'low', 'close']):
        print("  Building range features...")
        df['hl_range'] = df['high'] - df['low']
        df['hl_range_ma'] = df['hl_range'].rolling(20).mean()
        df['hl_range_ratio'] = df['hl_range'] / (df['hl_range_ma'] + 1e-10)
        
        # Upper/Lower wick ratio
        df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
        df['wick_ratio'] = (df['upper_wick'] - df['lower_wick']) / (df['hl_range'] + 1e-10)
    
    return df


def main():
    print("=" * 80)
    print("MODEL 6 v2: ORDER FLOW / MICROSTRUCTURE")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Train Period: {TRAIN_START} to {TRAIN_END}")
    print(f"Validation Period: {VAL_START} to {VAL_END}")
    print(f"\nConfiguration:")
    print(f"  Label Horizon: {LABEL_HORIZON} bars")
    print(f"  TP/SL: {TP_MULT}/{SL_MULT} ATR")
    print(f"  Class Weighting: balanced")
    
    # Load features
    print(f"\n[1] Loading features...")
    if not FEATURES_PATH.exists():
        print(f"ERROR: Features file not found!")
        return
    
    df = pd.read_parquet(FEATURES_PATH)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    print(f"  Loaded: {len(df):,} rows")
    
    # Build orderflow features
    print("\n[2] Building order flow features...")
    df = build_orderflow_features(df)
    
    # Validate ATR
    print("\n[3] Validating ATR...")
    if 'atr_14' not in df.columns:
        df['atr_14'] = calculate_atr(df)
    min_atr = df['close'] * 0.001
    df['atr_14'] = df['atr_14'].clip(lower=min_atr)
    
    # Add regime
    print("\n[4] Adding regime detection...")
    df = add_regime_features(df)
    
    # Create labels
    print(f"\n[5] Creating {LABEL_HORIZON}-bar labels...")
    df['y_label'] = create_triple_barrier_labels(
        df, horizon=LABEL_HORIZON, tp_mult=TP_MULT, sl_mult=SL_MULT
    )
    
    label_dist = df['y_label'].value_counts(normalize=True).sort_index()
    print(f"  Down: {label_dist.get(-1, 0):.1%}, Neutral: {label_dist.get(0, 0):.1%}, Up: {label_dist.get(1, 0):.1%}")
    
    # Split
    print("\n[6] Splitting data...")
    train_df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    val_df = df[(df.index >= VAL_START) & (df.index <= VAL_END)].copy()
    print(f"  Train: {len(train_df):,}, Val: {len(val_df):,}")
    
    # Select features
    exclude_cols = ['y_label', 'y_tb_15', 'y_tb_30', 'y_tb_60', 'y_tb_120', 
                    'y_dir_15', 'timestamp', 'date', 'regime']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols 
                    and not c.startswith('y_') 
                    and train_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    print(f"\n[7] Preparing data ({len(feature_cols)} features)...")
    
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
    
    print(f"  Train: {len(X_train):,} (up={y_train.mean():.1%})")
    print(f"  Val: {len(X_val):,}")
    
    # Train
    print("\n[8] Hyperparameter search...")
    param_configs = [
        {'max_depth': 3, 'learning_rate': 0.05, 'min_samples_leaf': 500, 'l2_regularization': 0.3},
        {'max_depth': 4, 'learning_rate': 0.03, 'min_samples_leaf': 300, 'l2_regularization': 0.5},
        {'max_depth': 3, 'learning_rate': 0.01, 'min_samples_leaf': 1000, 'l2_regularization': 0.5},
    ]
    
    best_auc = 0
    best_params = None
    best_model = None
    
    for i, params in enumerate(param_configs):
        model = HistGradientBoostingClassifier(
            class_weight='balanced',
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            min_samples_leaf=params['min_samples_leaf'],
            l2_regularization=params['l2_regularization'],
            max_iter=200,
            early_stopping=True,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        val_proba = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_proba)
        print(f"  Config {i+1}: AUC={auc:.4f}")
        
        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_model = model
    
    print(f"\n  Best AUC: {best_auc:.4f}")
    
    # Metrics
    print("\n[9] Final metrics...")
    val_pred = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)[:, 1]
    
    print(f"  Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"  ROC-AUC:  {roc_auc_score(y_val, val_proba):.4f}")
    print(f"  F1 (up):  {f1_score(y_val, val_pred, pos_label=1):.4f}")
    print(f"  F1 (down): {f1_score(y_val, val_pred, pos_label=0):.4f}")
    
    # Signal balance
    print(f"\n[10] Signal distribution...")
    long_sig = val_proba >= HIGH_CONF_THRESHOLD
    short_sig = val_proba <= (1 - HIGH_CONF_THRESHOLD)
    print(f"  LONG: {long_sig.sum():,} ({long_sig.mean():.1%})")
    print(f"  SHORT: {short_sig.sum():,} ({short_sig.mean():.1%})")
    
    # Save
    print(f"\n[11] Saving model...")
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        'model': best_model,
        'features': feature_cols,
        'label_horizon': LABEL_HORIZON,
        'tp_mult': TP_MULT,
        'sl_mult': SL_MULT,
        'strategy': 'orderflow',
        'train_period': f"{TRAIN_START} to {TRAIN_END}",
        'val_auc': best_auc,
        'class_weight': 'balanced',
        'high_conf_threshold': HIGH_CONF_THRESHOLD,
        'saved_at': datetime.now().isoformat(),
    }
    
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    print(f"  Saved to {MODEL_OUTPUT_PATH}")
    
    print("\n" + "=" * 80)
    print("MODEL 6 v2 TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

