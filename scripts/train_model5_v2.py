#!/usr/bin/env python3
"""
Model 5 v2: Range Reversion

Improvements:
- 30-bar labels (slightly longer for mean reversion)
- class_weight='balanced' for class imbalance
- Only train on RANGING regime data
- Proper ATR validation
- Regime detection features
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
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "model5_v2.joblib"

# Date splits
TRAIN_START = "2014-01-01"
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2025-12-31"

# Model config - mean reversion needs slightly longer horizon
LABEL_HORIZON = 30  # 30 minutes for mean reversion
TP_MULT = 1.5
SL_MULT = 1.0
HIGH_CONF_THRESHOLD = 0.60
MAX_EFFICIENCY_RATIO = 0.3  # Only train on ranging markets


def build_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build mean reversion specific features."""
    df = df.copy()
    
    # Efficiency Ratio (already may exist)
    if 'er_30' not in df.columns:
        print("  Building Efficiency Ratio...")
        change = abs(df['close'] - df['close'].shift(30))
        volatility = df['close'].diff().abs().rolling(30).sum()
        df['er_30'] = change / (volatility + 1e-10)
    
    # Bollinger Band features
    if 'bb_position' not in df.columns:
        print("  Building Bollinger Bands...")
        for period in [20, 50]:
            ma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = ma + 2 * std
            df[f'bb_lower_{period}'] = ma - 2 * std
            df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / ma
    
    # RSI for overbought/oversold
    if 'rsi_14' not in df.columns:
        print("  Building RSI...")
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # RSI extremes (good for mean reversion)
    df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
    
    # Price Z-score
    if 'price_zscore' not in df.columns:
        print("  Building Z-score...")
        for period in [20, 50]:
            ma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            df[f'zscore_{period}'] = (df['close'] - ma) / (std + 1e-10)
    
    return df


def main():
    print("=" * 80)
    print("MODEL 5 v2: RANGE REVERSION")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Train Period: {TRAIN_START} to {TRAIN_END}")
    print(f"Validation Period: {VAL_START} to {VAL_END}")
    print(f"\nConfiguration:")
    print(f"  Label Horizon: {LABEL_HORIZON} bars (mean reversion)")
    print(f"  TP/SL: {TP_MULT}/{SL_MULT} ATR")
    print(f"  Regime Filter: RANGING only (ER < {MAX_EFFICIENCY_RATIO})")
    
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
    
    # Build range features
    print("\n[2] Building range reversion features...")
    df = build_range_features(df)
    
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
    
    # Split and filter for ranging markets
    print("\n[6] Filtering for ranging markets...")
    train_df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    val_df = df[(df.index >= VAL_START) & (df.index <= VAL_END)].copy()
    
    # Filter for ranging regime
    if 'er_30' in train_df.columns:
        train_range = train_df[train_df['er_30'] < MAX_EFFICIENCY_RATIO].copy()
        val_range = val_df[val_df['er_30'] < MAX_EFFICIENCY_RATIO].copy()
        print(f"  Train (ranging): {len(train_range):,} ({len(train_range)/len(train_df):.1%})")
        print(f"  Val (ranging): {len(val_range):,} ({len(val_range)/len(val_df):.1%})")
    else:
        train_range = train_df
        val_range = val_df
    
    # Select features
    exclude_cols = ['y_label', 'y_tb_15', 'y_tb_30', 'y_tb_60', 'y_tb_120', 
                    'y_dir_15', 'timestamp', 'date', 'regime']
    feature_cols = [c for c in train_range.columns if c not in exclude_cols 
                    and not c.startswith('y_') 
                    and train_range[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    print(f"\n[7] Preparing data ({len(feature_cols)} features)...")
    
    train_clean = train_range.dropna(subset=['y_label'] + feature_cols[:20])
    train_clean = train_clean[train_clean['y_label'] != 0]
    
    X_train = train_clean[feature_cols].values
    y_train = (train_clean['y_label'] > 0).astype(int).values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    val_clean = val_range.dropna(subset=['y_label'] + feature_cols[:20])
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
        {'max_depth': 2, 'learning_rate': 0.03, 'min_samples_leaf': 800, 'l2_regularization': 0.5},
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
    artifact = {
        'model': best_model,
        'features': feature_cols,
        'label_horizon': LABEL_HORIZON,
        'tp_mult': TP_MULT,
        'sl_mult': SL_MULT,
        'strategy': 'range_reversion',
        'max_efficiency_ratio': MAX_EFFICIENCY_RATIO,
        'train_period': f"{TRAIN_START} to {TRAIN_END}",
        'val_auc': best_auc,
        'class_weight': 'balanced',
        'high_conf_threshold': HIGH_CONF_THRESHOLD,
        'saved_at': datetime.now().isoformat(),
    }
    
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    print(f"  Saved to {MODEL_OUTPUT_PATH}")
    
    print("\n" + "=" * 80)
    print("MODEL 5 v2 TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

