#!/usr/bin/env python3
"""
Standalone Training Script for Model 5 (Range Reversion)

Train on 2014-2023, validate on 2024-2025
Uses pre-computed features from xauusd_features_2020_2025.parquet
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

# Paths
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "model5_strict_range_reversion.joblib"

# Date splits
TRAIN_START = "2014-01-01"
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2025-12-31"

# Range Reversion Strategy Parameters
MAX_EFFICIENCY_RATIO = 0.3  # Only trade in ranging markets (ER < 0.3)
HIGH_CONFIDENCE_THRESHOLD = 0.55


def build_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build range/mean reversion specific features."""
    df = df.copy()
    
    # Efficiency Ratio (ER) - measures trendiness
    if 'er_30' not in df.columns and 'close' in df.columns:
        print("  Building Efficiency Ratio...")
        change = abs(df['close'] - df['close'].shift(30))
        volatility = df['close'].diff().abs().rolling(30).sum()
        df['er_30'] = change / (volatility + 1e-10)
    
    # Bollinger Band position
    if 'bb_position' not in df.columns and 'close' in df.columns:
        print("  Building Bollinger Band features...")
        ma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_upper'] = ma20 + 2 * std20
        df['bb_lower'] = ma20 - 2 * std20
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
    
    # RSI for mean reversion signals
    if 'rsi_14' not in df.columns and 'close' in df.columns:
        print("  Building RSI...")
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Z-score of price
    if 'price_zscore' not in df.columns and 'close' in df.columns:
        print("  Building price Z-score...")
        ma50 = df['close'].rolling(50).mean()
        std50 = df['close'].rolling(50).std()
        df['price_zscore'] = (df['close'] - ma50) / (std50 + 1e-10)
    
    return df


def main():
    print("=" * 80)
    print("MODEL 5 (RANGE REVERSION) - STANDALONE TRAINING")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Train Period: {TRAIN_START} to {TRAIN_END}")
    print(f"Validation Period: {VAL_START} to {VAL_END}")
    print(f"\nStrategy: Mean Reversion in Range-bound Markets")
    print(f"  Max Efficiency Ratio: {MAX_EFFICIENCY_RATIO}")
    
    # Load features
    print(f"\n[1] Loading features from {FEATURES_PATH}...")
    if not FEATURES_PATH.exists():
        print(f"ERROR: Features file not found!")
        return
    
    df = pd.read_parquet(FEATURES_PATH)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    # Build range features
    print("\n[2] Building range reversion features...")
    df = build_range_features(df)
    
    # Split data
    print("\n[3] Splitting data...")
    train_df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    val_df = df[(df.index >= VAL_START) & (df.index <= VAL_END)].copy()
    
    print(f"  Train: {len(train_df):,} rows ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"  Validation: {len(val_df):,} rows ({val_df.index.min().date()} to {val_df.index.max().date()})")
    
    # Use y_tb_60 as label
    label_col = 'y_tb_60'
    if label_col not in train_df.columns:
        print(f"ERROR: Label column '{label_col}' not found!")
        return
    
    # Select features
    exclude_cols = ['y_tb_15', 'y_tb_30', 'y_tb_60', 'y_tb_120', 'timestamp', 'date', 
                    'signal_direction', 'target', 'label']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols 
                    and not c.startswith('y_') and train_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    print(f"\n[4] Preparing training data...")
    print(f"  Using {len(feature_cols)} features")
    
    # Filter for ranging markets only (ER < threshold)
    if 'er_30' in train_df.columns:
        train_range = train_df[train_df['er_30'] < MAX_EFFICIENCY_RATIO].copy()
        val_range = val_df[val_df['er_30'] < MAX_EFFICIENCY_RATIO].copy()
        print(f"  Filtered to ranging markets (ER < {MAX_EFFICIENCY_RATIO}):")
        print(f"    Train: {len(train_range):,} rows ({len(train_range)/len(train_df):.1%} of data)")
        print(f"    Validation: {len(val_range):,} rows ({len(val_range)/len(val_df):.1%} of data)")
    else:
        train_range = train_df
        val_range = val_df
        print("  Warning: ER feature not available, using all data")
    
    # Clean training data
    train_clean = train_range.dropna(subset=[label_col] + feature_cols[:10])
    train_clean = train_clean[train_clean[label_col] != 0]
    
    X_train = train_clean[feature_cols].values
    y_train = (train_clean[label_col] > 0).astype(int).values
    X_train = np.nan_to_num(X_train, nan=0.0)
    
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Class distribution: up={y_train.sum():,} ({y_train.mean():.1%}), down={len(y_train)-y_train.sum():,}")
    
    # Clean validation data
    val_clean = val_range.dropna(subset=[label_col] + feature_cols[:10])
    val_clean = val_clean[val_clean[label_col] != 0]
    
    X_val = val_clean[feature_cols].values
    y_val = (val_clean[label_col] > 0).astype(int).values
    X_val = np.nan_to_num(X_val, nan=0.0)
    
    print(f"  Validation samples: {len(X_val):,}")
    
    # Hyperparameter search
    print("\n[5] Hyperparameter search...")
    param_configs = [
        {'max_depth': 3, 'learning_rate': 0.05, 'min_samples_leaf': 500, 'l2_regularization': 0.3},
        {'max_depth': 4, 'learning_rate': 0.03, 'min_samples_leaf': 300, 'l2_regularization': 0.5},
        {'max_depth': 3, 'learning_rate': 0.01, 'min_samples_leaf': 1000, 'l2_regularization': 0.5},
        {'max_depth': 2, 'learning_rate': 0.02, 'min_samples_leaf': 800, 'l2_regularization': 0.4},
    ]
    
    best_auc = 0
    best_params = None
    best_model = None
    
    for i, params in enumerate(param_configs):
        model = HistGradientBoostingClassifier(
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
    print("\n[6] Final metrics...")
    
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
    
    # High confidence filter analysis
    print(f"\n[7] High confidence filter (threshold={HIGH_CONFIDENCE_THRESHOLD})...")
    high_conf_long = val_proba >= HIGH_CONFIDENCE_THRESHOLD
    high_conf_short = val_proba <= (1.0 - HIGH_CONFIDENCE_THRESHOLD)
    n_long = high_conf_long.sum()
    n_short = high_conf_short.sum()
    print(f"  LONG signals: {n_long:,}")
    print(f"  SHORT signals: {n_short:,}")
    print(f"  Total high-conf trades: {n_long + n_short:,} ({(n_long + n_short) / len(val_proba):.1%} of bars)")
    
    # Save model
    print(f"\n[8] Saving model to {MODEL_OUTPUT_PATH}...")
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        'model': best_model,
        'features': feature_cols,
        'label': label_col,
        'train_period': f"{TRAIN_START} to {TRAIN_END}",
        'val_period': f"{VAL_START} to {VAL_END}",
        'best_params': best_params,
        'val_auc': best_auc,
        'strategy': 'range_reversion',
        'max_efficiency_ratio': MAX_EFFICIENCY_RATIO,
        'high_confidence_threshold': HIGH_CONFIDENCE_THRESHOLD,
        'saved_at': datetime.now().isoformat(),
    }
    
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    print(f"  Model saved successfully!")
    
    print("\n" + "=" * 80)
    print("MODEL 5 TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

