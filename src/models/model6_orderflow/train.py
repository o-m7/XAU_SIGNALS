#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from datetime import datetime
import logging

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
logger = logging.getLogger(__name__)

# Import from config directly (no src prefix since sys.path is set)
from models.model6_orderflow.config import (
    MODEL_NAME,
    HORIZON_MINUTES,
    TP_MULT,
    SL_MULT,
    THRESHOLD_LONG,
    THRESHOLD_SHORT,
    IMBALANCE_THRESHOLD,
    DEFAULT_PARAMS,
    SUCCESS_CRITERIA,
)
from models.model6_orderflow.features import calculate_order_features
from models.model6_orderflow.labels import create_labels
from data_loader import load_multi_year_data

DATA_DIR = PROJECT_ROOT.parent / "Data"
MINUTE_DIR = DATA_DIR / "ohlcv_minute"
QUOTES_DIR = DATA_DIR / "quotes"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" / "model6_orderflow"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUTPUT_PATH = MODEL_OUTPUT_DIR / f"{MODEL_NAME}.joblib"

TRAIN_START = "2014-01-01"
TRAIN_END = "2023-12-31"
VAL_START = "2023-01-01"
VAL_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"


def load_data() -> pd.DataFrame:
    logger.info("Loading data (minute bars + quotes)...")
    years = list(range(2014, 2026))
    df = load_multi_year_data(
        minute_dir=str(MINUTE_DIR),
        quotes_dir=str(QUOTES_DIR),
        years=years,
        require_sizes=False
    )
    logger.info(f"Loaded {len(df):,} rows")
    logger.info(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    return df


def prepare_training_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    logger.info("Preparing training data...")
    df = calculate_order_features(df)
    df = create_labels(df)
    feature_cols = ['order_imbalance', 'mid_price_momentum_1m', 'vwap_deviation']
    common_features = ['ret_1', 'ret_5', 'ret_10', 'sigma_60', 'spread_pct']
    available_common = [f for f in common_features if f in df.columns]
    feature_cols.extend(available_common)
    logger.info(f"Using {len(feature_cols)} features")
    df_clean = df.dropna(subset=['y_tb_30'] + feature_cols)
    df_clean = df_clean[df_clean['y_tb_30'] != 0]
    logger.info(f"After filtering: {len(df_clean):,} samples")
    y = df_clean['y_tb_30'].values
    y_binary = np.where(y == 1, 1, 0)
    X = df_clean[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info(f"Final dataset: {len(X):,} samples, {X.shape[1]} features")
    logger.info(f"Label distribution: +1={np.sum(y==1):,} ({100*np.sum(y==1)/len(y):.1f}%), -1={np.sum(y==-1):,} ({100*np.sum(y==-1)/len(y):.1f}%)")
    return X, y_binary, df_clean, feature_cols


def train_model(X_train, y_train, X_val, y_val):
    logger.info("Training model...")
    best_model = None
    best_auc = 0
    best_params = None
    
    param_grid = [
        {**DEFAULT_PARAMS, 'max_depth':2, 'min_samples_leaf': 800},
        {**DEFAULT_PARAMS, 'max_depth':2, 'min_samples_leaf': 1000},
        {**DEFAULT_PARAMS, 'max_depth':3, 'min_samples_leaf': 500},
        {**DEFAULT_PARAMS, 'max_depth':3, 'min_samples_leaf': 800},
        {**DEFAULT_PARAMS, 'max_depth':3, 'min_samples_leaf': 300, 'learning_rate': 0.03},
    ]
    for idx, params in enumerate(param_grid):
        logger.info(f"Testing config {idx+1}/{len(param_grid)}")
        model = HistGradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        try:
            auc = roc_auc_score(y_val, y_val_proba)
        except ValueError:
            auc = 0.5
        if auc > best_auc:
            best_auc = auc
            best_params = params
            best_model = model
        logger.info(f"  Validation AUC: {auc:.4f}")
    
    logger.info(f"Best config: {best_params}")
    logger.info(f"Best validation AUC: {best_auc:.4f}")
    
    y_train_pred = best_model.predict(X_train)
    y_train_proba = best_model.predict_proba(X_train)[:, 1]
    y_val_pred = best_model.predict(X_val)
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    
    metrics = {
        'train': {
            'accuracy': float(accuracy_score(y_train, y_train_pred)),
            'precision': float(precision_score(y_train, y_train_pred)),
            'recall': float(recall_score(y_train, y_train_pred)),
            'f1': float(f1_score(y_train, y_train_pred)),
            'roc_auc': float(roc_auc_score(y_train, y_train_proba)),
        },
        'val': {
            'accuracy': float(accuracy_score(y_val, y_val_pred)),
            'precision': float(precision_score(y_val, y_val_pred)),
            'recall': float(recall_score(y_val, y_val_pred)),
            'f1': float(f1_score(y_val, y_val_pred)),
            'roc_auc': float(roc_auc_score(y_val, y_val_proba)),
        },
        'best_params': best_params,
        'best_auc': float(best_auc),
    }
    return best_model, metrics


def main():
    logger.info("=" * 80)
    logger.info(f"TRAINING {MODEL_NAME}")
    logger.info("=" * 80)
    logger.info(f"Horizon: {HORIZON_MINUTES} minutes")
    logger.info(f"TP: {TP_MULT} ATR")
    logger.info(f"SL: {SL_MULT} ATR")
    
    df = load_data()
    
    train_df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    val_df = df[(df.index >= VAL_START) & (df.index <= VAL_END)].copy()
    
    logger.info(f"Data split:")
    logger.info(f"  Train: {len(train_df):,} rows ({TRAIN_START} to {TRAIN_END})")
    logger.info(f"  Val:   {len(val_df):,} rows ({VAL_START} to {VAL_END})")
    
    if len(train_df) == 0:
        raise ValueError(f"No training data found for {TRAIN_START} to {TRAIN_END}")
    if len(val_df) == 0:
        raise ValueError(f"No validation data found for {VAL_START} to {VAL_END}")
    
    X_train, y_train, train_df_clean, feature_cols = prepare_training_data(train_df)
    X_val, y_val, _, _ = prepare_training_data(val_df)
    
    model, metrics = train_model(X_train, y_train, X_val, y_val)
    
    logger.info("=" * 80)
    logger.info("RESULTS")
    logger.info("=" * 80)
    logger.info(f"Training Set:")
    for key, value in metrics['train'].items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info(f"Validation Set:")
    for key, value in metrics['val'].items():
        logger.info(f"  {key}: {value:.4f}")
    
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        'model': model,
        'features': feature_cols,
        'train_period': f"{TRAIN_START} to {TRAIN_END}",
        'val_period': f"{VAL_START} to {VAL_END}",
        'horizon_minutes': HORIZON_MINUTES,
        'tp_mult': TP_MULT,
        'sl_mult': SL_MULT,
        'threshold_long': THRESHOLD_LONG,
        'threshold_short': THRESHOLD_SHORT,
        'metrics': metrics,
        'trained_at': datetime.now().isoformat(),
    }
    
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    logger.info(f"Model saved to: {MODEL_OUTPUT_PATH}")
    
    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)
    
    return model, artifact


if __name__ == "__main__":
    main()
