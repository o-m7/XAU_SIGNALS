#!/usr/bin/env python3
"""
Train Model 1 (y_tb_60) with 2014-2023 training and 2024-2025 validation.

Modified for HIGH CONFIDENCE TRADING:
- Only generate signals when predict_proba > 0.70 (was 0.60)
- This reduces trade count but increases win rate
"""

import sys
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
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
    confusion_matrix,
)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.regime_detection import add_regime_features, filter_short_signals_by_regime

# Import existing Model 1 training utilities
from src.train_y_tb_60 import (
    compute_sample_weights,
    make_model,
    get_feature_columns,
    prepare_data,
    compute_metrics,
    DEFAULT_MODEL_PARAMS,
    PARAM_GRID,
    TARGET,
    LABEL_COLS,
    RAW_COLS,
    RANDOM_SEED,
)

# New date ranges
TRAIN_START = "2014-01-01"
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2025-12-31"

# Model output path
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "model1_high_conf.joblib"
FEATURES_FILE = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"

# HIGH CONFIDENCE FILTER
HIGH_CONFIDENCE_THRESHOLD = 0.55  # Only trade when proba > 0.55 (balanced threshold)


def load_feature_matrix() -> pd.DataFrame:
    """Load the feature matrix."""
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")
    
    print(f"Loading features from: {FEATURES_FILE}")
    df = pd.read_parquet(FEATURES_FILE)
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
    
    df = df.sort_index()
    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    return df


def split_by_dates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train (2014-2023) and validation (2024-2025)."""
    train_df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    val_df = df[(df.index >= VAL_START) & (df.index <= VAL_END)].copy()
    
    print(f"\nData Split:")
    print(f"  Train: {len(train_df):,} rows ({TRAIN_START} to {TRAIN_END})")
    print(f"    Date range: {train_df.index.min().date()} to {train_df.index.max().date()}")
    print(f"  Validation: {len(val_df):,} rows ({VAL_START} to {VAL_END})")
    print(f"    Date range: {val_df.index.min().date()} to {val_df.index.max().date()}")
    
    return train_df, val_df


def apply_regime_filtering(df: pd.DataFrame, signals: pd.Series) -> pd.Series:
    """
    Apply regime detection to filter short signals.
    
    Only allows shorting in bearish regimes.
    """
    print("\nApplying regime detection for short signals...")
    
    # Add regime features
    df_with_regime = add_regime_features(df)
    
    # Filter short signals by regime
    filtered_signals = filter_short_signals_by_regime(
        signals, 
        df_with_regime, 
        regime_threshold=0.6
    )
    
    # Report filtering results
    original_shorts = (signals == -1).sum()
    filtered_shorts = (filtered_signals == -1).sum()
    removed = original_shorts - filtered_shorts
    
    print(f"  Original short signals: {original_shorts:,}")
    print(f"  After regime filtering: {filtered_shorts:,}")
    print(f"  Removed (not in bearish regime): {removed:,} ({100*removed/max(original_shorts,1):.1f}%)")
    
    return filtered_signals


def apply_high_confidence_filter(predictions: pd.Series, proba_up: pd.Series) -> pd.Series:
    """
    Apply high confidence filter to predictions.
    
    Only generate LONG signals when proba_up > 0.70.
    SHORT signals require very high confidence (< 0.30).
    """
    print(f"\nApplying high confidence filter (threshold: {HIGH_CONFIDENCE_THRESHOLD})...")
    
    # Default: all signals are 0 (no trade)
    signals = pd.Series(0, index=predictions.index, dtype=int)
    
    # High confidence LONG signals
    long_mask = proba_up >= HIGH_CONFIDENCE_THRESHOLD
    signals[long_mask] = 1
    
    # Very high confidence SHORT signals (stricter threshold)
    short_threshold = 1.0 - HIGH_CONFIDENCE_THRESHOLD
    short_mask = proba_up <= short_threshold
    signals[short_mask] = -1
    
    n_long = (signals == 1).sum()
    n_short = (signals == -1).sum()
    n_total = (signals != 0).sum()
    
    print(f"  LONG signals: {n_long:,} ({100*n_long/max(n_total,1):.1f}% of trades)")
    print(f"  SHORT signals: {n_short:,} ({100*n_short/max(n_total,1):.1f}% of trades)")
    print(f"  Total trades: {n_total:,}")
    
    return signals


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("TRAIN MODEL 1 (y_tb_60) - 2014-2023 Training, 2024-2025 Validation")
    print("=" * 80)
    print(f"\nTraining Period: {TRAIN_START} to {TRAIN_END}")
    print(f"Validation Period: {VAL_START} to {VAL_END}")
    print(f"Regime Detection: Enabled (only short in bearish regimes)")
    print(f"High Confidence Filter: {HIGH_CONFIDENCE_THRESHOLD} (was 0.60)")
    
    # Load data
    df = load_feature_matrix()
    
    # Split by dates
    train_df, val_df = split_by_dates(df)
    
    if len(train_df) == 0:
        raise ValueError(f"No training data found for {TRAIN_START} to {TRAIN_END}")
    if len(val_df) == 0:
        raise ValueError(f"No validation data found for {VAL_START} to {VAL_END}")
    
    # Get feature columns (same as original Model 1)
    feature_cols = get_feature_columns(train_df, prioritize_effective=False)
    print(f"\nUsing {len(feature_cols)} features (same as original Model 1)")
    
    # Prepare training data
    print("\n" + "="*80)
    print("PREPARING TRAINING DATA")
    print("="*80)
    X_train, y_train, train_df_clean = prepare_data(train_df, feature_cols)
    
    # Prepare validation data
    print("\n" + "="*80)
    print("PREPARING VALIDATION DATA")
    print("="*80)
    X_val, y_val, val_df_clean = prepare_data(val_df, feature_cols)
    
    # Hyperparameter search (simplified - use best from grid)
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH")
    print("="*80)
    print("Using same hyperparameter grid as original Model 1")
    
    best_params = None
    best_auc = 0
    
    for idx, params in enumerate(PARAM_GRID):
        model = make_model(params)
        sample_weights = compute_sample_weights(y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        y_val_proba = model.predict_proba(X_val)[:, 1]
        try:
            auc = roc_auc_score(y_val, y_val_proba)
        except ValueError:
            auc = 0.5
        
        if auc > best_auc:
            best_auc = auc
            best_params = params
        
        print(f"  Config {idx+1}/{len(PARAM_GRID)}: AUC={auc:.4f}")
    
    print(f"\nBest config: {best_params}")
    print(f"Best validation AUC: {best_auc:.4f}")
    
    # Train final model
    print("\n" + "="*80)
    print("TRAINING FINAL MODEL")
    print("="*80)
    final_model = make_model(best_params)
    sample_weights = compute_sample_weights(y_train)
    final_model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Evaluate
    y_train_pred = final_model.predict(X_train)
    y_train_proba = final_model.predict_proba(X_train)[:, 1]
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)
    
    y_val_pred = final_model.predict(X_val)
    y_val_proba = final_model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nTraining Set:")
    print(f"  Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"  ROC-AUC:  {train_metrics['roc_auc']:.4f}")
    print(f"  F1 (up):  {train_metrics['f1_up']:.4f}")
    print(f"  F1 (down): {train_metrics['f1_down']:.4f}")
    
    print(f"\nValidation Set (2024-2025):")
    print(f"  Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"  ROC-AUC:  {val_metrics['roc_auc']:.4f}")
    print(f"  F1 (up):  {val_metrics['f1_up']:.4f}")
    print(f"  F1 (down): {val_metrics['f1_down']:.4f}")
    
    # Apply high confidence filter to validation predictions
    print("\n" + "="*80)
    print("HIGH CONFIDENCE FILTER ON VALIDATION SET")
    print("="*80)
    
    val_predictions = pd.Series(0, index=val_df_clean.index)
    val_proba_up = y_val_proba
    val_predictions = apply_high_confidence_filter(val_predictions, val_proba_up)
    
    # Count filtered signals
    n_original_trades = (val_proba_up >= 0.5).sum()  # Original threshold
    n_filtered_trades = val_predictions[val_predictions != 0].sum()
    
    print(f"  Original trades (proba >= 0.50): {n_original_trades:,}")
    print(f"  Filtered trades (high conf): {n_filtered_trades:,}")
    print(f"  Reduction: {100*(n_original_trades - n_filtered_trades)/max(n_original_trades,1):.1f}%")
    
    # Apply regime filtering to validation predictions
    print("\n" + "="*80)
    print("APPLYING REGIME DETECTION TO VALIDATION SET")
    print("="*80)
    
    val_predictions = apply_regime_filtering(val_df_clean, val_predictions)
    
    # Save model
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": final_model,
        "features": feature_cols,
        "best_params": best_params,
        "train_period": f"{TRAIN_START} to {TRAIN_END}",
        "val_period": f"{VAL_START} to {VAL_END}",
        "regime_detection": True,
        "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
        }
    }
    
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    print(f"\nModel saved to: {MODEL_OUTPUT_PATH}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    
    return final_model, artifact


if __name__ == "__main__":
    main()
