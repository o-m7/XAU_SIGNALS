"""
Model Training Module - Multiclass Classification with Class Balancing.

Trains models to predict microstructure-based directional labels.
Uses class weights to handle imbalance and prevent collapse to majority class.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.utils.class_weight import compute_sample_weight

from .feature_engineering import get_microstructure_features


# =============================================================================
# CONFIGURATION
# =============================================================================

XGBOOST_PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "max_depth": 4,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "min_child_weight": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_training_data(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare feature matrix and labels for training.
    
    Drops rows with NaN in any feature or label.
    Maps labels from {-1, 0, 1} to {0, 1, 2} for XGBoost.
    """
    if feature_cols is None:
        feature_cols = get_microstructure_features()
    
    # Filter to available columns
    available = [c for c in feature_cols if c in df.columns]
    
    if len(available) == 0:
        raise ValueError("No feature columns found in DataFrame")
    
    # Drop NaN rows
    work_df = df[available + [label_col]].dropna()
    
    X = work_df[available].values
    y_raw = work_df[label_col].values
    
    # Map labels: -1 -> 0, 0 -> 1, +1 -> 2
    y = y_raw + 1
    
    return X, y, available


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute sample weights for class balancing.
    
    Uses sklearn's compute_sample_weight for balanced weighting.
    """
    return compute_sample_weight("balanced", y)


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_model_for_horizon(
    df: pd.DataFrame,
    horizon: int,
    model_path: str,
    feature_cols: Optional[List[str]] = None,
    train_end: str = "2023-12-31",
    val_end: str = "2024-06-30",
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a multiclass model for a single horizon.
    
    Uses class weights to prevent collapse to majority class.
    Optimizes for macro-F1.
    """
    label_col = f"y_{horizon}m"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training model for {horizon}m horizon")
        print(f"{'='*60}")
    
    # Time-based split
    train_end_dt = pd.Timestamp(train_end, tz='UTC')
    val_end_dt = pd.Timestamp(val_end, tz='UTC')
    
    train_df = df[df.index <= train_end_dt]
    val_df = df[(df.index > train_end_dt) & (df.index <= val_end_dt)]
    test_df = df[df.index > val_end_dt]
    
    if verbose:
        print(f"Train: {len(train_df):,} rows ({train_df.index.min().date()} to {train_df.index.max().date()})")
        print(f"Val:   {len(val_df):,} rows")
        print(f"Test:  {len(test_df):,} rows")
    
    # Prepare data
    X_train, y_train, used_features = prepare_training_data(train_df, label_col, feature_cols)
    X_val, y_val, _ = prepare_training_data(val_df, label_col, used_features)
    X_test, y_test, _ = prepare_training_data(test_df, label_col, used_features)
    
    if verbose:
        print(f"\nFeatures: {len(used_features)}")
        print(f"Train samples: {len(y_train):,}")
        
        # Label distribution
        for label, name in [(0, "SHORT"), (1, "FLAT"), (2, "LONG")]:
            count = (y_train == label).sum()
            pct = 100 * count / len(y_train)
            print(f"  {name}: {count:,} ({pct:.1f}%)")
    
    # Compute sample weights for class balancing
    sample_weights = compute_class_weights(y_train)
    
    if verbose:
        print(f"\nUsing balanced class weights")
    
    # Create and train model
    model = XGBClassifier(**XGBOOST_PARAMS)
    
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    def compute_metrics(y_true, y_pred, name):
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro"),
            "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        }
        
        if verbose:
            print(f"\n{name} Metrics:")
            print(f"  Accuracy:  {metrics['accuracy']:.3f}")
            print(f"  Macro F1:  {metrics['macro_f1']:.3f}")
            print(f"  Precision: {metrics['macro_precision']:.3f}")
            print(f"  Recall:    {metrics['macro_recall']:.3f}")
        
        return metrics
    
    train_metrics = compute_metrics(y_train, y_train_pred, "Train")
    val_metrics = compute_metrics(y_val, y_val_pred, "Validation")
    test_metrics = compute_metrics(y_test, y_test_pred, "Test")
    
    # Confusion matrix
    if verbose:
        print(f"\nTest Confusion Matrix:")
        print("           Pred SHORT  Pred FLAT  Pred LONG")
        cm = confusion_matrix(y_test, y_test_pred)
        labels = ["SHORT", "FLAT ", "LONG "]
        for i, row in enumerate(cm):
            print(f"  True {labels[i]}: {row}")
        
        print(f"\nClassification Report (Test):")
        print(classification_report(
            y_test, y_test_pred,
            target_names=["SHORT", "FLAT", "LONG"],
            zero_division=0
        ))
    
    # Save model
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump({
        "model": model,
        "features": used_features,
        "horizon": horizon,
        "label_mapping": {-1: 0, 0: 1, 1: 2},  # Original to XGBoost
        "inverse_mapping": {0: -1, 1: 0, 2: 1},  # XGBoost to original
    }, model_path)
    
    if verbose:
        print(f"\nModel saved: {model_path}")
    
    return {
        "horizon": horizon,
        "model_path": str(model_path),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "n_features": len(used_features),
    }


def train_all_horizon_models(
    df: pd.DataFrame,
    model_dir: str,
    feature_cols: Optional[List[str]] = None,
    train_end: str = "2023-12-31",
    val_end: str = "2024-06-30",
    verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Train models for all horizons (5m, 15m, 30m).
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for horizon in [5, 15, 30]:
        model_path = model_dir / f"model_{horizon}m.pkl"
        result = train_model_for_horizon(
            df=df,
            horizon=horizon,
            model_path=str(model_path),
            feature_cols=feature_cols,
            train_end=train_end,
            val_end=val_end,
            verbose=verbose
        )
        results[f"{horizon}m"] = result
    
    if verbose:
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"{'Horizon':<10} {'Train F1':<12} {'Val F1':<12} {'Test F1':<12}")
        print("-" * 46)
        for h, r in results.items():
            print(f"{h:<10} {r['train_metrics']['macro_f1']:<12.3f} "
                  f"{r['val_metrics']['macro_f1']:<12.3f} "
                  f"{r['test_metrics']['macro_f1']:<12.3f}")
    
    return results


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_path: str) -> Tuple[XGBClassifier, List[str], Dict]:
    """Load a trained model."""
    data = joblib.load(model_path)
    return data["model"], data["features"], data


# =============================================================================
# LEGACY COMPATIBILITY
# =============================================================================

def time_series_train_test_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """Simple time-based split."""
    n = len(df)
    split_idx = int(n * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
