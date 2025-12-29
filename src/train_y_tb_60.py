#!/usr/bin/env python3
"""
Train a classifier for y_tb_60 (triple-barrier 60-min direction).

This script:
1. Loads the saved feature matrix (or builds it if not found)
2. Prepares features and labels for binary classification (+1 vs -1)
3. Performs hyperparameter search using walk-forward CV on training block
4. Trains a final HistGradientBoostingClassifier with best params
5. Evaluates on train/val/test and prints metrics
6. Saves model, config, feature importance, and metrics to disk

Usage:
    cd /Users/omar/Desktop/ML && source xauusd_signals/venv/bin/activate && python xauusd_signals/src/train_y_tb_60.py
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

# =============================================================================
# CONFIGURATION
# =============================================================================

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT.parent / "Data"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
FEATURES_FILE = FEATURES_DIR / "xauusd_features_2020_2025.parquet"
MODELS_DIR = PROJECT_ROOT / "models" / "y_tb_60"

# Random seed for reproducibility
RANDOM_SEED = 42

# Label columns to exclude from features
LABEL_COLS = ["y_ret_5", "y_dir_5", "y_ret_15", "y_dir_15", "y_ret_60", "y_dir_60", "y_tb_60", "y_tb_15"]

# Metadata/raw columns to exclude from features
# Original Model #1 configuration - only exclude raw OHLCV and metadata
RAW_COLS = [
    "open", "high", "low", "close", "volume", "bid_price", "ask_price", "vwap", "trades"
]

# Target label - Original 60-minute horizon
TARGET = "y_tb_60"

# Train/Val/Test split ratios (chronological)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Walk-forward CV settings
WF_N_FOLDS = 5
WF_MIN_TRAIN_FRAC = 0.4

# Default model hyperparameters (STRONG regularization to prevent overfitting)
DEFAULT_MODEL_PARAMS = {
    "max_depth": 2,              # Very shallow trees (reduced from 3 to prevent overfitting)
    "learning_rate": 0.01,        # Very slow learning (reduced from 0.02)
    "max_iter": 200,              # Fewer iterations (reduced from 300)
    "min_samples_leaf": 1000,     # Much higher (increased from 500 - requires more samples per leaf)
    "l2_regularization": 0.5,     # Stronger L2 (increased from 0.3)
    "max_leaf_nodes": None,
    "early_stopping": True,
    "validation_fraction": 0.15,   # More validation data for early stopping
    "random_state": RANDOM_SEED,
    "verbose": 0,
}

# Hyperparameter search grid (more conservative to prevent overfitting)
PARAM_GRID = [
    {"max_depth": 2, "min_samples_leaf": 1000, "learning_rate": 0.01, "l2_regularization": 0.5},
    {"max_depth": 2, "min_samples_leaf": 1200, "learning_rate": 0.01, "l2_regularization": 0.6},
    {"max_depth": 2, "min_samples_leaf": 800, "learning_rate": 0.01, "l2_regularization": 0.5},
    {"max_depth": 2, "min_samples_leaf": 1000, "learning_rate": 0.015, "l2_regularization": 0.5},
    {"max_depth": 3, "min_samples_leaf": 1000, "learning_rate": 0.01, "l2_regularization": 0.5},
]


# =============================================================================
# MODEL CREATION
# =============================================================================

def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute balanced sample weights to force class balance.
    
    This tells the model "1 Short Error = N Long Errors" where N = ratio of class frequencies.
    Equivalent to class_weight="balanced" for classifiers that support it.
    
    Args:
        y: Binary labels (0 or 1)
        
    Returns:
        Sample weights array
    """
    from sklearn.utils.class_weight import compute_sample_weight
    
    # Use sklearn's balanced weight computation
    # This gives more weight to the minority class
    weights = compute_sample_weight('balanced', y)
    return weights


def make_model(params: Optional[Dict[str, Any]] = None) -> HistGradientBoostingClassifier:
    """
    Create a HistGradientBoostingClassifier with configurable hyperparameters.
    
    Starts from a regularized default config and allows overrides via params dict.
    
    Note: HistGradientBoostingClassifier doesn't support class_weight directly,
    but we'll compute sample weights during training to balance classes.
    
    Args:
        params: Optional dict of hyperparameter overrides
        
    Returns:
        Configured HistGradientBoostingClassifier instance
    """
    # Start from defaults
    config = DEFAULT_MODEL_PARAMS.copy()
    
    # FORCE BALANCE: We'll compute sample weights during training
    # This tells the model "1 Short Error = N Long Errors" where N = ratio of class frequencies
    # HistGradientBoostingClassifier doesn't have class_weight parameter,
    # so we'll handle this via sample_weight in fit()
    
    # Apply overrides
    if params:
        for key, value in params.items():
            if key in config or key in ["max_depth", "learning_rate", "max_iter", 
                                         "min_samples_leaf", "l2_regularization",
                                         "max_leaf_nodes", "early_stopping",
                                         "validation_fraction"]:
                config[key] = value
    
    return HistGradientBoostingClassifier(**config)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_feature_matrix() -> pd.DataFrame:
    """
    Load the saved feature matrix, or build it if not found.
    
    Returns:
        DataFrame with features and labels, sorted by timestamp.
    """
    # Check if saved file has the required target column
    if FEATURES_FILE.exists():
        print(f"Checking features file: {FEATURES_FILE}")
        df = pd.read_parquet(FEATURES_FILE)
        
        if TARGET in df.columns:
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                    df = df.set_index("timestamp")
            
            df = df.sort_index()
            print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
            return df
        else:
            print(f"  File exists but missing '{TARGET}'. Rebuilding...")
    
    # Build features
    print("Building features from raw data...")
    
    from features_complete import build_complete_features
    
    # Load data from 2020-2025 (all available data)
    print("Loading data from 2020-2025...")
    import time
    start_time = time.time()
    
    bars_list = []
    quotes_list = []
    
    for year in range(2020, 2026):  # 2020 to 2025
        bars_path = DATA_DIR / "ohlcv_minute" / f"XAUUSD_minute_{year}.parquet"
        quotes_path = DATA_DIR / "quotes" / f"XAUUSD_quotes_{year}.parquet"
        
        if bars_path.exists():
            print(f"  Loading {year} bars...", end=" ", flush=True)
            year_start = time.time()
            year_bars = pd.read_parquet(bars_path)
            if "timestamp" in year_bars.columns:
                year_bars["timestamp"] = pd.to_datetime(year_bars["timestamp"], utc=True)
                year_bars = year_bars.set_index("timestamp")
            bars_list.append(year_bars)
            print(f"✓ ({len(year_bars):,} rows, {time.time()-year_start:.1f}s)")
        else:
            print(f"  Warning: {bars_path} not found, skipping {year}")
        
        if quotes_path.exists():
            print(f"  Loading {year} quotes...", end=" ", flush=True)
            year_start = time.time()
            year_quotes = pd.read_parquet(quotes_path)
            if "timestamp" in year_quotes.columns:
                year_quotes["timestamp"] = pd.to_datetime(year_quotes["timestamp"], utc=True)
                year_quotes = year_quotes.set_index("timestamp")
            quotes_list.append(year_quotes)
            print(f"✓ ({len(year_quotes):,} rows, {time.time()-year_start:.1f}s)")
    
    if not bars_list:
        raise FileNotFoundError("No bar data files found for 2024-2025")
    
    # Concatenate all years
    print(f"\n  Concatenating {len(bars_list)} years of bars...", end=" ", flush=True)
    concat_start = time.time()
    bars = pd.concat(bars_list).sort_index()
    print(f"✓ ({time.time()-concat_start:.1f}s)")
    print(f"  Total bars: {len(bars):,} rows from {bars.index.min()} to {bars.index.max()}")
    
    quotes = None
    if quotes_list:
        print(f"  Concatenating {len(quotes_list)} years of quotes...", end=" ", flush=True)
        concat_start = time.time()
        quotes = pd.concat(quotes_list).sort_index()
        print(f"✓ ({time.time()-concat_start:.1f}s)")
        print(f"  Total quotes: {len(quotes):,} rows from {quotes.index.min()} to {quotes.index.max()}")
    
    print(f"\n  Data loading complete: {time.time()-start_time:.1f}s total")
    
    df = build_complete_features(bars, quotes, horizons=[60])  # Only generate y_tb_60 labels
    
    # Save for future use
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FEATURES_FILE)
    print(f"  Saved features to: {FEATURES_FILE}")
    
    return df


# =============================================================================
# FEATURE PREPARATION
# =============================================================================

def get_feature_columns(df: pd.DataFrame, prioritize_effective: bool = True) -> List[str]:
    """
    Get list of numeric feature columns, excluding labels and raw columns.
    
    If prioritize_effective=True, will prioritize features that correlate with profitability.
    """
    features = []
    
    for col in df.columns:
        if col in LABEL_COLS:
            continue
        if col in RAW_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        features.append(col)
    
    # Prioritize effective features if available
    if prioritize_effective:
        effective_features_file = PROJECT_ROOT / "data" / "effective_features.txt"
        if effective_features_file.exists():
            with open(effective_features_file, 'r') as f:
                effective_features = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            # Reorder: effective features first
            effective_set = set(effective_features)
            effective_in_data = [f for f in effective_features if f in features]
            other_features = [f for f in features if f not in effective_set]
            
            features = effective_in_data + other_features
            print(f"  Prioritized {len(effective_in_data)} effective features")
    
    return features


def prepare_data(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Prepare X and y for training.
    
    - Drops rows with NaN in features or target
    - Drops y_tb_60 == 0 (very rare class, ~0.1%)
    - Maps labels to binary: -1 -> 0, +1 -> 1
    """
    print("\nPreparing data...")
    
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in data")
    
    # Drop rows with NaN
    cols_to_check = feature_cols + [TARGET]
    initial_rows = len(df)
    df_clean = df.dropna(subset=cols_to_check)
    dropped_nan = initial_rows - len(df_clean)
    print(f"  Dropped {dropped_nan:,} rows with NaN")
    
    # Label distribution before filtering
    print(f"\n  Label distribution (before filtering):")
    dist = df_clean[TARGET].value_counts()
    for val in sorted(dist.index):
        pct = 100 * dist[val] / len(df_clean)
        print(f"    {int(val):+d}: {dist[val]:,} ({pct:.2f}%)")
    
    # Drop y_tb_60 == 0
    n_zeros = (df_clean[TARGET] == 0).sum()
    df_clean = df_clean[df_clean[TARGET] != 0]
    print(f"\n  Dropped {n_zeros:,} rows where y_tb_60 == 0")
    
    # Final distribution
    print(f"\n  Label distribution (after filtering):")
    dist = df_clean[TARGET].value_counts()
    for val in sorted(dist.index):
        pct = 100 * dist[val] / len(df_clean)
        print(f"    {int(val):+d}: {dist[val]:,} ({pct:.2f}%)")
    
    # Extract features and labels
    X = df_clean[feature_cols].values
    y_raw = df_clean[TARGET].values
    
    # Map labels: -1 -> 0, +1 -> 1
    y = np.where(y_raw == 1, 1, 0)
    
    print(f"\n  Final: {len(df_clean):,} samples, {len(feature_cols)} features")
    print(f"  Binary labels: class 0 (down) = {(y == 0).sum():,}, class 1 (up) = {(y == 1).sum():,}")
    
    return X, y, df_clean


# =============================================================================
# TIME-BASED SPLITS
# =============================================================================

def make_time_splits(
    n: int,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create chronological train/val/test indices.
    
    NO SHUFFLING - preserves time order.
    """
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)
    
    return train_idx, val_idx, test_idx


def generate_walk_forward_splits(
    n_train: int,
    n_folds: int = WF_N_FOLDS,
    min_train_frac: float = WF_MIN_TRAIN_FRAC,
    val_frac: float = 0.10
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward CV splits for the training block.
    
    Each fold has a growing training window and a small forward validation window.
    
    Args:
        n_train: Number of samples in training block
        n_folds: Number of walk-forward folds
        min_train_frac: Minimum fraction of data for first training set
        val_frac: Fraction of used segment for validation (~10%)
        
    Returns:
        List of (train_indices, val_indices) relative to training block
    """
    splits = []
    
    # Compute fold boundaries
    min_train_size = int(n_train * min_train_frac)
    remaining = n_train - min_train_size
    fold_size = remaining // n_folds
    
    for fold in range(n_folds):
        train_end = min_train_size + fold_size * (fold + 1)
        val_size = max(int(train_end * val_frac), 100)
        
        if train_end > n_train:
            train_end = n_train
        
        val_start = train_end - val_size
        
        train_idx = np.arange(0, val_start)
        val_idx = np.arange(val_start, train_end)
        
        splits.append((train_idx, val_idx))
    
    return splits


# =============================================================================
# METRICS COMPUTATION
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_down": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "precision_up": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_down": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_up": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_down": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_up": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    except ValueError:
        metrics["roc_auc"] = np.nan
    
    return metrics


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    params: Optional[Dict] = None
) -> Tuple[HistGradientBoostingClassifier, Dict[str, float]]:
    """
    Train model and evaluate on a dataset.
    """
    model = make_model(params)
    # Use balanced sample weights to force class balance
    sample_weights = compute_sample_weights(y_train)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    y_pred = model.predict(X_eval)
    y_proba = model.predict_proba(X_eval)[:, 1]
    
    metrics = compute_metrics(y_eval, y_pred, y_proba)
    
    return model, metrics


# =============================================================================
# HYPERPARAMETER SEARCH
# =============================================================================

def hyperparameter_search_walk_forward(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_folds: int = WF_N_FOLDS
) -> Dict[str, Any]:
    """
    Run hyperparameter search using walk-forward CV on the training block.
    
    Uses ROC-AUC as the primary metric. Ties broken by model complexity
    (prefer lower max_depth, higher min_samples_leaf).
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_folds: Number of walk-forward folds
        
    Returns:
        Best params dict
    """
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER SEARCH (walk-forward CV, {n_folds} folds)")
    print(f"{'='*70}")
    
    n_train = len(X_train)
    splits = generate_walk_forward_splits(n_train, n_folds=n_folds)
    
    results = []
    
    for idx, params in enumerate(PARAM_GRID):
        fold_aucs = []
        
        for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(splits):
            X_fold_train = X_train[fold_train_idx]
            y_fold_train = y_train[fold_train_idx]
            X_fold_val = X_train[fold_val_idx]
            y_fold_val = y_train[fold_val_idx]
            
            # Train and evaluate with balanced sample weights
            model = make_model(params)
            sample_weights = compute_sample_weights(y_fold_train)
            model.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
            
            y_proba = model.predict_proba(X_fold_val)[:, 1]
            
            try:
                auc = roc_auc_score(y_fold_val, y_proba)
            except ValueError:
                auc = 0.5  # Default for degenerate cases
            
            fold_aucs.append(auc)
        
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        
        results.append({
            "idx": idx,
            "params": params,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "fold_aucs": fold_aucs,
        })
        
        print(f"  Config {idx+1}/{len(PARAM_GRID)}: "
              f"depth={params.get('max_depth', 'def'):2}, "
              f"leaf={params.get('min_samples_leaf', 'def'):4}, "
              f"lr={params.get('learning_rate', 'def'):.3f}, "
              f"l2={params.get('l2_regularization', 'def'):.2f} "
              f"-> AUC={mean_auc:.4f} ± {std_auc:.4f}")
    
    # Sort by mean_auc (desc), then by complexity (lower max_depth, higher min_samples_leaf)
    results.sort(
        key=lambda x: (
            -x["mean_auc"],
            x["params"].get("max_depth", 999),
            -x["params"].get("min_samples_leaf", 0)
        )
    )
    
    best = results[0]
    
    # Print summary table
    print(f"\n{'-'*70}")
    print("SEARCH RESULTS (sorted by mean AUC)")
    print(f"{'-'*70}")
    print(f"\n  {'Idx':>3}  {'max_depth':>9}  {'min_leaf':>8}  {'lr':>6}  {'l2':>5}  {'mean_AUC':>9}  {'std_AUC':>8}")
    print(f"  {'-'*65}")
    
    for r in results:
        p = r["params"]
        marker = " *" if r["idx"] == best["idx"] else ""
        print(f"  {r['idx']+1:>3}  {p.get('max_depth', '-'):>9}  {p.get('min_samples_leaf', '-'):>8}  "
              f"{p.get('learning_rate', 0):.4f}  {p.get('l2_regularization', 0):.2f}  "
              f"{r['mean_auc']:>9.4f}  {r['std_auc']:>8.4f}{marker}")
    
    print(f"\n  Best config (idx={best['idx']+1}): {best['params']}")
    print(f"  Best mean AUC: {best['mean_auc']:.4f} ± {best['std_auc']:.4f}")
    
    return best["params"]


# =============================================================================
# WALK-FORWARD CV (for reporting with chosen config)
# =============================================================================

def run_walk_forward_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict] = None,
    n_folds: int = WF_N_FOLDS
) -> List[Dict[str, float]]:
    """
    Run walk-forward CV with a specific config for detailed reporting.
    """
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD CV WITH BEST CONFIG ({n_folds} folds)")
    print(f"{'='*60}")
    
    if params:
        print(f"\n  Using params: {params}")
    
    n_train = len(X_train)
    splits = generate_walk_forward_splits(n_train, n_folds=n_folds)
    
    fold_metrics = []
    
    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(splits):
        X_fold_train = X_train[fold_train_idx]
        y_fold_train = y_train[fold_train_idx]
        X_fold_val = X_train[fold_val_idx]
        y_fold_val = y_train[fold_val_idx]
        
        model, metrics = train_and_evaluate(
            X_fold_train, y_fold_train,
            X_fold_val, y_fold_val,
            params=params
        )
        fold_metrics.append(metrics)
        
        print(f"\n  Fold {fold_idx + 1}: train={len(fold_train_idx):,}, val={len(fold_val_idx):,}")
        print(f"    Acc={metrics['accuracy']:.4f}, AUC={metrics['roc_auc']:.4f}, "
              f"F1_down={metrics['f1_down']:.4f}, F1_up={metrics['f1_up']:.4f}")
    
    # Summary statistics
    print(f"\n{'-'*60}")
    print("WALK-FORWARD CV SUMMARY (tuned config)")
    print(f"{'-'*60}")
    
    metric_names = ["accuracy", "roc_auc", "precision_down", "precision_up", 
                    "recall_down", "recall_up", "f1_down", "f1_up"]
    
    print(f"\n  {'Metric':<15} {'Mean':>8} {'Std':>8}")
    print(f"  {'-'*35}")
    
    for metric in metric_names:
        values = [m[metric] for m in fold_metrics]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {metric:<15} {mean_val:>8.4f} {std_val:>8.4f}")
    
    return fold_metrics


# =============================================================================
# PRINTING UTILITIES
# =============================================================================

def print_metrics(metrics: Dict[str, float], split_name: str, cm: np.ndarray = None) -> None:
    """Print metrics in a clear format."""
    print(f"\n{'='*60}")
    print(f"{split_name.upper()} METRICS")
    print(f"{'='*60}")
    
    print(f"\n  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
    
    print(f"\n  Per-class metrics:")
    print(f"                Precision    Recall    F1")
    print(f"    Down (-1):    {metrics['precision_down']:.4f}      {metrics['recall_down']:.4f}    {metrics['f1_down']:.4f}")
    print(f"    Up   (+1):    {metrics['precision_up']:.4f}      {metrics['recall_up']:.4f}    {metrics['f1_up']:.4f}")
    
    if cm is not None:
        print(f"\n  Confusion Matrix:")
        print(f"                 Pred Down  Pred Up")
        print(f"    True Down:     {cm[0, 0]:6,}    {cm[0, 1]:6,}")
        print(f"    True Up:       {cm[1, 0]:6,}    {cm[1, 1]:6,}")


def print_overfit_check(train_metrics: Dict, val_metrics: Dict, test_metrics: Dict) -> None:
    """Print quick overfitting check."""
    print(f"\n{'='*60}")
    print("OVERFITTING CHECK")
    print(f"{'='*60}")
    
    print(f"\n  Metric        Train     Val      Test     Gap (Train-Test)")
    print(f"  {'-'*55}")
    
    for metric in ["accuracy", "roc_auc"]:
        train_val = train_metrics[metric]
        val_val = val_metrics[metric]
        test_val = test_metrics[metric]
        gap = train_val - test_val
        
        status = "✓" if gap < 0.05 else "⚠" if gap < 0.10 else "❌"
        
        print(f"  {metric:12}  {train_val:.4f}   {val_val:.4f}   {test_val:.4f}   {gap:+.4f} {status}")


# =============================================================================
# ARTIFACT SAVING
# =============================================================================

def save_artifacts(
    model: HistGradientBoostingClassifier,
    feature_cols: List[str],
    best_params: Dict[str, Any],
    train_metrics: Dict,
    val_metrics: Dict,
    test_metrics: Dict,
    wf_metrics: List[Dict] = None,
    X_val: np.ndarray = None,
    y_val: np.ndarray = None
) -> List[Path]:
    """
    Save model and artifacts to disk.
    """
    print(f"\n{'='*60}")
    print("SAVING ARTIFACTS")
    print(f"{'='*60}")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    
    # 1. Save model
    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(model, model_path)
    print(f"\n  ✓ Model saved: {model_path}")
    saved_paths.append(model_path)
    
    # 2. Save config (with tuned params)
    full_params = DEFAULT_MODEL_PARAMS.copy()
    full_params.update(best_params)
    
    config = {
        "feature_columns": feature_cols,
        "label_column": TARGET,
        "model_type": "HistGradientBoostingClassifier",
        "model_params": full_params,
        "tuned_params": best_params,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "walk_forward_cv": {
            "n_folds": WF_N_FOLDS,
            "min_train_frac": WF_MIN_TRAIN_FRAC,
        },
        "random_seed": RANDOM_SEED,
    }
    
    config_path = MODELS_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Config saved: {config_path}")
    saved_paths.append(config_path)
    
    # 3. Save feature importance
    importance_path = MODELS_DIR / "feature_importance.csv"
    
    if X_val is not None and y_val is not None:
        try:
            from sklearn.inspection import permutation_importance
            
            print("  Computing permutation importance...")
            perm_imp = permutation_importance(
                model, X_val, y_val, 
                n_repeats=5, 
                random_state=RANDOM_SEED, 
                n_jobs=-1
            )
            
            importance_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": perm_imp.importances_mean,
                "importance_std": perm_imp.importances_std,
            }).sort_values("importance", ascending=False)
            
            importance_df.to_csv(importance_path, index=False)
            print(f"  ✓ Feature importance saved: {importance_path}")
            saved_paths.append(importance_path)
            
        except Exception as e:
            print(f"  ⚠ Could not compute feature importance: {e}")
    
    # 4. Save metrics
    metrics_data = {
        "train": {k: float(v) if not np.isnan(v) else None for k, v in train_metrics.items()},
        "val": {k: float(v) if not np.isnan(v) else None for k, v in val_metrics.items()},
        "test": {k: float(v) if not np.isnan(v) else None for k, v in test_metrics.items()},
    }
    
    if wf_metrics:
        wf_summary = {}
        for metric in ["accuracy", "roc_auc", "f1_down", "f1_up"]:
            values = [m[metric] for m in wf_metrics]
            wf_summary[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        metrics_data["walk_forward_cv"] = wf_summary
    
    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  ✓ Metrics saved: {metrics_path}")
    saved_paths.append(metrics_path)
    
    return saved_paths


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main training pipeline."""
    print("=" * 70)
    print("TRAIN y_tb_60 CLASSIFIER (with hyperparameter tuning)")
    print("=" * 70)
    
    # Load data
    df = load_feature_matrix()
    
    # Get feature columns
    feature_cols = get_feature_columns(df, prioritize_effective=False)  # Use all features
    print(f"\nUsing {len(feature_cols)} features:")
    for i, col in enumerate(feature_cols):
        print(f"  {i+1:2d}. {col}")
    
    # Prepare data
    X, y, df_clean = prepare_data(df, feature_cols)
    
    # Time-based split
    train_idx, val_idx, test_idx = make_time_splits(len(X))
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # Sanity check: print split sizes
    print(f"\n{'='*60}")
    print("DATA SPLIT SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Total samples:  {len(X):,}")
    print(f"  Train samples:  {len(X_train):,} ({100*len(X_train)/len(X):.1f}%)")
    print(f"  Val samples:    {len(X_val):,} ({100*len(X_val)/len(X):.1f}%)")
    print(f"  Test samples:   {len(X_test):,} ({100*len(X_test)/len(X):.1f}%)")
    
    # Hyperparameter search on training block
    best_params = hyperparameter_search_walk_forward(X_train, y_train, n_folds=WF_N_FOLDS)
    
    # Run detailed walk-forward CV with best config
    wf_metrics = run_walk_forward_cv(X_train, y_train, params=best_params, n_folds=WF_N_FOLDS)
    
    # Train final model on full training set with best params
    print(f"\n{'='*60}")
    print("TRAINING FINAL MODEL (tuned)")
    print(f"{'='*60}")
    print(f"\n  Using hyperparameters: {best_params}")
    
    final_model = make_model(best_params)
    # Use balanced sample weights to force class balance (fix long bias)
    print("  Computing balanced sample weights to force class balance...")
    sample_weights = compute_sample_weights(y_train)
    class_counts = np.bincount(y_train)
    print(f"    Class 0 (down): {class_counts[0]:,} samples, weight: {sample_weights[y_train == 0][0]:.4f}")
    print(f"    Class 1 (up):   {class_counts[1]:,} samples, weight: {sample_weights[y_train == 1][0]:.4f}")
    final_model.fit(X_train, y_train, sample_weight=sample_weights)
    print("  ✓ Final model trained on full training set with balanced weights")
    
    # Evaluate on all splits
    y_train_pred = final_model.predict(X_train)
    y_train_proba = final_model.predict_proba(X_train)[:, 1]
    train_metrics = compute_metrics(y_train, y_train_pred, y_train_proba)
    train_cm = confusion_matrix(y_train, y_train_pred)
    
    y_val_pred = final_model.predict(X_val)
    y_val_proba = final_model.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_proba)
    val_cm = confusion_matrix(y_val, y_val_pred)
    
    y_test_pred = final_model.predict(X_test)
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    test_metrics = compute_metrics(y_test, y_test_pred, y_test_proba)
    test_cm = confusion_matrix(y_test, y_test_pred)
    
    # Print results
    print_metrics(train_metrics, "Train (tuned)", train_cm)
    print_metrics(val_metrics, "Validation (tuned)", val_cm)
    print_metrics(test_metrics, "Test (tuned)", test_cm)
    
    # Overfitting check
    print_overfit_check(train_metrics, val_metrics, test_metrics)
    
    # Feature importance (top 15)
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE (Top 15)")
    print(f"{'='*60}")
    
    try:
        from sklearn.inspection import permutation_importance
        
        print("\n  Computing permutation importance on validation set...")
        perm_imp = permutation_importance(
            final_model, X_val, y_val, 
            n_repeats=5, 
            random_state=RANDOM_SEED, 
            n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": perm_imp.importances_mean,
            "std": perm_imp.importances_std,
        }).sort_values("importance", ascending=False)
        
        print(f"\n  {'Feature':<25}  Importance")
        print(f"  {'-'*40}")
        for _, row in importance_df.head(15).iterrows():
            print(f"  {row['feature']:<25}  {row['importance']:.4f} ± {row['std']:.4f}")
            
    except Exception as e:
        print(f"\n  Could not compute feature importance: {e}")
    
    # Save artifacts
    saved_paths = save_artifacts(
        model=final_model,
        feature_cols=feature_cols,
        best_params=best_params,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        wf_metrics=wf_metrics,
        X_val=X_val,
        y_val=y_val,
    )
    
    # Print saved paths
    print(f"\n{'='*60}")
    print("SAVED ARTIFACTS")
    print(f"{'='*60}")
    for path in saved_paths:
        print(f"  {path}")
    
    # Save bundled artifact for backtest script
    model_artifact_path = PROJECT_ROOT / "models" / "y_tb_60_hgb_tuned.joblib"
    model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        "model": final_model,
        "features": feature_cols,
        "best_params": best_params,
    }
    joblib.dump(artifact, model_artifact_path)
    print(f"\n  ✓ Bundled artifact saved: {model_artifact_path}")
    
    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Tuned hyperparameters: {best_params}")
    print(f"\n  Final model (tuned) – validation AUC: {val_metrics['roc_auc']:.4f}, test AUC: {test_metrics['roc_auc']:.4f}")
    print(f"  Final model (tuned) – validation Acc: {val_metrics['accuracy']:.4f}, test Acc: {test_metrics['accuracy']:.4f}")
    
    train_test_gap = train_metrics['roc_auc'] - test_metrics['roc_auc']
    print(f"\n  Overfitting (Train-Test AUC gap): {train_test_gap:+.4f}", end="")
    if train_test_gap < 0.05:
        print(" ✓ (good)")
    elif train_test_gap < 0.10:
        print(" ⚠ (moderate)")
    else:
        print(" ❌ (high)")
    
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")
    
    return final_model, best_params, train_metrics, val_metrics, test_metrics


if __name__ == "__main__":
    main()
