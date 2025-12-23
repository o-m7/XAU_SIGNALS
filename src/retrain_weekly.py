#!/usr/bin/env python3
"""
Weekly Model Retraining Script with Reinforcement Learning.

This script:
1. Downloads the latest XAUUSD data from Polygon
2. Loads trade feedback (actual signal outcomes)
3. Rebuilds features with fresh data
4. Retrains the model with performance-weighted samples
5. Validates against hold-out set
6. Only deploys if new model beats current model

Schedule to run every Saturday:
    crontab -e
    0 6 * * 6 cd /Users/omar/Desktop/ML/xauusd_signals && ./venv/bin/python src/retrain_weekly.py

Usage:
    python src/retrain_weekly.py [--dry-run] [--force]
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import shutil

import numpy as np
import pandas as pd
import joblib
import requests

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features_complete import build_complete_features

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
FEEDBACK_DIR = DATA_DIR / "feedback"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model paths
CURRENT_MODEL_PATH = MODELS_DIR / "y_tb_60_hgb_tuned.joblib"
BACKUP_DIR = MODELS_DIR / "backups"

# Data settings
POLYGON_BASE_URL = "https://api.polygon.io"
SYMBOL = "C:XAUUSD"
LOOKBACK_DAYS = 365  # 1 year of training data
MIN_BARS_REQUIRED = 50000  # Minimum bars needed

# Training settings
TARGET = "y_tb_60"
TEST_SIZE = 0.15
VAL_SIZE = 0.15
TRAIN_SIZE = 1.0 - TEST_SIZE - VAL_SIZE

# Performance thresholds
MIN_AUC_IMPROVEMENT = 0.005  # 0.5% improvement required to deploy
MIN_AUC_THRESHOLD = 0.52    # Minimum AUC to deploy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGS_DIR / "retrain.log")
    ]
)
logger = logging.getLogger("Retrain")

# Ensure directories exist
for d in [DATA_DIR, FEATURES_DIR, MODELS_DIR, FEEDBACK_DIR, LOGS_DIR, BACKUP_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA DOWNLOAD
# =============================================================================

def get_polygon_api_key() -> str:
    """Get Polygon API key from environment."""
    key = os.environ.get("POLYGON_API_KEY")
    if not key:
        # Try loading from .env
        env_file = PROJECT_ROOT / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("POLYGON_API_KEY="):
                        key = line.strip().split("=", 1)[1]
                        break
    
    if not key:
        raise ValueError("POLYGON_API_KEY not found in environment or .env file")
    
    return key


def download_minute_bars(
    api_key: str,
    start_date: datetime,
    end_date: datetime,
    output_path: Path
) -> pd.DataFrame:
    """
    Download minute bars from Polygon.
    
    Args:
        api_key: Polygon API key
        start_date: Start date
        end_date: End date
        output_path: Path to save CSV
        
    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Downloading minute bars: {start_date.date()} to {end_date.date()}")
    
    all_bars = []
    current_start = start_date
    
    while current_start < end_date:
        # Polygon allows max 50k results per request
        current_end = min(current_start + timedelta(days=30), end_date)
        
        url = (
            f"{POLYGON_BASE_URL}/v2/aggs/ticker/{SYMBOL}/range/1/minute/"
            f"{current_start.strftime('%Y-%m-%d')}/{current_end.strftime('%Y-%m-%d')}"
        )
        
        params = {
            "apiKey": api_key,
            "limit": 50000,
            "sort": "asc"
        }
        
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            if data.get("results"):
                all_bars.extend(data["results"])
                logger.info(f"  Downloaded {len(data['results'])} bars for {current_start.date()}")
            
        except Exception as e:
            logger.error(f"Error downloading bars for {current_start.date()}: {e}")
        
        current_start = current_end + timedelta(days=1)
    
    if not all_bars:
        raise ValueError("No bars downloaded")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_bars)
    df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.rename(columns={
        "o": "open",
        "h": "high", 
        "l": "low",
        "c": "close",
        "v": "volume",
        "vw": "vwap",
        "n": "num_trades"
    })
    
    df = df.set_index("timestamp").sort_index()
    df = df[["open", "high", "low", "close", "volume"]]
    df = df[~df.index.duplicated(keep='first')]
    
    # Save
    df.to_csv(output_path)
    logger.info(f"Saved {len(df)} bars to {output_path}")
    
    return df


def download_quotes(
    api_key: str,
    start_date: datetime,
    end_date: datetime,
    output_path: Path
) -> pd.DataFrame:
    """
    Download forex quotes from Polygon.
    
    Note: This uses the quotes endpoint which may have different availability.
    Falls back to using bars for bid/ask estimation if quotes not available.
    """
    logger.info(f"Downloading quotes: {start_date.date()} to {end_date.date()}")
    
    # For forex, quotes endpoint may not be available on all plans
    # We'll create synthetic quotes from bars with typical spread
    
    bars_path = output_path.parent / "xauusd_bars_latest.csv"
    if bars_path.exists():
        bars = pd.read_csv(bars_path, index_col="timestamp", parse_dates=True)
    else:
        logger.warning("Bars not found, downloading first...")
        bars = download_minute_bars(api_key, start_date, end_date, bars_path)
    
    # Create synthetic quotes with typical gold spread (~$0.30-0.50)
    df = pd.DataFrame(index=bars.index)
    spread = 0.40  # Typical spread for gold
    
    df["bid_price"] = bars["close"] - spread / 2
    df["ask_price"] = bars["close"] + spread / 2
    
    df.to_csv(output_path)
    logger.info(f"Created synthetic quotes for {len(df)} bars")
    
    return df


# =============================================================================
# FEEDBACK INTEGRATION (Reinforcement Learning)
# =============================================================================

def load_trade_feedback() -> pd.DataFrame:
    """
    Load trade feedback from stored signal outcomes.
    
    Feedback file format (JSON lines):
    {
        "timestamp": "2024-01-15T10:30:00Z",
        "signal": "LONG",
        "entry_price": 2050.50,
        "tp": 2070.50,
        "sl": 2030.50,
        "outcome": "TP",  # "TP", "SL", or "TIMEOUT"
        "exit_price": 2070.50,
        "r_multiple": 1.0,
        "duration_minutes": 45
    }
    """
    feedback_file = FEEDBACK_DIR / "trade_outcomes.jsonl"
    
    if not feedback_file.exists():
        logger.info("No trade feedback file found - starting fresh")
        return pd.DataFrame()
    
    records = []
    with open(feedback_file) as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    if not records:
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    logger.info(f"Loaded {len(df)} trade feedback records")
    logger.info(f"  Win rate: {(df['outcome'] == 'TP').mean()*100:.1f}%")
    
    return df


def compute_sample_weights(
    features_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    recency_decay: float = 0.95
) -> np.ndarray:
    """
    Compute sample weights for reinforcement learning.
    
    Weights are based on:
    1. Recency (newer data weighted higher)
    2. Trade feedback (samples near winning trades weighted higher)
    3. Regime importance (volatile periods weighted higher)
    
    Args:
        features_df: Full feature DataFrame with index
        feedback_df: Trade feedback DataFrame
        recency_decay: Decay factor for older samples (0.9 = 10% decay per month)
        
    Returns:
        Array of sample weights
    """
    n = len(features_df)
    weights = np.ones(n)
    
    # 1. Recency weighting (exponential decay)
    if hasattr(features_df.index, 'to_pydatetime'):
        timestamps = features_df.index.to_pydatetime()
        max_ts = max(timestamps)
        
        for i, ts in enumerate(timestamps):
            # Days since most recent
            days_old = (max_ts - ts).days
            # Exponential decay: 0.95^(days/30) means ~5% decay per month
            weights[i] *= recency_decay ** (days_old / 30)
    
    # 2. Trade feedback weighting
    if len(feedback_df) > 0 and "timestamp" in feedback_df.columns:
        for _, trade in feedback_df.iterrows():
            trade_ts = trade["timestamp"]
            
            # Find samples within 2 hours of trade
            if hasattr(features_df.index, 'to_pydatetime'):
                for i, ts in enumerate(timestamps):
                    time_diff = abs((ts - trade_ts).total_seconds())
                    if time_diff < 7200:  # 2 hours
                        # Boost winning trades, reduce losing trades
                        if trade.get("outcome") == "TP":
                            weights[i] *= 1.2  # 20% boost for wins
                        elif trade.get("outcome") == "SL":
                            weights[i] *= 0.9  # 10% reduction for losses
    
    # 3. Volatility regime weighting (if vol_60 exists)
    if "vol_60" in features_df.columns:
        vol = features_df["vol_60"].values
        vol_median = np.nanmedian(vol)
        
        for i, v in enumerate(vol):
            if not np.isnan(v) and vol_median > 0:
                # Upweight high volatility periods (more learning signal)
                vol_ratio = v / vol_median
                if vol_ratio > 1.5:
                    weights[i] *= 1.1
    
    # Normalize weights to mean=1
    weights = weights / np.mean(weights)
    
    logger.info(f"Sample weights: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
    
    return weights


# =============================================================================
# MODEL TRAINING
# =============================================================================

def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build feature matrix from DataFrame.
    
    Returns:
        X: Feature array
        y: Target array
        feature_cols: List of feature column names
    """
    # Define label columns to exclude
    label_cols = ["y_ret_5", "y_dir_5", "y_ret_15", "y_dir_15", 
                  "y_ret_60", "y_dir_60", "y_tb_60"]
    
    # Meta columns to exclude
    meta_cols = ["timestamp", "symbol", "open", "high", "low", "close", 
                 "volume", "bid_price", "ask_price"]
    
    exclude = set(label_cols + meta_cols)
    
    # Get numeric feature columns
    feature_cols = [
        c for c in df.columns 
        if c not in exclude 
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]
    
    # Filter to rows with valid target
    mask = df[TARGET].notna() & (df[TARGET] != 0)  # Drop y=0 class
    df_valid = df[mask].copy()
    
    X = df_valid[feature_cols].values
    y = df_valid[TARGET].values
    
    # Map -1 to 0 for binary classification
    y = (y == 1).astype(int)
    
    logger.info(f"Feature matrix: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Class distribution: {np.bincount(y)}")
    
    return X, y, feature_cols


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    params: Optional[Dict] = None
) -> object:
    """
    Train gradient boosting model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sample_weights: Optional sample weights
        params: Optional hyperparameters
        
    Returns:
        Trained model
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    
    # Default hyperparameters (from hyperparameter search)
    default_params = {
        "max_depth": 4,
        "learning_rate": 0.03,
        "max_iter": 400,
        "min_samples_leaf": 200,
        "l2_regularization": 0.1,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "random_state": 42,
    }
    
    if params:
        default_params.update(params)
    
    logger.info(f"Training with params: {default_params}")
    
    model = HistGradientBoostingClassifier(**default_params)
    
    # HistGradientBoostingClassifier doesn't support sample_weight directly
    # We'll use it in a custom way via oversampling
    if sample_weights is not None:
        # Weighted sampling
        n_samples = len(X_train)
        indices = np.random.choice(
            n_samples, 
            size=n_samples, 
            replace=True,
            p=sample_weights / sample_weights.sum()
        )
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    name: str = "Eval"
) -> Dict:
    """
    Evaluate model performance.
    
    Returns:
        Dict with metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, 
        f1_score, roc_auc_score, confusion_matrix
    )
    
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
    }
    
    cm = confusion_matrix(y, y_pred)
    
    logger.info(f"\n{name} Metrics:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1:        {metrics['f1']:.4f}")
    logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    logger.info(f"  Confusion Matrix:\n{cm}")
    
    return metrics


# =============================================================================
# MAIN RETRAINING PIPELINE
# =============================================================================

def run_retraining(dry_run: bool = False, force: bool = False) -> bool:
    """
    Run the full retraining pipeline.
    
    Args:
        dry_run: If True, don't save model
        force: If True, deploy even if not better
        
    Returns:
        True if new model deployed
    """
    logger.info("=" * 60)
    logger.info("WEEKLY MODEL RETRAINING")
    logger.info(f"Started: {datetime.now()}")
    logger.info("=" * 60)
    
    # Get API key
    try:
        api_key = get_polygon_api_key()
    except ValueError as e:
        logger.error(str(e))
        return False
    
    # Calculate date range
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)
    
    # Download latest data
    logger.info("\n--- Step 1: Download Latest Data ---")
    bars_path = DATA_DIR / "xauusd_bars_latest.csv"
    quotes_path = DATA_DIR / "xauusd_quotes_latest.csv"
    
    try:
        bars = download_minute_bars(api_key, start_date, end_date, bars_path)
        quotes = download_quotes(api_key, start_date, end_date, quotes_path)
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        return False
    
    if len(bars) < MIN_BARS_REQUIRED:
        logger.error(f"Insufficient data: {len(bars)} bars < {MIN_BARS_REQUIRED} required")
        return False
    
    # Build features
    logger.info("\n--- Step 2: Build Features ---")
    try:
        features_df = build_complete_features(
            bars_df=bars,
            quotes_df=quotes,
            tb_h_max=60,
            tb_tp_mult=1.0,
            tb_sl_mult=1.0
        )
        
        features_path = FEATURES_DIR / "xauusd_features_latest.parquet"
        features_df.to_parquet(features_path)
        logger.info(f"Features saved: {len(features_df)} rows, {len(features_df.columns)} columns")
        
    except Exception as e:
        logger.error(f"Failed to build features: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load trade feedback
    logger.info("\n--- Step 3: Load Trade Feedback ---")
    feedback_df = load_trade_feedback()
    
    # Build feature matrix
    logger.info("\n--- Step 4: Build Feature Matrix ---")
    X, y, feature_cols = build_feature_matrix(features_df)
    
    # Time-based train/val/test split
    n = len(X)
    train_end = int(n * TRAIN_SIZE)
    val_end = int(n * (TRAIN_SIZE + VAL_SIZE))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Compute sample weights (reinforcement learning)
    logger.info("\n--- Step 5: Compute Sample Weights ---")
    features_train = features_df.iloc[:train_end]
    sample_weights = compute_sample_weights(features_train, feedback_df)
    
    # Train new model
    logger.info("\n--- Step 6: Train New Model ---")
    new_model = train_model(X_train, y_train, sample_weights)
    
    # Evaluate new model
    logger.info("\n--- Step 7: Evaluate New Model ---")
    train_metrics = evaluate_model(new_model, X_train, y_train, "Train")
    val_metrics = evaluate_model(new_model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(new_model, X_test, y_test, "Test")
    
    new_auc = val_metrics["roc_auc"]
    
    # Load and evaluate current model
    logger.info("\n--- Step 8: Compare with Current Model ---")
    current_auc = 0.5
    
    if CURRENT_MODEL_PATH.exists():
        try:
            current_artifact = joblib.load(CURRENT_MODEL_PATH)
            current_model = current_artifact["model"]
            current_features = current_artifact["features"]
            
            # Align features
            common_features = [f for f in current_features if f in feature_cols]
            if len(common_features) == len(current_features):
                X_val_aligned = features_df.iloc[train_end:val_end][current_features].values
                y_val_clean = y_val[~np.isnan(X_val_aligned).any(axis=1)]
                X_val_aligned = X_val_aligned[~np.isnan(X_val_aligned).any(axis=1)]
                
                current_metrics = evaluate_model(current_model, X_val_aligned, y_val_clean, "Current Model")
                current_auc = current_metrics["roc_auc"]
            else:
                logger.warning("Feature mismatch - cannot compare with current model")
                
        except Exception as e:
            logger.warning(f"Could not load current model: {e}")
    
    # Decide whether to deploy
    logger.info("\n--- Step 9: Deployment Decision ---")
    
    improvement = new_auc - current_auc
    
    logger.info(f"Current AUC: {current_auc:.4f}")
    logger.info(f"New AUC:     {new_auc:.4f}")
    logger.info(f"Improvement: {improvement:+.4f}")
    
    should_deploy = (
        force or
        (new_auc >= MIN_AUC_THRESHOLD and improvement >= MIN_AUC_IMPROVEMENT)
    )
    
    if dry_run:
        logger.info("DRY RUN - not deploying")
        return False
    
    if should_deploy:
        logger.info("\n--- Step 10: Deploy New Model ---")
        
        # Backup current model
        if CURRENT_MODEL_PATH.exists():
            backup_name = f"y_tb_60_hgb_tuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            backup_path = BACKUP_DIR / backup_name
            shutil.copy(CURRENT_MODEL_PATH, backup_path)
            logger.info(f"Backed up current model to: {backup_path}")
        
        # Save new model
        artifact = {
            "model": new_model,
            "features": feature_cols,
            "best_params": {
                "max_depth": 4,
                "learning_rate": 0.03,
                "max_iter": 400,
                "min_samples_leaf": 200,
                "l2_regularization": 0.1,
            },
            "trained_at": datetime.now().isoformat(),
            "data_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
            },
            "metrics": {
                "train": train_metrics,
                "val": val_metrics,
                "test": test_metrics,
            },
            "n_feedback_samples": len(feedback_df),
        }
        
        joblib.dump(artifact, CURRENT_MODEL_PATH)
        logger.info(f"✅ NEW MODEL DEPLOYED: {CURRENT_MODEL_PATH}")
        
        # Log summary
        summary = {
            "deployed_at": datetime.now().isoformat(),
            "old_auc": current_auc,
            "new_auc": new_auc,
            "improvement": improvement,
            "n_train_samples": len(X_train),
            "n_feedback_samples": len(feedback_df),
        }
        
        summary_path = LOGS_DIR / "retrain_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        return True
    else:
        logger.info("❌ New model not better - keeping current model")
        return False


# =============================================================================
# FEEDBACK RECORDING UTILITY
# =============================================================================

def record_trade_outcome(
    timestamp: str,
    signal: str,
    entry_price: float,
    tp: float,
    sl: float,
    outcome: str,
    exit_price: float,
    r_multiple: float,
    duration_minutes: int
):
    """
    Record a trade outcome for reinforcement learning.
    
    Call this after each trade closes to improve future training.
    """
    feedback_file = FEEDBACK_DIR / "trade_outcomes.jsonl"
    
    record = {
        "timestamp": timestamp,
        "signal": signal,
        "entry_price": entry_price,
        "tp": tp,
        "sl": sl,
        "outcome": outcome,
        "exit_price": exit_price,
        "r_multiple": r_multiple,
        "duration_minutes": duration_minutes,
        "recorded_at": datetime.now().isoformat()
    }
    
    with open(feedback_file, "a") as f:
        f.write(json.dumps(record) + "\n")
    
    logger.info(f"Recorded trade outcome: {signal} -> {outcome} ({r_multiple:+.1f}R)")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Weekly Model Retraining with Reinforcement Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run training but don't deploy model"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Deploy model even if not better than current"
    )
    
    args = parser.parse_args()
    
    success = run_retraining(dry_run=args.dry_run, force=args.force)
    
    if success:
        logger.info("\n✅ RETRAINING COMPLETE - New model deployed!")
        sys.exit(0)
    else:
        logger.info("\n⚠️ RETRAINING COMPLETE - No changes made")
        sys.exit(1)


if __name__ == "__main__":
    main()

