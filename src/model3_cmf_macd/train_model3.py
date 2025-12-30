"""
Train Model #3: CMF and MACD-Based Classifier

This script trains a classifier using:
- Chaikin Money Flow (CMF)
- MACD (Moving Average Convergence Divergence)
- Additional technical indicators
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model3_cmf_macd.features import build_cmf_macd_features, get_feature_columns_for_model3
from src.model3_cmf_macd.labeling import add_triple_barrier_labels

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models" / "model3_cmf_macd"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def load_data(start_date: str = "2020-01-01", end_date: str = "2025-12-22") -> Optional[pd.DataFrame]:
    """Load 1-minute bar data."""
    logger.info(f"Loading data from {start_date} to {end_date}...")
    
    # Try 2020-2025 features first
    features_path = FEATURES_DIR / "xauusd_features_2020_2025.parquet"
    
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        logger.info("Run src/features_complete.py first to generate features")
        return None
    
    df = pd.read_parquet(features_path)
    
    # Filter date range
    df = df.loc[start_date:end_date].copy()
    logger.info(f"Loaded {len(df):,} 1-minute bars")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Ensure we have OHLCV
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return None
    
    return df


def prepare_training_data(df: pd.DataFrame):
    """Prepare features and labels for training."""
    logger.info("Building features and labels...")
    
    # Build CMF and MACD features
    df = build_cmf_macd_features(df)
    
    # Add triple-barrier labels
    df = add_triple_barrier_labels(df, h_max=60, tp_mult=1.5, sl_mult=1.0)
    
    # Get feature columns
    feature_cols = get_feature_columns_for_model3()
    available_features = [f for f in feature_cols if f in df.columns]
    
    logger.info(f"Using {len(available_features)} features out of {len(feature_cols)}")
    
    # Filter to rows with valid labels (drop 0 and NaN)
    df_clean = df.dropna(subset=available_features + ['y_tb_60'])
    df_clean = df_clean[df_clean['y_tb_60'] != 0]
    
    logger.info(f"After filtering: {len(df_clean):,} samples")
    
    # Map labels: -1 -> 0, +1 -> 1 (for binary classification)
    y = df_clean['y_tb_60'].values
    y_binary = np.where(y == 1, 1, 0)  # +1 -> 1, -1 -> 0
    
    X = df_clean[available_features].values
    
    logger.info(f"Final dataset: {len(X):,} samples, {X.shape[1]} features")
    logger.info(f"Label distribution: +1={np.sum(y == 1):,}, -1={np.sum(y == -1):,}")
    
    return X, y_binary, available_features, df_clean


def train_model(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train the Model #3 classifier."""
    logger.info("Training Model #3...")
    
    model = HistGradientBoostingClassifier(
        max_depth=5,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=200,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=RANDOM_SEED,
        verbose=1
    )
    
    logger.info("Fitting model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    logger.info("\n" + "="*80)
    logger.info("TRAIN PERFORMANCE")
    logger.info("="*80)
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    logger.info(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    logger.info(f"Precision: {precision_score(y_train, y_train_pred):.4f}")
    logger.info(f"Recall: {recall_score(y_train, y_train_pred):.4f}")
    logger.info(f"F1: {f1_score(y_train, y_train_pred):.4f}")
    logger.info(f"ROC-AUC: {roc_auc_score(y_train, y_train_proba):.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("VALIDATION PERFORMANCE")
    logger.info("="*80)
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    logger.info(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    logger.info(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
    logger.info(f"Recall: {recall_score(y_val, y_val_pred):.4f}")
    logger.info(f"F1: {f1_score(y_val, y_val_pred):.4f}")
    logger.info(f"ROC-AUC: {roc_auc_score(y_val, y_val_proba):.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("TEST PERFORMANCE")
    logger.info("="*80)
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    logger.info(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    logger.info(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
    logger.info(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
    logger.info(f"F1: {f1_score(y_test, y_test_pred):.4f}")
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}")
    
    logger.info("\nConfusion Matrix (Test):")
    cm = confusion_matrix(y_test, y_test_pred)
    logger.info(f"\n{cm}")
    
    return model, {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_auc': roc_auc_score(y_train, y_train_proba),
        'val_accuracy': accuracy_score(y_val, y_val_pred),
        'val_auc': roc_auc_score(y_val, y_val_proba),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_auc': roc_auc_score(y_test, y_test_proba),
    }


def main():
    logger.info("="*80)
    logger.info("MODEL #3 - CMF AND MACD CLASSIFIER TRAINING")
    logger.info("="*80)
    
    # Load data
    df = load_data(start_date="2020-01-01", end_date="2025-12-22")
    
    if df is None:
        logger.error("Failed to load data")
        return
    
    # Prepare training data
    X, y, features, df_clean = prepare_training_data(df)
    
    # Time-series split
    n = len(X)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    logger.info(f"\nSplit: Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")
    
    # Train model
    model, metrics = train_model(X_train, y_train, X_val, y_val, X_test, y_test)
    
    # Save model
    artifact = {
        'model': model,
        'features': features,
        'trained_at': datetime.now().isoformat(),
        'train_samples': len(X_train),
        'metrics': metrics,
    }
    
    model_path = MODELS_DIR / "model3_cmf_macd.joblib"
    joblib.dump(artifact, model_path)
    logger.info(f"\nSaved model to: {model_path}")
    
    # Save features
    features_path = MODELS_DIR / "model3_features.parquet"
    df_clean.to_parquet(features_path)
    logger.info(f"Saved features to: {features_path}")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"Test AUC: {metrics['test_auc']:.4f}")
    logger.info(f"Features used: {len(features)}")


if __name__ == "__main__":
    main()

