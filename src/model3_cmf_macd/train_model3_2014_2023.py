"""
Train Model #3 (CMF/MACD) with 2014-2023 training and 2024-2025 validation.

Same strategy as current Model 3 but:
- Train on 2014-2023 data
- Validate on 2024-2025 data
- Implements regime detection for short signals (only short in bearish regimes)
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

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model3_cmf_macd.features import build_cmf_macd_features, get_feature_columns_for_model3
from src.model3_cmf_macd.labeling import add_triple_barrier_labels
from src.regime_detection import add_regime_features, filter_short_signals_by_regime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models" / "model3_cmf_macd_2014_2023"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# New date ranges
TRAIN_START = "2014-01-01"
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2025-12-31"

FEATURES_FILE = FEATURES_DIR / "xauusd_features_2020_2025.parquet"


def load_data() -> pd.DataFrame:
    """Load 1-minute bar data."""
    logger.info(f"Loading data from features file...")
    
    if not FEATURES_FILE.exists():
        logger.error(f"Features file not found: {FEATURES_FILE}")
        logger.info("Run src/features_complete.py first to generate features")
        return None
    
    df = pd.read_parquet(FEATURES_FILE)
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
    
    df = df.sort_index()
    logger.info(f"Loaded {len(df):,} 1-minute bars")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Ensure we have OHLCV
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return None
    
    return df


def split_by_dates(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train (2014-2023) and validation (2024-2025)."""
    train_df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)].copy()
    val_df = df[(df.index >= VAL_START) & (df.index <= VAL_END)].copy()
    
    logger.info(f"\nData Split:")
    logger.info(f"  Train: {len(train_df):,} rows ({TRAIN_START} to {TRAIN_END})")
    logger.info(f"    Date range: {train_df.index.min().date()} to {train_df.index.max().date()}")
    logger.info(f"  Validation: {len(val_df):,} rows ({VAL_START} to {VAL_END})")
    logger.info(f"    Date range: {val_df.index.min().date()} to {val_df.index.max().date()}")
    
    return train_df, val_df


def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute balanced sample weights to handle class imbalance.
    
    Weights are inversely proportional to class frequency.
    """
    from sklearn.utils.class_weight import compute_sample_weight
    
    # Compute class weights (balanced)
    class_weight = 'balanced'
    sample_weights = compute_sample_weight(class_weight, y)
    
    return sample_weights


def prepare_training_data(df: pd.DataFrame):
    """Prepare features and labels for training using 15-minute labels."""
    logger.info("Building CMF/MACD features and labels...")
    
    # Build CMF and MACD features
    df = build_cmf_macd_features(df)
    
    # Add triple-barrier labels for 15-minute horizon (not 60-minute)
    # Using symmetric barriers (1.0/1.0) to avoid directional bias
    from src.features_complete import add_triple_barrier_labels
    df = add_triple_barrier_labels(df, h_max=15, tp_mult=1.0, sl_mult=1.0, horizons=[15])
    
    # Get feature columns
    feature_cols = get_feature_columns_for_model3()
    available_features = [f for f in feature_cols if f in df.columns]
    
    logger.info(f"Using {len(available_features)} features out of {len(feature_cols)}")
    
    # Filter to rows with valid labels (drop 0 and NaN)
    df_clean = df.dropna(subset=available_features + ['y_tb_15'])
    df_clean = df_clean[df_clean['y_tb_15'] != 0]
    
    logger.info(f"After filtering: {len(df_clean):,} samples")
    
    # Map labels: -1 -> 0, +1 -> 1 (for binary classification)
    y = df_clean['y_tb_15'].values
    y_binary = np.where(y == 1, 1, 0)  # +1 -> 1, -1 -> 0
    
    X = df_clean[available_features].values
    
    # Check label distribution
    n_long = np.sum(y == 1)
    n_short = np.sum(y == -1)
    total = len(y)
    
    logger.info(f"Final dataset: {len(X):,} samples, {X.shape[1]} features")
    logger.info(f"Label distribution: +1={n_long:,} ({100*n_long/total:.1f}%), -1={n_short:,} ({100*n_short/total:.1f}%)")
    
    return X, y_binary, available_features, df_clean


def train_model(X_train, y_train, X_val, y_val, sample_weights=None):
    """Train the Model #3 classifier with balanced sample weights."""
    logger.info("Training Model #3 with balanced sample weights...")
    
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
    
    logger.info("Fitting model with balanced weights...")
    if sample_weights is not None:
        logger.info(f"  Using sample weights (min={sample_weights.min():.3f}, max={sample_weights.max():.3f}, mean={sample_weights.mean():.3f})")
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    return model


def apply_regime_filtering(df: pd.DataFrame, signals: pd.Series) -> pd.Series:
    """Apply regime detection to filter short signals."""
    logger.info("\nApplying regime detection for short signals...")
    
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
    
    logger.info(f"  Original short signals: {original_shorts:,}")
    logger.info(f"  After regime filtering: {filtered_shorts:,}")
    logger.info(f"  Removed (not in bearish regime): {removed:,} ({100*removed/max(original_shorts,1):.1f}%)")
    
    return filtered_signals


def main():
    logger.info("="*80)
    logger.info("MODEL #3 - CMF AND MACD CLASSIFIER TRAINING (2014-2023)")
    logger.info("="*80)
    logger.info(f"\nTraining Period: {TRAIN_START} to {TRAIN_END}")
    logger.info(f"Validation Period: {VAL_START} to {VAL_END}")
    logger.info(f"Labels: 15-minute triple-barrier (y_tb_15)")
    logger.info(f"Sample Weights: Balanced (to reduce class imbalance)")
    logger.info(f"Regime Detection: Enabled (only short in bearish regimes)")
    
    # Load data
    df = load_data()
    if df is None:
        logger.error("Failed to load data")
        return
    
    # Split by dates
    train_df, val_df = split_by_dates(df)
    
    if len(train_df) == 0:
        logger.error(f"No training data found for {TRAIN_START} to {TRAIN_END}")
        return
    if len(val_df) == 0:
        logger.error(f"No validation data found for {VAL_START} to {VAL_END}")
        return
    
    # Prepare training data
    X_train, y_train, features, train_df_clean = prepare_training_data(train_df)
    
    # Prepare validation data
    X_val, y_val, _, val_df_clean = prepare_training_data(val_df)
    
    logger.info(f"\nSplit: Train={len(X_train):,} | Val={len(X_val):,}")
    
    # Compute balanced sample weights for training
    logger.info("\nComputing balanced sample weights...")
    sample_weights = compute_sample_weights(y_train)
    
    # Train model with balanced weights
    model = train_model(X_train, y_train, X_val, y_val, sample_weights=sample_weights)
    
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
    logger.info("VALIDATION PERFORMANCE (2024-2025)")
    logger.info("="*80)
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    logger.info(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    logger.info(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
    logger.info(f"Recall: {recall_score(y_val, y_val_pred):.4f}")
    logger.info(f"F1: {f1_score(y_val, y_val_pred):.4f}")
    logger.info(f"ROC-AUC: {roc_auc_score(y_val, y_val_proba):.4f}")
    
    logger.info("\nConfusion Matrix (Validation):")
    cm = confusion_matrix(y_val, y_val_pred)
    logger.info(f"\n{cm}")
    
    # Apply regime filtering to validation predictions
    logger.info("\n" + "="*80)
    logger.info("APPLYING REGIME DETECTION TO VALIDATION SET")
    logger.info("="*80)
    
    # Convert predictions to signals
    val_signals = pd.Series(0, index=val_df_clean.index)
    val_signals[y_val_proba >= 0.6] = 1   # Long
    val_signals[y_val_proba <= 0.3] = -1  # Short
    
    # Apply regime filtering
    val_signals_filtered = apply_regime_filtering(val_df_clean, val_signals)
    
    # Save model
    artifact = {
        'model': model,
        'features': features,
        'trained_at': datetime.now().isoformat(),
        'train_samples': len(X_train),
        'train_period': f"{TRAIN_START} to {TRAIN_END}",
        'val_period': f"{VAL_START} to {VAL_END}",
        'regime_detection': True,
        'metrics': {
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'train_auc': roc_auc_score(y_train, y_train_proba),
            'val_accuracy': accuracy_score(y_val, y_val_pred),
            'val_auc': roc_auc_score(y_val, y_val_proba),
        }
    }
    
    model_path = MODELS_DIR / "model3_cmf_macd_2014_2023_15min_balanced.joblib"
    joblib.dump(artifact, model_path)
    logger.info(f"\nSaved model to: {model_path}")
    
    # Also save with the original name for compatibility
    model_path_orig = MODELS_DIR / "model3_cmf_macd_2014_2023.joblib"
    joblib.dump(artifact, model_path_orig)
    logger.info(f"Also saved as: {model_path_orig}")
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    logger.info(f"Validation AUC: {roc_auc_score(y_val, y_val_proba):.4f}")
    logger.info(f"Features used: {len(features)}")


if __name__ == "__main__":
    main()

