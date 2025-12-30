"""
Train Regime Classifier (Model #2 - Stage 1)

This script trains the market regime classifier using:
- 5-minute aggregated data with order flow features
- 5 regime labels (TRENDING, MEAN_REVERTING, BREAKOUT, HIGH_VOL, LOW_LIQUIDITY)
- Multi-class gradient boosting classifier
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model2_regime.tick_aggregator import aggregate_from_minute_bars
from src.model2_regime.order_flow_features import build_order_flow_features
from src.model2_regime.regime_features import build_regime_features
from src.model2_regime.regime_labeling import (
    add_regime_labels,
    get_feature_columns_for_regime_classifier,
    REGIME_NAMES
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models" / "model2_regime"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_aggregate_data(start_date: str = "2020-01-01", end_date: str = "2025-12-22"):
    """Load 1-minute data and aggregate to 5-minute bars."""
    logger.info(f"Loading data from {start_date} to {end_date}...")
    
    # Load existing features (has 1-minute bars and quotes merged)
    # Try 2020-2025 features first, fallback to older files
    features_path = DATA_DIR / "features" / "xauusd_features_2020_2025.parquet"
    
    if not features_path.exists():
        features_path = DATA_DIR / "features" / "xauusd_features_all.parquet"
    
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        logger.info("Run src/features_complete.py first to generate features")
        return None
    
    df = pd.read_parquet(features_path)
    
    # Filter date range
    df = df.loc[start_date:end_date].copy()
    logger.info(f"Loaded {len(df):,} 1-minute bars")
    
    # Aggregate to 5-minute bars
    # We need OHLCV and quotes
    bars_5min = df.resample('5T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    })
    
    # Add quote data if available
    if 'bid_price' in df.columns and 'ask_price' in df.columns:
        quotes_5min = df.resample('5T').agg({
            'bid_price': ['first', 'last', 'mean', 'std'],
            'ask_price': ['first', 'last', 'mean', 'std'],
        })
        quotes_5min.columns = ['_'.join(col).strip() for col in quotes_5min.columns.values]
        
        # Compute spread
        quotes_5min['avg_spread'] = quotes_5min['ask_price_mean'] - quotes_5min['bid_price_mean']
        quotes_5min['avg_mid'] = (quotes_5min['bid_price_mean'] + quotes_5min['ask_price_mean']) / 2
        
        # Approximate depth (use last values as proxy)
        quotes_5min['avg_bid_size'] = np.nan  # Not available from 1-min aggregates
        quotes_5min['avg_ask_size'] = np.nan
        
        # Merge
        df_5min = pd.concat([bars_5min, quotes_5min], axis=1)
    else:
        df_5min = bars_5min
        df_5min['avg_spread'] = np.nan
        df_5min['avg_mid'] = (df_5min['open'] + df_5min['close']) / 2
    
    df_5min = df_5min.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
    
    logger.info(f"Aggregated to {len(df_5min):,} 5-minute bars")
    
    return df_5min


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all regime classification features."""
    logger.info("Building features for regime classifier...")
    
    # Order flow features
    df = build_order_flow_features(df, lookback=20)
    
    # Regime features (volume profile, VWAP, volatility, range)
    df = build_regime_features(df, lookback=60)
    
    # Add regime labels
    df = add_regime_labels(df)
    
    # Drop NaN rows only for critical columns
    initial_rows = len(df)
    critical_cols = ['open', 'high', 'low', 'close', 'volume', 'regime']
    df = df.dropna(subset=critical_cols)
    
    # Fill remaining NaNs with 0 (for numeric columns only)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    dropped = initial_rows - len(df)
    logger.info(f"Dropped {dropped:,} rows with NaN in critical columns, {len(df):,} remaining")
    
    return df


def train_regime_classifier(df: pd.DataFrame):
    """Train the regime classifier."""
    logger.info("Training regime classifier...")
    
    # Get feature columns
    feature_cols = get_feature_columns_for_regime_classifier()
    
    # Filter to only available features
    available_features = [f for f in feature_cols if f in df.columns]
    logger.info(f"Using {len(available_features)} features out of {len(feature_cols)}")
    
    if len(available_features) == 0:
        logger.error("No features available for training!")
        return None
    
    # Prepare data
    X = df[available_features].values
    y = df['regime'].values
    
    logger.info(f"Training set: {len(X):,} samples, {X.shape[1]} features")
    
    # Time-series split
    n_train = int(len(X) * 0.70)
    n_val = int(len(X) * 0.15)
    
    X_train = X[:n_train]
    y_train = y[:n_train]
    
    X_val = X[n_train:n_train + n_val]
    y_val = y[n_train:n_train + n_val]
    
    X_test = X[n_train + n_val:]
    y_test = y[n_train + n_val:]
    
    logger.info(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Train model
    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=100,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=1
    )
    
    logger.info("Training...")
    model.fit(X_train, y_train)
    
    # Evaluate
    logger.info("\n" + "="*80)
    logger.info("TRAIN PERFORMANCE")
    logger.info("="*80)
    y_train_pred = model.predict(X_train)
    logger.info(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    
    # Get actual classes present
    actual_classes = sorted(np.unique(np.concatenate([y_train, y_train_pred])))
    actual_names = [REGIME_NAMES[c] for c in actual_classes]
    logger.info("\n" + classification_report(y_train, y_train_pred, labels=actual_classes, target_names=actual_names))
    
    logger.info("\n" + "="*80)
    logger.info("VALIDATION PERFORMANCE")
    logger.info("="*80)
    y_val_pred = model.predict(X_val)
    logger.info(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    logger.info("\n" + classification_report(y_val, y_val_pred, labels=actual_classes, target_names=actual_names))
    
    logger.info("\n" + "="*80)
    logger.info("TEST PERFORMANCE")
    logger.info("="*80)
    y_test_pred = model.predict(X_test)
    logger.info(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    logger.info("\n" + classification_report(y_test, y_test_pred, labels=actual_classes, target_names=actual_names))
    
    # Confusion matrix
    logger.info("\nConfusion Matrix (Test):")
    cm = confusion_matrix(y_test, y_test_pred)
    logger.info(f"\n{cm}")
    
    # Save model
    artifact = {
        'model': model,
        'features': available_features,
        'regime_names': REGIME_NAMES,
        'trained_at': datetime.now().isoformat(),
        'train_samples': len(X_train),
        'val_accuracy': accuracy_score(y_val, y_val_pred),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
    }
    
    model_path = MODELS_DIR / "regime_classifier.joblib"
    joblib.dump(artifact, model_path)
    logger.info(f"\nSaved model to: {model_path}")
    
    return artifact


def main():
    logger.info("="*80)
    logger.info("MODEL #2 - REGIME CLASSIFIER TRAINING")
    logger.info("="*80)
    
    # Load and aggregate data
    df = load_and_aggregate_data(
        start_date="2020-01-01",
        end_date="2025-12-22"
    )
    
    if df is None:
        logger.error("Failed to load data")
        return
    
    # Build features
    df = build_features(df)
    
    # Save features for later analysis
    features_path = MODELS_DIR / "regime_features_5min.parquet"
    df.to_parquet(features_path)
    logger.info(f"\nSaved features to: {features_path}")
    
    # Train classifier
    artifact = train_regime_classifier(df)
    
    if artifact:
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info(f"Validation Accuracy: {artifact['val_accuracy']:.4f}")
        logger.info(f"Test Accuracy: {artifact['test_accuracy']:.4f}")
        logger.info(f"Features used: {len(artifact['features'])}")
        logger.info("\nNext steps:")
        logger.info("1. Validate on December 2025 data")
        logger.info("2. Build regime-specific signal generators (Phase 2)")
        logger.info("3. Integrate with Model #1")


if __name__ == "__main__":
    main()

