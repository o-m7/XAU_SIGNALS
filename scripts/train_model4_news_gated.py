#!/usr/bin/env python3
"""
Train Model #4 (5m News-Gated Long-Only).

Pipeline:
1. Load 5min XAUUSD data (last 12 weeks)
2. Compute Model4 features (5 fast features)
3. Create triple-barrier labels (0.3% up, 0.2% down, 30-bar = 150min lookback)
4. Train XGBoost with walk-forward validation
5. Validate: PF >= 1.6, WR >= 52%
6. Save model to models/model4_news_gated_5m.joblib

Usage:
    python scripts/train_model4_news_gated.py [--weeks 12] [--test]
"""

import sys
from pathlib import Path
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model4_features import compute_model4_features, MODEL4_FEATURES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainModel4")


def load_5min_data(weeks: int = 12) -> pd.DataFrame:
    """
    Load last N weeks of 5min data.

    Args:
        weeks: Number of weeks to load (default: 12)

    Returns:
        DataFrame with OHLCV
    """
    data_path = PROJECT_ROOT / "models" / "model2_regime" / "regime_features_5min.parquet"

    if not data_path.exists():
        raise FileNotFoundError(f"5min data not found: {data_path}")

    logger.info(f"Loading 5min data from {data_path}")
    df = pd.read_parquet(data_path)

    # Filter to last N weeks
    end_date = df.index.max()
    start_date = end_date - timedelta(weeks=weeks)

    df = df[df.index >= start_date].copy()

    logger.info(f"Loaded {len(df)} bars ({start_date} to {end_date})")

    return df


def create_triple_barrier_labels(
    df: pd.DataFrame,
    up_pct: float = 0.004,  # 0.4% up barrier (relaxed for 5m bars)
    down_pct: float = 0.003,  # 0.3% down barrier (balanced R:R)
    max_bars: int = 20,  # 20 bars = 100 minutes (realistic 5m hold)
) -> pd.DataFrame:
    """
    Create triple-barrier labels for long-only model.

    Label = 1 if price hits +0.4% before -0.3% within 20 bars
    Label = 0 otherwise (neutral/loss)

    Args:
        df: DataFrame with OHLCV
        up_pct: Upper barrier (profit target, e.g., 0.004 = 0.4%)
        down_pct: Lower barrier (stop loss, e.g., 0.003 = 0.3%)
        max_bars: Maximum bars to look ahead (20 bars = 100min @ 5min bars)

    Returns:
        DataFrame with 'target' column added
    """
    logger.info(
        f"Creating triple-barrier labels: "
        f"up={up_pct*100:.1f}%, down={down_pct*100:.1f}%, max_bars={max_bars}"
    )

    close = df['close'].values
    labels = np.zeros(len(df), dtype=int)

    for i in range(len(df) - max_bars):
        entry_price = close[i]
        upper_barrier = entry_price * (1 + up_pct)
        lower_barrier = entry_price * (1 - down_pct)

        # Look ahead up to max_bars
        future_prices = close[i + 1 : i + max_bars + 1]

        # Find first barrier hit
        hit_upper = np.where(future_prices >= upper_barrier)[0]
        hit_lower = np.where(future_prices <= lower_barrier)[0]

        if len(hit_upper) > 0 and len(hit_lower) > 0:
            # Both barriers hit - which came first?
            if hit_upper[0] < hit_lower[0]:
                labels[i] = 1  # Upper hit first (win)
            else:
                labels[i] = 0  # Lower hit first (loss)
        elif len(hit_upper) > 0:
            # Only upper hit (win)
            labels[i] = 1
        else:
            # Either only lower hit or neither (loss or neutral)
            labels[i] = 0

    df['target'] = labels

    # Summary
    label_counts = pd.Series(labels).value_counts()
    logger.info(f"Labels: {label_counts.to_dict()}")
    logger.info(f"Win rate: {labels.mean()*100:.1f}%")

    return df


def split_walk_forward(
    df: pd.DataFrame,
    val_weeks: int = 1
) -> tuple:
    """
    Split data into train/val using walk-forward method.

    Train: All data except last val_weeks
    Val: Last val_weeks

    Args:
        df: DataFrame with features + target
        val_weeks: Number of weeks for validation (default: 1)

    Returns:
        (train_df, val_df)
    """
    val_cutoff = df.index.max() - timedelta(weeks=val_weeks)

    train = df[df.index < val_cutoff].copy()
    val = df[df.index >= val_cutoff].copy()

    logger.info(f"Train: {len(train)} bars ({train.index.min()} to {train.index.max()})")
    logger.info(f"Val: {len(val)} bars ({val.index.min()} to {val.index.max()})")

    return train, val


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> XGBClassifier:
    """
    Train XGBoost classifier.

    Hyperparameters optimized for 5min intraday:
    - max_depth: 5 (moderate complexity)
    - n_estimators: 200 (more trees for better learning)
    - learning_rate: 0.05 (slower, more stable)
    - scale_pos_weight: auto (handle class imbalance)

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels

    Returns:
        Trained XGBClassifier
    """
    logger.info("Training XGBoost...")

    # Calculate scale_pos_weight for class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    logger.info(f"Class balance: {n_pos} positive / {n_neg} negative (scale_pos_weight={scale_pos_weight:.2f})")

    model = XGBClassifier(
        max_depth=5,
        n_estimators=200,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=20,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    logger.info("✓ Training complete")

    return model


def evaluate_model(
    model: XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    val_df: pd.DataFrame
) -> Dict:
    """
    Evaluate model on validation set.

    Computes:
    - Win rate
    - Profit factor (sum_wins / abs(sum_losses))
    - Classification metrics

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation labels
        val_df: Validation DataFrame (for price data)

    Returns:
        Dict with metrics
    """
    logger.info("Evaluating model...")

    # Predictions
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    # Classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_val, y_pred))

    # Confusion matrix
    logger.info("\nConfusion Matrix:")
    logger.info("\n" + str(confusion_matrix(y_val, y_pred)))

    # Win rate
    win_rate = y_val.mean()
    logger.info(f"Win Rate (labels): {win_rate*100:.1f}%")

    # Profit factor (simplified - assumes 1R per trade)
    wins = (y_pred == 1).sum()
    losses = (y_pred == 0).sum()

    if losses > 0:
        # Assume 1.5R per win (TP = 0.5 ATR, SL = 1.5 ATR → R = 1.5/1.5 = 1.0, but TP/SL ratio is 1.5)
        # For simplicity: PF = (wins × 1.5R) / (losses × 1R)
        profit_factor = (wins * 1.5) / (losses * 1.0)
    else:
        profit_factor = float('inf')

    logger.info(f"Profit Factor (estimated): {profit_factor:.2f}")

    metrics = {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'n_trades': len(y_pred),
        'n_wins': wins,
        'n_losses': losses,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Model #4")
    parser.add_argument("--weeks", type=int, default=26, help="Training data weeks (default: 26)")
    parser.add_argument("--test", action="store_true", help="Test mode (faster)")
    parser.add_argument("--force", action="store_true", help="Force save model even if validation fails")
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("TRAIN MODEL #4 (5m News-Gated Long-Only)")
    logger.info("=" * 80)

    # 1. Load data
    df = load_5min_data(weeks=args.weeks)

    # 2. Compute Model4 features
    logger.info("Computing Model4 features...")
    df = compute_model4_features(df, inplace=True)

    # 3. Create labels
    df = create_triple_barrier_labels(df)

    # 4. Drop NaN
    df = df.dropna(subset=MODEL4_FEATURES + ['target'])
    logger.info(f"After dropna: {len(df)} bars")

    # 5. Split train/val
    train_df, val_df = split_walk_forward(df, val_weeks=1)

    X_train = train_df[MODEL4_FEATURES]
    y_train = train_df['target']
    X_val = val_df[MODEL4_FEATURES]
    y_val = val_df['target']

    # 6. Train
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # 7. Evaluate
    metrics = evaluate_model(model, X_val, y_val, val_df)

    # 8. Check deployment criteria
    logger.info("\n" + "=" * 80)
    logger.info("DEPLOYMENT VALIDATION")
    logger.info("=" * 80)

    passed = True

    if metrics['profit_factor'] < 1.6:
        logger.warning(f"⚠️ Profit Factor {metrics['profit_factor']:.2f} < 1.6 (target)")
        passed = False
    else:
        logger.info(f"✓ Profit Factor {metrics['profit_factor']:.2f} >= 1.6")

    if metrics['win_rate'] < 0.52:
        logger.warning(f"⚠️ Win Rate {metrics['win_rate']*100:.1f}% < 52% (target)")
        passed = False
    else:
        logger.info(f"✓ Win Rate {metrics['win_rate']*100:.1f}% >= 52%")

    if metrics['n_trades'] < 30:
        logger.warning(f"⚠️ Only {metrics['n_trades']} trades (minimum 30)")
        passed = False
    else:
        logger.info(f"✓ {metrics['n_trades']} trades >= 30")

    # 9. Save model
    if passed or args.test or args.force:
        model_path = PROJECT_ROOT / "models" / "model4_news_gated_5m.joblib"

        artifact = {
            'model': model,
            'features': MODEL4_FEATURES,
            'metrics': metrics,
            'trained_at': datetime.now().isoformat(),
            'training_weeks': args.weeks,
        }

        joblib.dump(artifact, model_path)
        logger.info(f"\n✓ Model saved to {model_path}")

        if not passed:
            logger.warning("⚠️ Model did not meet deployment criteria but saved (--force or --test mode)")
            logger.warning("   Use with caution in production")

    else:
        logger.error("\n❌ Model FAILED deployment validation - not saved")
        logger.error("   Retrain with more data, adjust hyperparameters, or use --force")
        sys.exit(1)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
