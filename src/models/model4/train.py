"""
Model 4 Training Pipeline
"""
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from lightgbm import LGBMClassifier

from .config import Model4Config
from .features import build_model4_features, get_model4_feature_columns
from .labels import add_trend_aligned_labels


def load_training_data(
    data_path: str = "data/features/xauusd_1t_2020_2025.parquet",
    quotes_path: str = None,
    config: Model4Config = None
) -> pd.DataFrame:
    """Load and prepare training data."""

    if config is None:
        config = Model4Config()

    print(f"Loading data from {data_path}...")
    df_1t = pd.read_parquet(data_path)

    # Ensure proper datetime index
    if not isinstance(df_1t.index, pd.DatetimeIndex):
        if 'timestamp' in df_1t.columns:
            df_1t['timestamp'] = pd.to_datetime(df_1t['timestamp'], utc=True)
            df_1t = df_1t.set_index('timestamp')
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")

    # Load quotes if available
    df_quotes = None
    if quotes_path and Path(quotes_path).exists():
        print(f"Loading quotes from {quotes_path}...")
        df_quotes = pd.read_parquet(quotes_path)
        if not isinstance(df_quotes.index, pd.DatetimeIndex):
            if 'timestamp' in df_quotes.columns:
                df_quotes['timestamp'] = pd.to_datetime(df_quotes['timestamp'], utc=True)
                df_quotes = df_quotes.set_index('timestamp')

    # Build features
    print("Building features...")
    df = build_model4_features(df_1t, df_quotes, config.base_timeframe)

    # Add labels
    print("Adding labels...")
    df = add_trend_aligned_labels(
        df,
        horizon=config.horizon_bars,
        threshold_atr_mult=config.threshold_atr_mult
    )

    return df


def prepare_training_set(
    df: pd.DataFrame,
    config: Model4Config = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X and y for training.
    Filter to tradeable conditions only.
    """

    if config is None:
        config = Model4Config()

    feature_cols = get_model4_feature_columns()

    # Filter conditions:
    # 1. ADX > threshold (trending market)
    # 2. London or NY session (good liquidity)
    # 3. Direction label is not 0 (clear move happened)

    mask = (
        (df['adx'] > config.min_adx) &
        ((df['is_london'] == 1) | (df['is_ny'] == 1)) &
        (df['y_direction'] != 0)
    )

    df_filtered = df[mask].copy()

    print(f"Training samples: {len(df_filtered):,} / {len(df):,} ({100*len(df_filtered)/len(df):.1f}%)")
    print(f"Label distribution: {df_filtered['y_good_entry'].value_counts(normalize=True).to_dict()}")

    X = df_filtered[feature_cols]
    y = df_filtered['y_good_entry']

    return X, y


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: Model4Config = None,
    n_splits: int = 5
) -> Tuple[LGBMClassifier, Dict[str, Any]]:
    """
    Train LightGBM with walk-forward validation.
    """

    if config is None:
        config = Model4Config()

    # Model hyperparameters
    model = LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # Walk-forward cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)

    cv_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
    }

    print(f"\nWalk-forward CV with {n_splits} splits...")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        cv_results['accuracy'].append(accuracy_score(y_val, y_pred))
        cv_results['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        cv_results['recall'].append(recall_score(y_val, y_pred, zero_division=0))
        cv_results['f1'].append(f1_score(y_val, y_pred, zero_division=0))
        cv_results['roc_auc'].append(roc_auc_score(y_val, y_proba))

        print(f"  Fold {fold+1}: Acc={cv_results['accuracy'][-1]:.3f}, "
              f"AUC={cv_results['roc_auc'][-1]:.3f}, "
              f"F1={cv_results['f1'][-1]:.3f}")

    # Summary
    print("\nCV Summary:")
    for metric, values in cv_results.items():
        print(f"  {metric}: {np.mean(values):.3f} +/- {np.std(values):.3f}")

    # Final training on all data
    print("\nFinal training on all data...")
    model.fit(X, y)

    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 Features:")
    print(importance.head(10).to_string(index=False))

    return model, {
        'cv_results': cv_results,
        'feature_importance': importance,
        'n_samples': len(X),
        'n_features': len(X.columns),
        'label_distribution': y.value_counts(normalize=True).to_dict()
    }


def save_model(
    model: LGBMClassifier,
    metrics: Dict[str, Any],
    config: Model4Config,
    output_path: str = "models/model4_lgbm.joblib"
) -> None:
    """Save model artifact with metadata."""

    artifact = {
        'model': model,
        'config': config,
        'metrics': metrics,
        'feature_columns': get_model4_feature_columns(),
        'trained_at': datetime.now().isoformat(),
        'version': 'v4.0.0'
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    print(f"\nModel saved to {output_path}")


def run_training_pipeline(
    data_path: str = "data/features/xauusd_1t_2020_2025.parquet",
    quotes_path: str = None,
    output_path: str = "models/model4_lgbm.joblib"
) -> None:
    """Main training pipeline."""

    config = Model4Config()

    # Load and prepare data
    df = load_training_data(data_path, quotes_path, config)
    X, y = prepare_training_set(df, config)

    # Train model
    model, metrics = train_model(X, y, config)

    # Save
    save_model(model, metrics, config, output_path)

    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Model 4 (Trend + Entry Timing)")

    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to 1-minute OHLCV parquet file"
    )
    parser.add_argument(
        "--quotes_path",
        type=str,
        default=None,
        help="Path to quotes parquet file (optional)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="models/model4_lgbm.joblib",
        help="Output path for trained model"
    )

    args = parser.parse_args()

    run_training_pipeline(
        data_path=args.data_path,
        quotes_path=args.quotes_path,
        output_path=args.output_path
    )
