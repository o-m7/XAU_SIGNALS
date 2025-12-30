"""
Training Pipeline for VWAP Mean Reversion Model

The model predicts: "Given price is stretched from VWAP,
will it revert before hitting the stop loss?"
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
    f1_score, roc_auc_score, brier_score_loss
)
from lightgbm import LGBMClassifier

from .config import Model4Config
from .features import build_model4_features, get_model4_feature_columns
from .labels import add_reversion_labels, analyze_label_distribution


def prepare_training_data(
    data_path: str,
    quotes_path: str = None,
    config: Model4Config = None
) -> pd.DataFrame:
    """Load data and build features + labels."""

    config = config or Model4Config()

    print("Loading data...")
    df_1t = pd.read_parquet(data_path)

    # Ensure proper datetime index
    if not isinstance(df_1t.index, pd.DatetimeIndex):
        if 'timestamp' in df_1t.columns:
            df_1t['timestamp'] = pd.to_datetime(df_1t['timestamp'], utc=True)
            df_1t = df_1t.set_index('timestamp')
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")

    df_quotes = None
    if quotes_path and Path(quotes_path).exists():
        print(f"Loading quotes from {quotes_path}...")
        df_quotes = pd.read_parquet(quotes_path)
        if not isinstance(df_quotes.index, pd.DatetimeIndex):
            if 'timestamp' in df_quotes.columns:
                df_quotes['timestamp'] = pd.to_datetime(df_quotes['timestamp'], utc=True)
                df_quotes = df_quotes.set_index('timestamp')

    print("Building features...")
    df = build_model4_features(
        df_1t, df_quotes,
        config.base_timeframe,
        config.vwap_session_hours
    )

    print("Creating reversion labels...")
    df = add_reversion_labels(
        df,
        zscore_threshold=config.entry_zscore_threshold,
        stop_atr_mult=config.stop_atr_mult,
        max_bars=config.max_bars_in_trade
    )

    # Analyze label distribution
    stats = analyze_label_distribution(df)
    print(f"\nLabel Statistics:")
    print(f"  Total setups: {stats['total_setups']:,}")
    print(f"  Base win rate: {stats['win_rate']:.1%}")
    print(f"  Long setups: {stats['long_setups']:,} (WR: {stats['long_win_rate']:.1%})")
    print(f"  Short setups: {stats['short_setups']:,} (WR: {stats['short_win_rate']:.1%})")

    return df


def get_training_set(
    df: pd.DataFrame,
    config: Model4Config = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Filter to tradeable setups only.

    Filter conditions:
    1. Has a setup (y_reversion is not NaN)
    2. Regime is tradeable (not strong trend, not dead/explosive vol)
    3. Session is London or NY
    """
    config = config or Model4Config()
    feature_cols = get_model4_feature_columns()

    mask = (
        df['y_reversion'].notna() &
        (df['regime_tradeable'] == 1) &
        ((df['is_london'] == 1) | (df['is_ny'] == 1))
    )

    df_filtered = df[mask].copy()

    print(f"\nTraining samples: {len(df_filtered):,} (from {len(df):,} total)")
    print(f"Filtered win rate: {df_filtered['y_reversion'].mean():.1%}")

    X = df_filtered[feature_cols]
    y = df_filtered['y_reversion'].astype(int)

    return X, y


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> Tuple[LGBMClassifier, Dict]:
    """Train with walk-forward validation."""

    model = LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=20,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)

    results = {
        'accuracy': [], 'precision': [], 'recall': [],
        'f1': [], 'roc_auc': [], 'brier': []
    }

    print(f"\nWalk-forward CV ({n_splits} splits)...")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        results['accuracy'].append(accuracy_score(y_val, y_pred))
        results['precision'].append(precision_score(y_val, y_pred, zero_division=0))
        results['recall'].append(recall_score(y_val, y_pred, zero_division=0))
        results['f1'].append(f1_score(y_val, y_pred, zero_division=0))
        results['roc_auc'].append(roc_auc_score(y_val, y_proba))
        results['brier'].append(brier_score_loss(y_val, y_proba))

        print(f"  Fold {fold+1}: Acc={results['accuracy'][-1]:.3f}, "
              f"AUC={results['roc_auc'][-1]:.3f}, Brier={results['brier'][-1]:.3f}")

    print("\nCV Summary:")
    for metric, values in results.items():
        print(f"  {metric}: {np.mean(values):.3f} +/- {np.std(values):.3f}")

    # Check for actual predictive power
    mean_auc = np.mean(results['roc_auc'])
    if mean_auc < 0.52:
        print("\n[WARN] AUC < 0.52 suggests no predictive power!")
        print("       Consider: different features, different setup criteria, or rule-based approach")

    # Final fit on all data
    print("\nFinal training on all data...")
    model.fit(X, y)

    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print(importance.to_string(index=False))

    return model, {'cv_results': results, 'feature_importance': importance}


def run_training_pipeline(
    data_path: str,
    quotes_path: str = None,
    output_path: str = "models/model4_lgbm.joblib"
):
    """Main training pipeline."""

    config = Model4Config()

    df = prepare_training_data(data_path, quotes_path, config)
    X, y = get_training_set(df, config)

    # Validate base win rate
    base_wr = y.mean()
    print(f"\nBase win rate (before ML): {base_wr:.1%}")

    if base_wr < 0.45:
        print("[WARN] Base win rate < 45% - setup parameters may need adjustment")
        print("       Try: lower zscore_threshold, tighter stop, or different target")

    model, metrics = train_model(X, y)

    # Save artifact
    artifact = {
        'model': model,
        'config': config,
        'metrics': metrics,
        'feature_columns': get_model4_feature_columns(),
        'base_win_rate': base_wr,
        'trained_at': datetime.now().isoformat(),
        'version': 'v4.1.0-vwap-mr'
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    print(f"\nModel saved to {output_path}")

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Model 4 (VWAP Mean Reversion)")

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
