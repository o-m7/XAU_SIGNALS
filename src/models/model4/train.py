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
    """
    Load data and build features + labels.

    Handles two input formats:
    1. Raw 1-minute OHLCV data -> resamples and builds all features
    2. Pre-computed 5T features with OHLCV -> builds VWAP features on top
    """

    config = config or Model4Config()

    print("Loading data...")
    df_raw = pd.read_parquet(data_path)

    # Ensure proper datetime index
    if not isinstance(df_raw.index, pd.DatetimeIndex):
        if 'timestamp' in df_raw.columns:
            df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], utc=True)
            df_raw = df_raw.set_index('timestamp')
        else:
            raise ValueError("DataFrame must have DatetimeIndex or 'timestamp' column")

    # Detect data type based on frequency and existing columns
    has_ohlcv = all(col in df_raw.columns for col in ['open', 'high', 'low', 'close', 'volume'])
    has_vwap = 'vwap' in df_raw.columns and 'vwap_zscore' in df_raw.columns

    # Detect timeframe from index
    if len(df_raw) > 1:
        time_diff = (df_raw.index[1] - df_raw.index[0]).total_seconds()
        is_1t = time_diff <= 60
        is_5t = 240 < time_diff <= 360
    else:
        is_1t = False
        is_5t = True

    print(f"  Data shape: {df_raw.shape}")
    print(f"  Timeframe: {'1T' if is_1t else '5T'}")
    print(f"  Has OHLCV: {has_ohlcv}")
    print(f"  Has VWAP: {has_vwap}")

    df_quotes = None
    if quotes_path and Path(quotes_path).exists():
        print(f"Loading quotes from {quotes_path}...")
        df_quotes = pd.read_parquet(quotes_path)
        if not isinstance(df_quotes.index, pd.DatetimeIndex):
            if 'timestamp' in df_quotes.columns:
                df_quotes['timestamp'] = pd.to_datetime(df_quotes['timestamp'], utc=True)
                df_quotes = df_quotes.set_index('timestamp')

    # Build features based on input type
    if is_1t and has_ohlcv:
        # Raw 1T data - full feature build with resampling
        print(f"Building features from 1T OHLCV data (resampling to {config.base_timeframe})...")
        df = build_model4_features(
            df_raw, df_quotes,
            config.base_timeframe,
            config.vwap_session_hours
        )
        print(f"  After feature build: {len(df):,} rows")
    elif has_ohlcv and not has_vwap:
        # 5T OHLCV data - build VWAP features directly (no resampling)
        print("Building VWAP features from 5T OHLCV data...")
        df = build_model4_features_from_5t(
            df_raw, df_quotes,
            config.vwap_session_hours
        )
    elif has_vwap:
        # Already has VWAP - just ensure regime and session features exist
        print("Using pre-computed VWAP features...")
        df = ensure_model4_features(df_raw, config)
    else:
        raise ValueError(
            "Input data must have OHLCV columns (open, high, low, close, volume). "
            f"Found columns: {list(df_raw.columns)[:10]}..."
        )

    # Diagnostic: show z-score distribution before labeling
    if 'vwap_zscore' in df.columns:
        zscore = df['vwap_zscore'].dropna()
        print(f"\nVWAP Z-Score Distribution:")
        print(f"  Count: {len(zscore):,}")
        print(f"  Mean: {zscore.mean():.3f}")
        print(f"  Std: {zscore.std():.3f}")
        print(f"  Min/Max: {zscore.min():.3f} / {zscore.max():.3f}")
        print(f"  Percentiles (5/25/50/75/95): "
              f"{zscore.quantile(0.05):.2f} / {zscore.quantile(0.25):.2f} / "
              f"{zscore.quantile(0.50):.2f} / {zscore.quantile(0.75):.2f} / "
              f"{zscore.quantile(0.95):.2f}")

        # Count potential setups at different thresholds
        for thresh in [0.5, 1.0, 1.5, 2.0]:
            n_setups = (zscore.abs() >= thresh).sum()
            pct = n_setups / len(zscore) * 100
            print(f"  Bars with |z| >= {thresh}: {n_setups:,} ({pct:.1f}%)")
    else:
        print("\n[WARN] vwap_zscore column not found!")

    print(f"\nCreating reversion labels (threshold={config.entry_zscore_threshold})...")
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


def build_model4_features_from_5t(
    df: pd.DataFrame,
    df_quotes: pd.DataFrame = None,
    session_hours: int = 8
) -> pd.DataFrame:
    """Build VWAP features from pre-resampled 5T data."""
    from .vwap import calculate_session_vwap, calculate_vwap_zscore
    from .regime import classify_regime, calculate_atr
    from .features import calculate_rsi

    df = df.copy()

    # ATR first (needed for z-score)
    df = calculate_atr(df, period=14)
    df = calculate_atr(df, period=5)

    # VWAP features
    df = calculate_session_vwap(df, session_hours=session_hours)
    df = calculate_vwap_zscore(df)

    # Regime features
    df = classify_regime(df)

    # Session position features
    session_bars = int(session_hours * 60 / 5)  # 5T timeframe

    df['session_high'] = df['high'].rolling(session_bars, min_periods=1).max()
    df['session_low'] = df['low'].rolling(session_bars, min_periods=1).min()
    df['session_range'] = df['session_high'] - df['session_low']

    df['price_vs_session_high'] = (df['session_high'] - df['close']) / df['atr_14'].replace(0, np.nan)
    df['price_vs_session_low'] = (df['close'] - df['session_low']) / df['atr_14'].replace(0, np.nan)
    df['price_in_session_range'] = (df['close'] - df['session_low']) / df['session_range'].replace(0, np.nan)

    # Momentum features
    df['rsi_14'] = calculate_rsi(df['close'], length=14)
    df['rsi_7'] = calculate_rsi(df['close'], length=7)

    # RSI divergence
    df['price_high_5'] = df['high'].rolling(5, min_periods=1).max()
    df['price_low_5'] = df['low'].rolling(5, min_periods=1).min()
    df['rsi_high_5'] = df['rsi_14'].rolling(5, min_periods=1).max()
    df['rsi_low_5'] = df['rsi_14'].rolling(5, min_periods=1).min()

    df['bearish_divergence'] = (
        (df['close'] >= df['price_high_5'] * 0.999) &
        (df['rsi_14'] < df['rsi_high_5'] - 5)
    ).astype(int)

    df['bullish_divergence'] = (
        (df['close'] <= df['price_low_5'] * 1.001) &
        (df['rsi_14'] > df['rsi_low_5'] + 5)
    ).astype(int)

    df['rsi_divergence'] = df['bearish_divergence'] - df['bullish_divergence']

    # Bars at extreme z-score
    df['at_upper_extreme'] = (df['vwap_zscore'] > 1.5).astype(int)
    df['at_lower_extreme'] = (df['vwap_zscore'] < -1.5).astype(int)

    df['bars_at_upper'] = df['at_upper_extreme'].groupby(
        (~df['at_upper_extreme'].astype(bool)).cumsum()
    ).cumsum()
    df['bars_at_lower'] = df['at_lower_extreme'].groupby(
        (~df['at_lower_extreme'].astype(bool)).cumsum()
    ).cumsum()
    df['bars_since_extreme'] = np.maximum(df['bars_at_upper'], df['bars_at_lower'])

    # Spread features (placeholder if no quotes)
    if df_quotes is not None and len(df_quotes) > 0:
        quotes_resampled = df_quotes.resample('5min').agg({
            'ask_price': 'mean',
            'bid_price': 'mean',
        })
        quotes_resampled['spread'] = quotes_resampled['ask_price'] - quotes_resampled['bid_price']
        quotes_resampled['mid'] = (quotes_resampled['ask_price'] + quotes_resampled['bid_price']) / 2
        quotes_resampled['spread_pct'] = quotes_resampled['spread'] / quotes_resampled['mid'].replace(0, np.nan)
        quotes_resampled['spread_zscore'] = (
            (quotes_resampled['spread_pct'] - quotes_resampled['spread_pct'].rolling(60).mean()) /
            quotes_resampled['spread_pct'].rolling(60).std().replace(0, np.nan)
        )
        quote_counts = df_quotes.resample('5min').size()
        quotes_resampled['quote_rate'] = quote_counts
        quotes_resampled['quote_rate_zscore'] = (
            (quotes_resampled['quote_rate'] - quotes_resampled['quote_rate'].rolling(60).mean()) /
            quotes_resampled['quote_rate'].rolling(60).std().replace(0, np.nan)
        )
        df = df.join(quotes_resampled[['spread_pct', 'spread_zscore', 'quote_rate_zscore']], how='left')
    else:
        df['spread_pct'] = 0.0001
        df['spread_zscore'] = 0.0
        df['quote_rate_zscore'] = 0.0

    # Time features
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['minutes_since_london'] = (df['hour'] - 7) * 60 + df.index.minute
    df['minutes_since_london'] = df['minutes_since_london'].clip(lower=0)

    df['is_london'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
    df['is_ny'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
    df['is_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)

    # Cleanup
    df = df.replace([np.inf, -np.inf], np.nan).ffill().dropna()

    return df


def ensure_model4_features(df: pd.DataFrame, config: Model4Config) -> pd.DataFrame:
    """Ensure all required Model 4 features exist, adding missing ones."""
    from .regime import classify_regime, calculate_atr

    df = df.copy()

    # Ensure ATR exists
    if 'atr_14' not in df.columns:
        df = calculate_atr(df, period=14)

    # Ensure regime exists
    if 'regime_tradeable' not in df.columns:
        df = classify_regime(df)

    # Ensure session features
    if 'is_london' not in df.columns:
        df['hour'] = df.index.hour
        df['is_london'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
        df['is_ny'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)

    # Fill any remaining required features with defaults
    required_features = get_model4_feature_columns()
    for feat in required_features:
        if feat not in df.columns:
            print(f"  [WARN] Missing feature '{feat}', filling with 0")
            df[feat] = 0.0

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

    # Show filter breakdown
    print(f"\n=== FILTER BREAKDOWN ===")
    print(f"Total rows: {len(df):,}")

    has_setup = df['y_reversion'].notna()
    print(f"Has setup (y_reversion not NaN): {has_setup.sum():,} ({has_setup.mean()*100:.1f}%)")

    if 'regime_tradeable' in df.columns:
        regime_ok = df['regime_tradeable'] == 1
        print(f"Regime tradeable: {regime_ok.sum():,} ({regime_ok.mean()*100:.1f}%)")
    else:
        regime_ok = pd.Series(True, index=df.index)
        print("Regime tradeable: N/A (column missing, using all)")

    if 'is_london' in df.columns and 'is_ny' in df.columns:
        session_ok = (df['is_london'] == 1) | (df['is_ny'] == 1)
        print(f"London/NY session: {session_ok.sum():,} ({session_ok.mean()*100:.1f}%)")
    else:
        session_ok = pd.Series(True, index=df.index)
        print("London/NY session: N/A (columns missing, using all)")

    mask = has_setup & regime_ok & session_ok
    df_filtered = df[mask].copy()

    print(f"\nAfter all filters: {len(df_filtered):,} samples")

    if len(df_filtered) == 0:
        print("\n[ERROR] No training samples! Check:")
        print("  - Are setups being created? (y_reversion)")
        print("  - Is regime too strict? (regime_tradeable)")
        print("  - Are session hours correct? (is_london/is_ny)")
        # Return empty but valid dataframes to avoid crash
        X = pd.DataFrame(columns=feature_cols)
        y = pd.Series(dtype=float)
        return X, y

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

    # Check for empty training set
    if len(X) == 0:
        print("\n" + "="*60)
        print("TRAINING ABORTED: No training samples found!")
        print("="*60)
        print("\nPossible fixes:")
        print("  1. Lower entry_zscore_threshold in config.py (currently 1.0)")
        print("  2. Check regime_tradeable filter (ADX threshold)")
        print("  3. Verify data has London/NY session hours")
        return

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
        default="data/features/xauusd_features_2020_2025.parquet",
        help="Path to OHLCV parquet file (1T or 5T timeframe)"
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
