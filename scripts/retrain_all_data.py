#!/usr/bin/env python3
"""
Retrain model on ALL available data up to Dec 22, 2025.
Then backtest on 2025 data.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features_complete import build_complete_features

# Paths
DATA_DIR = Path("/Users/omar/Desktop/ML/Data")
PROJECT_DIR = Path(__file__).parent.parent
MODELS_DIR = PROJECT_DIR / "models"
FEATURES_DIR = PROJECT_DIR / "data" / "features"

# Training cutoff
TRAIN_CUTOFF = "2025-12-01"  # Train on everything before Dec 1
TEST_START = "2025-12-01"    # Test on Dec 1-22

TARGET = "y_tb_60"


def load_all_bars():
    """Load all minute bar data."""
    print("\n1. Loading all minute bars...")
    
    all_bars = []
    for year in [2024, 2025]:
        path = DATA_DIR / "ohlcv_minute" / f"XAUUSD_minute_{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            df = df.sort_index()
            all_bars.append(df)
            print(f"   {year}: {len(df):,} bars")
    
    bars = pd.concat(all_bars)
    bars = bars.sort_index()
    bars = bars[~bars.index.duplicated(keep='first')]
    
    print(f"   Total: {len(bars):,} bars")
    print(f"   Range: {bars.index.min()} to {bars.index.max()}")
    
    return bars


def load_all_quotes():
    """Load all quote data."""
    print("\n2. Loading all quotes...")
    
    all_quotes = []
    for year in [2024, 2025]:
        path = DATA_DIR / "quotes" / f"XAUUSD_quotes_{year}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            df = df.sort_index()
            all_quotes.append(df)
            print(f"   {year}: {len(df):,} quotes")
    
    if all_quotes:
        quotes = pd.concat(all_quotes)
        quotes = quotes.sort_index()
        quotes = quotes[~quotes.index.duplicated(keep='first')]
        print(f"   Total: {len(quotes):,} quotes")
        return quotes
    
    return None


def build_features(bars, quotes):
    """Build complete feature set."""
    print("\n3. Building features...")
    
    # Rename columns if needed
    col_map = {
        'bid_price': 'bid_price',
        'ask_price': 'ask_price',
        'bid': 'bid_price',
        'ask': 'ask_price',
    }
    
    if quotes is not None:
        for old, new in col_map.items():
            if old in quotes.columns and new not in quotes.columns:
                quotes = quotes.rename(columns={old: new})
    
    # Build features
    features_df = build_complete_features(
        bars=bars,
        quotes=quotes,
        tb_h_max=60,
        tb_tp_mult=1.0,
        tb_sl_mult=1.0
    )
    
    print(f"   Features built: {len(features_df):,} rows, {len(features_df.columns)} columns")
    
    return features_df


def prepare_data(df):
    """Prepare X, y for training."""
    print("\n4. Preparing training data...")
    
    # Label and meta columns to exclude
    label_cols = ["y_ret_5", "y_dir_5", "y_ret_15", "y_dir_15", 
                  "y_ret_60", "y_dir_60", "y_tb_60"]
    meta_cols = ["timestamp", "symbol", "open", "high", "low", "close", 
                 "volume", "bid_price", "ask_price"]
    # Exclude features not available in real-time
    live_unavailable = ["vwap", "trades"]
    
    exclude = set(label_cols + meta_cols + live_unavailable)
    
    # Get numeric feature columns
    feature_cols = [
        c for c in df.columns 
        if c not in exclude 
        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
    ]
    
    # Filter valid rows
    mask = df[TARGET].notna() & (df[TARGET] != 0)
    df_valid = df[mask].copy()
    
    print(f"   Valid rows (y_tb_60 != 0): {len(df_valid):,}")
    print(f"   Feature columns: {len(feature_cols)}")
    
    return df_valid, feature_cols


def train_model(X_train, y_train):
    """Train gradient boosting model."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    
    print("\n5. Training model...")
    
    model = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.03,
        max_iter=400,
        min_samples_leaf=200,
        l2_regularization=0.1,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    
    model.fit(X_train, y_train)
    print("   Training complete!")
    
    return model


def evaluate(model, X, y, name):
    """Evaluate model performance."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba),
    }
    
    print(f"\n   {name} Metrics:")
    print(f"      Accuracy:  {metrics['accuracy']:.4f}")
    print(f"      Precision: {metrics['precision']:.4f}")
    print(f"      Recall:    {metrics['recall']:.4f}")
    print(f"      F1:        {metrics['f1']:.4f}")
    print(f"      ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    return metrics


def run_backtest(y, proba_up, tl, ts):
    """Run backtest with given thresholds."""
    signal = np.zeros_like(proba_up, dtype=int)
    signal[proba_up >= tl] = 1   # LONG
    signal[proba_up <= ts] = -1  # SHORT
    
    trade_mask = signal != 0
    n_trades = int(trade_mask.sum())
    
    if n_trades == 0:
        return {"n_trades": 0, "win_rate": 0, "cum_r": 0, "sharpe": 0}
    
    y_trades = y[trade_mask]
    signal_trades = signal[trade_mask]
    trade_ret = y_trades * signal_trades
    
    win_rate = float((trade_ret > 0).mean())
    avg_r = float(trade_ret.mean())
    cum_r = float(trade_ret.sum())
    sharpe = float(avg_r / (trade_ret.std() + 1e-8) * np.sqrt(252))
    
    n_long = int((signal_trades == 1).sum())
    n_short = int((signal_trades == -1).sum())
    
    return {
        "n_trades": n_trades,
        "n_long": n_long,
        "n_short": n_short,
        "win_rate": win_rate,
        "cum_r": cum_r,
        "sharpe": sharpe
    }


def main():
    print("=" * 70)
    print("RETRAIN MODEL ON ALL DATA (2024 + 2025)")
    print("=" * 70)
    
    # Load data
    bars = load_all_bars()
    quotes = load_all_quotes()
    
    # Build features
    features_df = build_features(bars, quotes)
    
    # Save features
    features_path = FEATURES_DIR / "xauusd_features_all.parquet"
    features_df.to_parquet(features_path)
    print(f"\n   Saved features to: {features_path}")
    
    # Prepare data
    df_valid, feature_cols = prepare_data(features_df)
    
    # Time-based split
    print("\n6. Splitting data...")
    train_mask = df_valid.index < TRAIN_CUTOFF
    test_mask = df_valid.index >= TEST_START
    
    df_train = df_valid[train_mask]
    df_test = df_valid[test_mask]
    
    print(f"   Train: {len(df_train):,} samples ({df_train.index.min()} to {df_train.index.max()})")
    print(f"   Test:  {len(df_test):,} samples ({df_test.index.min()} to {df_test.index.max()})")
    
    X_train = df_train[feature_cols].values
    y_train = df_train[TARGET].values
    y_train = (y_train == 1).astype(int)  # Binary: 1=up, 0=down
    
    X_test = df_test[feature_cols].values
    y_test = df_test[TARGET].values
    y_test = (y_test == 1).astype(int)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    train_metrics = evaluate(model, X_train, y_train, "Train")
    test_metrics = evaluate(model, X_test, y_test, "Test (Dec 2025)")
    
    # Backtest on test set
    print("\n7. Backtesting on Dec 2025...")
    proba_test = model.predict_proba(X_test)[:, 1]
    y_test_signed = np.where(df_test[TARGET].values == 1, 1, -1)
    
    # Test different thresholds
    thresholds = [
        (0.70, 0.30),
        (0.70, 0.25),
        (0.75, 0.25),
        (0.65, 0.35),
    ]
    
    print(f"\n   {'Long':<6} {'Short':<6} {'Trades':<8} {'Win%':<8} {'CumR':<10} {'Sharpe':<8}")
    print("   " + "-" * 50)
    
    for tl, ts in thresholds:
        result = run_backtest(y_test_signed, proba_test, tl, ts)
        print(f"   {tl:<6.2f} {ts:<6.2f} {result['n_trades']:<8} "
              f"{result['win_rate']*100:<7.1f}% {result['cum_r']:<+9.1f} {result['sharpe']:<8.2f}")
    
    # Save model
    print("\n8. Saving model...")
    
    artifact = {
        "model": model,
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
            "start": str(df_train.index.min()),
            "end": str(df_train.index.max()),
        },
        "metrics": {
            "train": train_metrics,
            "test": test_metrics,
        },
    }
    
    model_path = MODELS_DIR / "y_tb_60_hgb_tuned.joblib"
    
    # Backup old model
    if model_path.exists():
        backup_path = MODELS_DIR / f"y_tb_60_hgb_tuned_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        import shutil
        shutil.copy(model_path, backup_path)
        print(f"   Backed up old model to: {backup_path}")
    
    joblib.dump(artifact, model_path)
    print(f"   Saved new model to: {model_path}")
    
    print("\n" + "=" * 70)
    print("RETRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n   Model trained on: 2024-01-01 to {TRAIN_CUTOFF}")
    print(f"   Test AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   Ready for live trading!")


if __name__ == "__main__":
    main()

