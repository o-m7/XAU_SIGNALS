#!/usr/bin/env python3
"""
Backtest and Threshold Tuning for y_tb_60 Model.

This script:
1. Loads the tuned model artifact from disk
2. Loads the features parquet
3. Recreates the same 70/15/15 time-based split
4. Runs a simple R-based backtest using triple-barrier labels
5. Tunes signal thresholds on validation set
6. Evaluates best thresholds on test set

Usage:
    cd /Users/omar/Desktop/ML && source xauusd_signals/venv/bin/activate && python xauusd_signals/src/backtest_y_tb_60.py
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "y_tb_60_hgb_tuned.joblib"

# Split ratios (must match train_y_tb_60.py)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Target label
TARGET = "y_tb_60"

# Threshold search grid
LONG_THRESHOLDS = np.arange(0.55, 0.80, 0.05)
SHORT_THRESHOLDS = np.arange(0.20, 0.46, 0.05)

# Minimum trades for a valid config
MIN_TRADES = 200


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BacktestResult:
    """Container for backtest results."""
    threshold_long: float
    threshold_short: float
    n_trades: int
    win_rate: float
    avg_ret_per_trade: float
    cum_ret: float
    sharpe: float


# =============================================================================
# DATA LOADING
# =============================================================================

def load_model_artifact() -> dict:
    """
    Load the tuned model artifact.
    
    Returns:
        Dict with 'model', 'features', 'best_params'
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}\n"
            f"Run train_y_tb_60.py first to train and save the model."
        )
    
    artifact = joblib.load(MODEL_PATH)
    print(f"Loaded model artifact from: {MODEL_PATH}")
    print(f"  Features: {len(artifact['features'])} columns")
    print(f"  Best params: {artifact['best_params']}")
    
    return artifact


def load_features_data() -> pd.DataFrame:
    """
    Load the features parquet file.
    
    Returns:
        DataFrame with features, labels, and close prices
    """
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"Features file not found: {FEATURES_PATH}\n"
            f"Run train_y_tb_60.py first to build features."
        )
    
    df = pd.read_parquet(FEATURES_PATH)
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
    
    df = df.sort_index()
    print(f"Loaded features from: {FEATURES_PATH}")
    print(f"  Rows: {len(df):,}, Columns: {len(df.columns)}")
    
    return df


# =============================================================================
# TIME-BASED SPLITS (reuse same logic as train_y_tb_60.py)
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


# =============================================================================
# BACKTEST FUNCTION
# =============================================================================

def run_backtest(
    y: np.ndarray,
    proba_up: np.ndarray,
    close: np.ndarray,
    threshold_long: float,
    threshold_short: float
) -> BacktestResult:
    """
    Simple directional backtest using y_tb_60 labels and model probabilities.
    
    - Go LONG if proba_up >= threshold_long
    - Go SHORT if proba_up <= threshold_short
    - Otherwise stay flat
    
    Assume:
    - y == +1 → trade would have hit TP before SL → +1R
    - y == -1 → trade would have hit SL before TP → -1R
    - 1 trade per signal, no compounding, R-based PnL.
    
    Args:
        y: Triple-barrier labels (+1 or -1, after filtering 0s)
        proba_up: Model predicted probability of +1 (up)
        close: Close prices (for reference, not directly used in R-based PnL)
        threshold_long: Probability threshold to go long
        threshold_short: Probability threshold to go short
        
    Returns:
        BacktestResult with performance metrics
    """
    # Construct signal array
    signal = np.zeros_like(proba_up, dtype=int)
    signal[proba_up >= threshold_long] = 1    # Long signal
    signal[proba_up <= threshold_short] = -1  # Short signal
    
    # Filter to entries where signal != 0
    trade_mask = signal != 0
    n_trades = int(trade_mask.sum())
    
    if n_trades == 0:
        return BacktestResult(
            threshold_long=threshold_long,
            threshold_short=threshold_short,
            n_trades=0,
            win_rate=np.nan,
            avg_ret_per_trade=0.0,
            cum_ret=0.0,
            sharpe=0.0,
        )
    
    # Extract trades
    y_trades = y[trade_mask]              # +1 or -1 from triple barrier
    signal_trades = signal[trade_mask]    # +1 (long) or -1 (short)
    
    # If signal matches y, profit (+1R); if opposite, loss (-1R)
    # Long (+1) on up (+1) = +1 * +1 = +1 (win)
    # Long (+1) on down (-1) = +1 * -1 = -1 (loss)
    # Short (-1) on down (-1) = -1 * -1 = +1 (win)
    # Short (-1) on up (+1) = -1 * +1 = -1 (loss)
    trade_ret = y_trades * signal_trades
    
    # Compute metrics
    win_rate = float((trade_ret > 0).mean())
    avg_ret = float(trade_ret.mean())
    cum_ret = float(trade_ret.sum())
    
    # Sharpe ratio (annualized, assuming ~252 trading days for relative comparison)
    trade_std = trade_ret.std(ddof=1) if len(trade_ret) > 1 else 1e-8
    sharpe = float(avg_ret / (trade_std + 1e-8) * np.sqrt(252))
    
    return BacktestResult(
        threshold_long=threshold_long,
        threshold_short=threshold_short,
        n_trades=n_trades,
        win_rate=win_rate,
        avg_ret_per_trade=avg_ret,
        cum_ret=cum_ret,
        sharpe=sharpe,
    )


# =============================================================================
# THRESHOLD GRID SEARCH
# =============================================================================

def search_thresholds(
    y: np.ndarray,
    proba_up: np.ndarray,
    close: np.ndarray,
    long_thresholds: np.ndarray = LONG_THRESHOLDS,
    short_thresholds: np.ndarray = SHORT_THRESHOLDS,
    min_trades: int = MIN_TRADES
) -> List[BacktestResult]:
    """
    Search over threshold combinations to find best config.
    
    Args:
        y: Labels
        proba_up: Model probabilities
        close: Close prices
        long_thresholds: Array of long thresholds to try
        short_thresholds: Array of short thresholds to try
        min_trades: Minimum number of trades to consider config valid
        
    Returns:
        List of BacktestResult, sorted by Sharpe descending
    """
    results = []
    
    for tl in long_thresholds:
        for ts in short_thresholds:
            # Skip invalid combinations (would overlap)
            if ts >= tl:
                continue
            
            res = run_backtest(y, proba_up, close, tl, ts)
            
            # Filter out configs with too few trades
            if res.n_trades < min_trades:
                continue
            
            results.append(res)
    
    # Sort by Sharpe, then win_rate, then n_trades (all descending)
    results.sort(
        key=lambda r: (r.sharpe, r.win_rate, r.n_trades),
        reverse=True
    )
    
    return results


# =============================================================================
# PRINTING UTILITIES
# =============================================================================

def print_results_table(results: List[BacktestResult], title: str, top_n: int = 10) -> None:
    """Print a table of backtest results."""
    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}")
    
    if not results:
        print("  No valid configurations found.")
        return
    
    print(f"\n  {'Idx':>3}  {'tl':>5}  {'ts':>5}  {'trades':>7}  {'win%':>6}  {'avg_R':>7}  {'cum_R':>8}  {'sharpe':>7}")
    print(f"  {'-'*60}")
    
    for i, r in enumerate(results[:top_n]):
        marker = " *" if i == 0 else ""
        print(f"  {i+1:>3}  {r.threshold_long:>5.2f}  {r.threshold_short:>5.2f}  "
              f"{r.n_trades:>7,}  {r.win_rate*100:>5.1f}%  {r.avg_ret_per_trade:>+7.4f}  "
              f"{r.cum_ret:>+8.1f}  {r.sharpe:>7.2f}{marker}")
    
    if len(results) > top_n:
        print(f"  ... and {len(results) - top_n} more configurations")


def print_final_summary(val_result: BacktestResult, test_result: BacktestResult) -> None:
    """Print final backtest summary."""
    print(f"\n{'='*70}")
    print("FINAL BACKTEST RESULTS (y_tb_60, tuned model)")
    print(f"{'='*70}")
    
    print(f"\n  Chosen thresholds:")
    print(f"    Long  (go long if proba >= tl):  {val_result.threshold_long:.2f}")
    print(f"    Short (go short if proba <= ts): {val_result.threshold_short:.2f}")
    
    print(f"\n  Validation Set:")
    print(f"    Trades:      {val_result.n_trades:,}")
    print(f"    Win Rate:    {val_result.win_rate*100:.1f}%")
    print(f"    Avg R/trade: {val_result.avg_ret_per_trade:+.4f}")
    print(f"    Cumulative:  {val_result.cum_ret:+.1f} R")
    print(f"    Sharpe:      {val_result.sharpe:.2f}")
    
    print(f"\n  Test Set:")
    print(f"    Trades:      {test_result.n_trades:,}")
    print(f"    Win Rate:    {test_result.win_rate*100:.1f}%")
    print(f"    Avg R/trade: {test_result.avg_ret_per_trade:+.4f}")
    print(f"    Cumulative:  {test_result.cum_ret:+.1f} R")
    print(f"    Sharpe:      {test_result.sharpe:.2f}")
    
    # Comparison
    print(f"\n  Val→Test Comparison:")
    wr_diff = (test_result.win_rate - val_result.win_rate) * 100
    sharpe_diff = test_result.sharpe - val_result.sharpe
    print(f"    Win Rate change:  {wr_diff:+.1f}pp")
    print(f"    Sharpe change:    {sharpe_diff:+.2f}")
    
    if test_result.sharpe > 0:
        print(f"\n  ✓ Test Sharpe is positive ({test_result.sharpe:.2f})")
    else:
        print(f"\n  ⚠ Test Sharpe is not positive ({test_result.sharpe:.2f})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main backtest pipeline."""
    print("=" * 70)
    print("BACKTEST y_tb_60 (with threshold tuning)")
    print("=" * 70)
    
    # Load model artifact
    artifact = load_model_artifact()
    model = artifact["model"]
    features = artifact["features"]
    
    # Load features data
    df = load_features_data()
    
    # Validate features exist
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")
    
    # Validate required columns
    required_cols = [TARGET, "close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    # Prepare data: drop NaN in features and target, filter out y == 0
    cols_to_check = features + [TARGET]
    initial_rows = len(df)
    df_clean = df.dropna(subset=cols_to_check)
    df_clean = df_clean[df_clean[TARGET] != 0]
    
    print(f"\nData preparation:")
    print(f"  Initial rows: {initial_rows:,}")
    print(f"  After NaN/0 filter: {len(df_clean):,}")
    
    # Time-based split
    train_idx, val_idx, test_idx = make_time_splits(len(df_clean))
    
    print(f"\nTime-based split:")
    print(f"  Train: {len(train_idx):,} samples ({100*len(train_idx)/len(df_clean):.1f}%)")
    print(f"  Val:   {len(val_idx):,} samples ({100*len(val_idx)/len(df_clean):.1f}%)")
    print(f"  Test:  {len(test_idx):,} samples ({100*len(test_idx)/len(df_clean):.1f}%)")
    
    # Extract arrays for val and test
    df_val = df_clean.iloc[val_idx]
    df_test = df_clean.iloc[test_idx]
    
    X_val = df_val[features].to_numpy()
    y_val = df_val[TARGET].to_numpy()
    close_val = df_val["close"].to_numpy()
    
    X_test = df_test[features].to_numpy()
    y_test = df_test[TARGET].to_numpy()
    close_test = df_test["close"].to_numpy()
    
    # Map labels: y_tb_60 is already +1/-1, but we may have loaded from binary format
    # The training script maps -1 -> 0, +1 -> 1 for classification
    # But original y_tb_60 is +1/-1, so we need to be consistent
    # Since we filtered y != 0, y should be +1 or -1 (original labels)
    print(f"\n  Val labels: +1={int((y_val == 1).sum()):,}, -1={int((y_val == -1).sum()):,}")
    print(f"  Test labels: +1={int((y_test == 1).sum()):,}, -1={int((y_test == -1).sum()):,}")
    
    # Get predicted probabilities
    print(f"\nGenerating predictions...")
    proba_val = model.predict_proba(X_val)[:, 1]   # Probability of up (class 1)
    proba_test = model.predict_proba(X_test)[:, 1]
    
    print(f"  Val proba range: [{proba_val.min():.3f}, {proba_val.max():.3f}]")
    print(f"  Test proba range: [{proba_test.min():.3f}, {proba_test.max():.3f}]")
    
    # Threshold grid search on validation set
    print(f"\nSearching thresholds on validation set...")
    print(f"  Long thresholds: {LONG_THRESHOLDS}")
    print(f"  Short thresholds: {SHORT_THRESHOLDS}")
    print(f"  Min trades: {MIN_TRADES}")
    
    results_val = search_thresholds(y_val, proba_val, close_val)
    
    print_results_table(results_val, "BEST THRESHOLDS (Validation)", top_n=10)
    
    if not results_val:
        print("\n❌ No valid threshold configurations found on validation set.")
        print("   Try lowering MIN_TRADES or adjusting threshold ranges.")
        return
    
    # Get best config
    best_val = results_val[0]
    print(f"\n  Chosen config: tl={best_val.threshold_long:.2f}, ts={best_val.threshold_short:.2f}")
    
    # Evaluate on test set with chosen thresholds
    print(f"\nEvaluating chosen thresholds on test set...")
    best_test = run_backtest(
        y_test,
        proba_test,
        close_test,
        best_val.threshold_long,
        best_val.threshold_short,
    )
    
    # Print final summary
    print_final_summary(best_val, best_test)
    
    # Also run a simple "no threshold" baseline (predict class directly)
    print(f"\n{'='*70}")
    print("BASELINE COMPARISON (default 0.5 threshold)")
    print(f"{'='*70}")
    
    baseline_val = run_backtest(y_val, proba_val, close_val, 0.5, 0.5)
    baseline_test = run_backtest(y_test, proba_test, close_test, 0.5, 0.5)
    
    print(f"\n  Baseline (tl=0.5, ts=0.5):")
    print(f"    Val:  trades={baseline_val.n_trades:,}, win%={baseline_val.win_rate*100:.1f}%, "
          f"sharpe={baseline_val.sharpe:.2f}")
    print(f"    Test: trades={baseline_test.n_trades:,}, win%={baseline_test.win_rate*100:.1f}%, "
          f"sharpe={baseline_test.sharpe:.2f}")
    
    print(f"\n  Tuned thresholds improvement:")
    if baseline_test.n_trades > 0:
        wr_improve = (best_test.win_rate - baseline_test.win_rate) * 100
        sharpe_improve = best_test.sharpe - baseline_test.sharpe
        print(f"    Win Rate: {wr_improve:+.1f}pp")
        print(f"    Sharpe:   {sharpe_improve:+.2f}")
    else:
        print(f"    Baseline had 0 trades, cannot compare.")
    
    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

