"""
Backtest for Model #3: CMF and MACD Classifier

Similar to Model #1's backtest, using triple-barrier labels.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from dataclasses import dataclass
from typing import List
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
MODELS_DIR = PROJECT_ROOT / "models" / "model3_cmf_macd"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

# Threshold search grid - balanced range for both long and short
LONG_THRESHOLDS = np.arange(0.52, 0.76, 0.02)   # 0.52 to 0.74
SHORT_THRESHOLDS = np.arange(0.26, 0.50, 0.02)  # 0.26 to 0.48
MIN_TRADES = 100  # Lower minimum to see more combinations


@dataclass
class BacktestResult:
    """Container for backtest results."""
    threshold_long: float
    threshold_short: float
    n_trades: int
    n_longs: int
    n_shorts: int
    win_rate: float
    avg_ret_per_trade: float
    cum_ret: float
    sharpe: float


def load_model():
    """Load Model #3."""
    model_path = MODELS_DIR / "model3_cmf_macd.joblib"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return None
    
    artifact = joblib.load(model_path)
    logger.info(f"Loaded Model #3 from: {model_path}")
    logger.info(f"  Features: {len(artifact['features'])}")
    logger.info(f"  Test AUC: {artifact['metrics'].get('test_auc', 0):.4f}")
    
    return artifact


def load_test_data():
    """Load test data (last 15% of 2020-2025)."""
    features_path = FEATURES_DIR / "xauusd_features_2020_2025.parquet"
    
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        return None
    
    df = pd.read_parquet(features_path)
    logger.info(f"Loaded {len(df):,} bars from features file")
    
    # Build CMF and MACD features
    df = build_cmf_macd_features(df)
    
    # Add labels - use symmetric barriers (1.0/1.0) to avoid directional bias
    df = add_triple_barrier_labels(df, h_max=60, tp_mult=1.0, sl_mult=1.0)
    
    # Get feature columns
    feature_cols = get_feature_columns_for_model3()
    available_features = [f for f in feature_cols if f in df.columns]
    
    # Filter to valid rows
    df_clean = df.dropna(subset=available_features + ['y_tb_60'])
    df_clean = df_clean[df_clean['y_tb_60'] != 0]
    
    # Time-based split (last 15% is test)
    n = len(df_clean)
    test_start = int(n * 0.85)
    df_test = df_clean.iloc[test_start:].copy()
    
    logger.info(f"Test set: {len(df_test):,} bars")
    logger.info(f"  Date range: {df_test.index.min()} to {df_test.index.max()}")
    
    return df_test, available_features


def run_backtest(y_test, proba_test, close_test, threshold_long, threshold_short):
    """Run backtest with given thresholds."""
    # Generate signals
    signal_long = (proba_test >= threshold_long).astype(int)
    signal_short = (proba_test <= threshold_short).astype(int)
    signal = signal_long - signal_short  # +1 for long, -1 for short, 0 for flat

    # Count longs and shorts
    n_longs = int((signal == 1).sum())
    n_shorts = int((signal == -1).sum())

    # Filter to only trades (non-zero signals)
    trade_mask = signal != 0
    n_trades = trade_mask.sum()

    if n_trades == 0:
        return BacktestResult(
            threshold_long=threshold_long,
            threshold_short=threshold_short,
            n_trades=0,
            n_longs=0,
            n_shorts=0,
            win_rate=0.0,
            avg_ret_per_trade=0.0,
            cum_ret=0.0,
            sharpe=0.0,
        )
    
    # Get labels and signals for trades only
    y_trades = y_test[trade_mask]
    signal_trades = signal[trade_mask]
    
    # Compute returns: signal * label
    # Long (+1) on up (+1) = +1 (win)
    # Long (+1) on down (-1) = -1 (loss)
    # Short (-1) on down (-1) = +1 (win)
    # Short (-1) on up (+1) = -1 (loss)
    trade_ret = y_trades * signal_trades
    
    # Compute metrics
    win_rate = float((trade_ret > 0).mean())
    avg_ret = float(trade_ret.mean())
    cum_ret = float(trade_ret.sum())
    
    # Sharpe ratio (annualized)
    trade_std = trade_ret.std(ddof=1) if len(trade_ret) > 1 else 1e-8
    sharpe = float(avg_ret / (trade_std + 1e-8) * np.sqrt(252))
    
    return BacktestResult(
        threshold_long=threshold_long,
        threshold_short=threshold_short,
        n_trades=n_trades,
        n_longs=n_longs,
        n_shorts=n_shorts,
        win_rate=win_rate,
        avg_ret_per_trade=avg_ret,
        cum_ret=cum_ret,
        sharpe=sharpe,
    )


def search_thresholds(y_test, proba_test, close_test):
    """Search for best thresholds."""
    results = []
    
    for tl in LONG_THRESHOLDS:
        for ts in SHORT_THRESHOLDS:
            if ts >= tl:
                continue
            
            res = run_backtest(y_test, proba_test, close_test, tl, ts)
            
            if res.n_trades < MIN_TRADES:
                continue
            
            results.append(res)
    
    # Sort by Sharpe
    results.sort(key=lambda r: (r.sharpe, r.win_rate, r.n_trades), reverse=True)
    
    return results


def main():
    logger.info("="*80)
    logger.info("MODEL #3 BACKTEST")
    logger.info("="*80)
    
    # Load model
    artifact = load_model()
    if artifact is None:
        return
    
    model = artifact['model']
    features = artifact['features']
    
    # Load test data
    df_test, available_features = load_test_data()
    if df_test is None:
        return
    
    # Ensure we have all required features
    missing = [f for f in features if f not in available_features]
    if missing:
        logger.warning(f"Missing {len(missing)} features, using available ones")
        features = [f for f in features if f in available_features]
    
    # Prepare data
    X_test = df_test[features].values
    y_test = df_test['y_tb_60'].values
    close_test = df_test['close'].values
    
    # Map labels: -1 -> 0, +1 -> 1 (for binary classification)
    y_test_binary = np.where(y_test == 1, 1, 0)
    
    # Get predictions
    logger.info("Generating predictions...")
    proba_test = model.predict_proba(X_test)[:, 1]  # Probability of class 1 (up)

    # Probability distribution analysis
    logger.info("\n" + "="*80)
    logger.info("PROBABILITY DISTRIBUTION ANALYSIS")
    logger.info("="*80)
    logger.info(f"  Min: {proba_test.min():.4f}")
    logger.info(f"  Max: {proba_test.max():.4f}")
    logger.info(f"  Mean: {proba_test.mean():.4f}")
    logger.info(f"  Median: {np.median(proba_test):.4f}")
    logger.info(f"  Std: {proba_test.std():.4f}")

    # Distribution buckets
    logger.info("\n  Probability Distribution:")
    buckets = [(0.0, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]
    for low, high in buckets:
        count = ((proba_test >= low) & (proba_test < high)).sum()
        pct = count / len(proba_test) * 100
        bar = "#" * int(pct / 2)
        logger.info(f"    {low:.1f}-{high:.1f}: {count:>6,} ({pct:>5.1f}%) {bar}")
    
    # Search thresholds
    logger.info("Searching for best thresholds...")
    results = search_thresholds(y_test, proba_test, close_test)
    
    if not results:
        logger.warning("No valid threshold combinations found")
        return
    
    # Print best results
    logger.info("\n" + "="*80)
    logger.info("TOP 10 THRESHOLD COMBINATIONS")
    logger.info("="*80)
    logger.info(f"\n{'Idx':>3}  {'tl':>5}  {'ts':>5}  {'trades':>7}  {'L/S':>10}  {'win%':>6}  {'avg_R':>7}  {'cum_R':>8}  {'sharpe':>7}")
    logger.info("-" * 75)

    for i, r in enumerate(results[:10]):
        marker = " *" if i == 0 else ""
        ls_ratio = f"{r.n_longs}/{r.n_shorts}"
        logger.info(f"{i+1:>3}  {r.threshold_long:>5.2f}  {r.threshold_short:>5.2f}  "
                   f"{r.n_trades:>7,}  {ls_ratio:>10}  {r.win_rate*100:>5.1f}%  {r.avg_ret_per_trade:>+7.4f}  "
                   f"{r.cum_ret:>+8.1f}  {r.sharpe:>7.2f}{marker}")
    
    # Best result
    best = results[0]
    logger.info("\n" + "="*80)
    logger.info("BEST CONFIGURATION")
    logger.info("="*80)
    logger.info(f"  Thresholds: Long={best.threshold_long:.2f}, Short={best.threshold_short:.2f}")
    logger.info(f"  Trades: {best.n_trades:,} (Longs: {best.n_longs}, Shorts: {best.n_shorts})")
    long_pct = best.n_longs / best.n_trades * 100 if best.n_trades > 0 else 0
    logger.info(f"  Long/Short Balance: {long_pct:.1f}% / {100-long_pct:.1f}%")
    logger.info(f"  Win Rate: {best.win_rate*100:.1f}%")
    logger.info(f"  Avg R/trade: {best.avg_ret_per_trade:+.4f}")
    logger.info(f"  Cumulative R: {best.cum_ret:+.1f}")
    logger.info(f"  Sharpe: {best.sharpe:.2f}")


if __name__ == "__main__":
    main()

