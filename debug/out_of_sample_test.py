#!/usr/bin/env python3
"""
Out-of-Sample Testing.

Tests the trained model on completely unseen data:
- 2025 data (future from training)
- 2023 data (past from training, different regime)

This is the ultimate test of model generalization.

Usage:
    cd /Users/omar/Desktop/ML && source xauusd_signals/venv/bin/activate && python xauusd_signals/debug/out_of_sample_test.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from dataclasses import dataclass
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATA_DIR = PROJECT_ROOT.parent / "Data"
MODEL_PATH = PROJECT_ROOT / "models" / "y_tb_60_hgb_tuned.joblib"

# Import feature engineering
from features_complete import build_complete_features


@dataclass
class BacktestResult:
    """Container for backtest results."""
    threshold_long: float
    threshold_short: float
    n_trades: int
    win_rate: float
    avg_ret: float
    cum_ret: float
    sharpe: float


def print_section(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def load_year_data(year: int) -> Optional[pd.DataFrame]:
    """Load and prepare data for a specific year."""
    bars_path = DATA_DIR / "ohlcv_minute" / f"XAUUSD_minute_{year}.parquet"
    quotes_path = DATA_DIR / "quotes" / f"XAUUSD_quotes_{year}.parquet"
    
    if not bars_path.exists():
        print(f"  Bars file not found: {bars_path}")
        return None
    
    print(f"\n  Loading {year} data...")
    
    # Load bars
    bars = pd.read_parquet(bars_path)
    if "timestamp" in bars.columns:
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        bars = bars.set_index("timestamp")
    bars = bars.sort_index()
    print(f"    Bars: {len(bars):,} rows")
    
    # Load quotes
    quotes = None
    if quotes_path.exists():
        quotes = pd.read_parquet(quotes_path)
        if "timestamp" in quotes.columns:
            quotes["timestamp"] = pd.to_datetime(quotes["timestamp"], utc=True)
            quotes = quotes.set_index("timestamp")
        quotes = quotes.sort_index()
        print(f"    Quotes: {len(quotes):,} rows")
    
    # Build features
    print(f"    Building features...")
    try:
        df = build_complete_features(bars, quotes, drop_na=True)
        print(f"    Complete: {len(df):,} rows, {len(df.columns)} columns")
        return df
    except Exception as e:
        print(f"    Error building features: {e}")
        return None


def run_backtest(
    y: np.ndarray,
    proba_up: np.ndarray,
    threshold_long: float,
    threshold_short: float
) -> BacktestResult:
    """Run simple R-based backtest."""
    signal = np.zeros_like(proba_up, dtype=int)
    signal[proba_up >= threshold_long] = 1
    signal[proba_up <= threshold_short] = -1
    
    trade_mask = signal != 0
    n_trades = int(trade_mask.sum())
    
    if n_trades == 0:
        return BacktestResult(threshold_long, threshold_short, 0, np.nan, 0.0, 0.0, 0.0)
    
    y_trades = y[trade_mask]
    signal_trades = signal[trade_mask]
    trade_ret = y_trades * signal_trades
    
    win_rate = float((trade_ret > 0).mean())
    avg_ret = float(trade_ret.mean())
    cum_ret = float(trade_ret.sum())
    trade_std = trade_ret.std(ddof=1) if len(trade_ret) > 1 else 1e-8
    sharpe = float(avg_ret / (trade_std + 1e-8) * np.sqrt(252))
    
    return BacktestResult(threshold_long, threshold_short, n_trades, win_rate, avg_ret, cum_ret, sharpe)


def evaluate_on_year(
    model,
    features: list,
    df: pd.DataFrame,
    year: int,
    threshold_long: float = 0.75,
    threshold_short: float = 0.20
) -> dict:
    """Evaluate model on a year's data."""
    print_section(f"EVALUATION ON {year} DATA (OUT-OF-SAMPLE)")
    
    TARGET = "y_tb_60"
    
    # Check required columns
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  Missing features: {missing[:5]}...")
        return None
    
    if TARGET not in df.columns:
        print(f"  Target '{TARGET}' not found")
        return None
    
    # Prepare data
    df_clean = df.dropna(subset=features + [TARGET])
    df_clean = df_clean[df_clean[TARGET] != 0]
    
    X = df_clean[features].values
    y_raw = df_clean[TARGET].values
    
    # Map to binary for classification metrics
    y_binary = (y_raw == 1).astype(int)
    
    print(f"\n  Data:")
    print(f"    Samples: {len(X):,}")
    print(f"    Date range: {df_clean.index.min().date()} to {df_clean.index.max().date()}")
    print(f"    Label balance: +1={int((y_raw == 1).sum()):,} ({(y_raw == 1).mean()*100:.1f}%), "
          f"-1={int((y_raw == -1).sum()):,} ({(y_raw == -1).mean()*100:.1f}%)")
    
    # Get predictions
    print(f"\n  Generating predictions...")
    proba = model.predict_proba(X)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    
    print(f"    Proba range: [{proba.min():.3f}, {proba.max():.3f}]")
    print(f"    Mean proba: {proba.mean():.3f}")
    
    # Classification metrics
    print(f"\n  Classification Metrics (threshold=0.5):")
    auc = roc_auc_score(y_binary, proba)
    acc = accuracy_score(y_binary, y_pred)
    print(f"    AUC:      {auc:.4f}")
    print(f"    Accuracy: {acc:.4f}")
    
    print(f"\n  Confusion Matrix:")
    cm = confusion_matrix(y_binary, y_pred)
    print(f"                 Pred Down  Pred Up")
    print(f"    True Down:     {cm[0,0]:6,}    {cm[0,1]:6,}")
    print(f"    True Up:       {cm[1,0]:6,}    {cm[1,1]:6,}")
    
    # Backtest with tuned thresholds
    print(f"\n  Backtest with tuned thresholds (tl={threshold_long}, ts={threshold_short}):")
    result = run_backtest(y_raw, proba, threshold_long, threshold_short)
    
    print(f"    Trades:      {result.n_trades:,}")
    print(f"    Win Rate:    {result.win_rate*100:.1f}%")
    print(f"    Avg R/trade: {result.avg_ret:+.4f}")
    print(f"    Cumulative:  {result.cum_ret:+.1f} R")
    print(f"    Sharpe:      {result.sharpe:.2f}")
    
    # Backtest at different thresholds
    print(f"\n  Threshold sensitivity analysis:")
    print(f"    {'tl':<6} {'ts':<6} {'Trades':<10} {'Win%':<10} {'Sharpe':<10}")
    print(f"    {'-'*45}")
    
    for tl in [0.60, 0.65, 0.70, 0.75, 0.80]:
        ts = 1.0 - tl
        res = run_backtest(y_raw, proba, tl, ts)
        print(f"    {tl:<6.2f} {ts:<6.2f} {res.n_trades:<10,} {res.win_rate*100:<10.1f} {res.sharpe:<10.2f}")
    
    return {
        "year": year,
        "n_samples": len(X),
        "auc": auc,
        "accuracy": acc,
        "n_trades": result.n_trades,
        "win_rate": result.win_rate,
        "avg_ret": result.avg_ret,
        "cum_ret": result.cum_ret,
        "sharpe": result.sharpe,
    }


def main():
    print("=" * 70)
    print("  OUT-OF-SAMPLE TESTING")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        print("Run train_y_tb_60.py first")
        return
    
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    features = artifact["features"]
    best_params = artifact.get("best_params", {})
    
    print(f"  Model loaded: {len(features)} features")
    print(f"  Best params: {best_params}")
    
    # Best thresholds from tuning
    threshold_long = 0.75
    threshold_short = 0.20
    print(f"  Using thresholds: tl={threshold_long}, ts={threshold_short}")
    
    results = []
    
    # Test on 2025 (future data - completely unseen)
    df_2025 = load_year_data(2025)
    if df_2025 is not None:
        res = evaluate_on_year(model, features, df_2025, 2025, threshold_long, threshold_short)
        if res:
            results.append(res)
    
    # Test on 2023 (past data - different regime)
    df_2023 = load_year_data(2023)
    if df_2023 is not None:
        res = evaluate_on_year(model, features, df_2023, 2023, threshold_long, threshold_short)
        if res:
            results.append(res)
    
    # Test on 2022 (even older)
    df_2022 = load_year_data(2022)
    if df_2022 is not None:
        res = evaluate_on_year(model, features, df_2022, 2022, threshold_long, threshold_short)
        if res:
            results.append(res)
    
    # Summary comparison
    print_section("SUMMARY COMPARISON")
    
    # Reference: 2024 in-sample performance
    print(f"\n  Reference (2024 test set from training):")
    print(f"    AUC: 0.556, Win Rate: 85.7%, Sharpe: 16.15")
    
    print(f"\n  Out-of-Sample Results:")
    print(f"    {'Year':<8} {'Samples':<12} {'AUC':<8} {'Win%':<8} {'Trades':<10} {'Cum R':<10} {'Sharpe':<10}")
    print(f"    {'-'*65}")
    
    for res in results:
        print(f"    {res['year']:<8} {res['n_samples']:<12,} {res['auc']:<8.4f} "
              f"{res['win_rate']*100:<8.1f} {res['n_trades']:<10,} "
              f"{res['cum_ret']:<+10.1f} {res['sharpe']:<10.2f}")
    
    # Verdict
    print_section("VERDICT")
    
    if results:
        avg_auc = np.mean([r['auc'] for r in results])
        avg_win_rate = np.mean([r['win_rate'] for r in results])
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        
        print(f"\n  Average Out-of-Sample Performance:")
        print(f"    AUC:      {avg_auc:.4f} (in-sample: 0.556)")
        print(f"    Win Rate: {avg_win_rate*100:.1f}% (in-sample: 85.7%)")
        print(f"    Sharpe:   {avg_sharpe:.2f} (in-sample: 16.15)")
        
        auc_degradation = 0.556 - avg_auc
        wr_degradation = 85.7 - avg_win_rate * 100
        
        if avg_auc > 0.52 and avg_win_rate > 0.55 and avg_sharpe > 0:
            print(f"\n  ✓ MODEL GENERALIZES TO OUT-OF-SAMPLE DATA")
            print(f"    Performance remains positive on unseen years")
        elif avg_auc > 0.50:
            print(f"\n  ⚠ MODEL SHOWS DEGRADATION ON OUT-OF-SAMPLE")
            print(f"    Some edge remains but weaker than in-sample")
        else:
            print(f"\n  ❌ MODEL FAILS ON OUT-OF-SAMPLE DATA")
            print(f"    Possible overfitting to 2024 regime")
        
        print(f"\n  Degradation from in-sample:")
        print(f"    AUC drop:      {auc_degradation:+.4f}")
        print(f"    Win Rate drop: {wr_degradation:+.1f}pp")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

