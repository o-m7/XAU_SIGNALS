#!/usr/bin/env python3
"""
COMPREHENSIVE MODEL EVALUATION - ALL 50 MODELS

Tests every model in the models/ directory and ranks by profitability.
Uses same rigorous framework as CMF+MACD evaluation.

Outputs:
- CSV with all model rankings
- Top 10 models summary
- Comparison vs CMF+MACD v4 baseline

Author: Quant Research Team
Date: 2026-01-06
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
RESULTS_DIR = PROJECT_ROOT / "research_results" / "all_models_evaluation"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Test period
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"

# Transaction costs
COST_PER_TRADE_BPS = 2.5

# CMF+MACD v4 baseline (to beat)
BASELINE_WR = 0.616
BASELINE_PF = 1.60
BASELINE_PROFIT = 23.17


def load_model_safe(model_path: Path) -> Tuple[Optional[object], list]:
    """Load model, handling various formats."""
    try:
        artifact = joblib.load(model_path)

        if isinstance(artifact, dict):
            model = artifact.get("model")
            features = artifact.get("features", [])
        else:
            model = artifact
            features = []

        return model, features
    except Exception as e:
        return None, []


def calculate_metrics_fast(signals: pd.Series, labels: pd.Series) -> Optional[Dict]:
    """Fast metrics calculation."""
    trade_mask = signals != 0
    n_trades = trade_mask.sum()

    if n_trades < 50:  # Minimum sample size
        return None

    signals_trades = signals[trade_mask]
    labels_trades = labels[trade_mask]

    # Correct predictions = wins
    correct = (signals_trades * labels_trades) > 0
    wins = correct.sum()

    win_rate = wins / n_trades

    # Returns (1R per trade)
    trade_returns = np.where(correct, 1.0, -1.0)

    total_wins = trade_returns[trade_returns > 0].sum()
    total_losses = abs(trade_returns[trade_returns < 0].sum())

    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    avg_return = trade_returns.mean()
    std_return = trade_returns.std()
    sharpe = avg_return / std_return if std_return > 0 else 0

    avg_return_after_costs = avg_return - (COST_PER_TRADE_BPS / 10000)
    profitable = avg_return_after_costs > 0

    # Drawdown
    cum_returns = np.cumsum(trade_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = running_max - cum_returns
    max_dd = drawdown.max()

    return {
        'n_trades': int(n_trades),
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor),
        'sharpe': float(sharpe),
        'avg_return_bps': float(avg_return * 10000),
        'avg_return_after_costs_bps': float(avg_return_after_costs * 10000),
        'max_dd': float(max_dd),
        'profitable': bool(profitable)
    }


def evaluate_model_fast(model_path: Path, df: pd.DataFrame,
                       threshold_long=0.65, threshold_short=0.35) -> Optional[Dict]:
    """Quick evaluation of a single model."""

    # Load model
    model, feature_cols = load_model_safe(model_path)

    if model is None:
        return None

    # Try to use specified features, fallback to common ones
    if not feature_cols:
        common = ['cmf', 'macd', 'macd_signal', 'rsi', 'atr_pct',
                 'volume_ratio', 'dist_ma_20', 'rvol_20']
        feature_cols = [f for f in common if f in df.columns]

    available = [f for f in feature_cols if f in df.columns]

    if len(available) < 3:  # Need minimum features
        return None

    # Prepare data
    df_clean = df.dropna(subset=available)

    if len(df_clean) < 1000:
        return None

    # Get predictions
    try:
        X = df_clean[available].values

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            proba_up = proba[:, 1] if proba.shape[1] == 2 else proba[:, 0]
        else:
            proba_up = model.predict(X)

        # Check if predictions are valid
        if np.isnan(proba_up).any() or (proba_up == proba_up[0]).all():
            return None

    except Exception:
        return None

    # Generate signals
    signals = pd.Series(0, index=df_clean.index)
    signals[proba_up >= threshold_long] = 1
    signals[proba_up <= threshold_short] = -1

    if (signals != 0).sum() < 50:
        return None

    # Get labels
    if 'y_tb_15' in df_clean.columns:
        labels = df_clean['y_tb_15']
    elif 'y_tb_60' in df_clean.columns:
        labels = df_clean['y_tb_60']
    else:
        return None

    labels = labels.apply(lambda x: 1 if x == 1 else -1)

    # Calculate metrics
    metrics = calculate_metrics_fast(signals, labels)

    if metrics is None:
        return None

    # Add metadata
    metrics['model_name'] = model_path.name
    metrics['model_category'] = get_model_category(model_path.name)

    # Score (for ranking)
    metrics['score'] = (
        metrics['win_rate'] * 100 +
        metrics['profit_factor'] * 50 +
        metrics['sharpe'] * 100 +
        (100 if metrics['profitable'] else 0)
    )

    return metrics


def get_model_category(model_name: str) -> str:
    """Categorize model by name."""
    if 'model1' in model_name:
        return 'Model1_Trend'
    elif 'model3' in model_name or 'cmf' in model_name or 'macd' in model_name:
        return 'Model3_CMF_MACD'
    elif 'model4' in model_name:
        return 'Model4'
    elif 'model5' in model_name or 'meanrev' in model_name:
        return 'Model5_MeanRev'
    elif 'model6' in model_name or 'micro' in model_name or 'orderflow' in model_name:
        return 'Model6_Microstructure'
    elif 'model7' in model_name:
        return 'Model7_RawPrice'
    elif 'model8' in model_name or 'momentum' in model_name:
        return 'Model8_Momentum'
    elif 'model9' in model_name or 'rejection' in model_name:
        return 'Model9_Rejection'
    elif 'ensemble' in model_name:
        return 'Ensemble'
    elif 'sniper' in model_name:
        return 'Sniper'
    elif 'breakout' in model_name:
        return 'Breakout'
    elif '5min' in model_name:
        return '5min_Ultra'
    else:
        return 'Other'


def main():
    print("=" * 80)
    print("COMPREHENSIVE MODEL EVALUATION - ALL 50 MODELS")
    print("=" * 80)
    print(f"\nTest Period: {TEST_START} to {TEST_END}")
    print(f"Baseline to Beat: CMF+MACD v4")
    print(f"  - Win Rate: {BASELINE_WR*100:.1f}%")
    print(f"  - Profit Factor: {BASELINE_PF:.2f}")
    print(f"  - Profit/Trade: +{BASELINE_PROFIT:.2f} bps")
    print()

    # Load data
    print("[1] Loading data...")
    df = pd.read_parquet(DATA_PATH)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    df_test = df[(df.index >= TEST_START) & (df.index <= TEST_END)].copy()

    print(f"  Test period: {len(df_test):,} rows")

    # Build CMF/MACD features
    print(f"\n[2] Building CMF/MACD features...")
    try:
        from src.model3_cmf_macd.features import build_cmf_macd_features
        df_test = build_cmf_macd_features(df_test)
        print(f"  ✓ Features added")
    except Exception as e:
        print(f"  WARNING: {e}")

    # Find all models
    model_files = sorted(MODELS_DIR.glob("*.joblib"))
    # Exclude journal files
    model_files = [f for f in model_files if 'journal' not in f.name.lower()]

    print(f"\n[3] Found {len(model_files)} models to evaluate")
    print(f"  (Excluding .journal files)")

    # Evaluate all models
    print(f"\n[4] Evaluating all models (this may take 5-10 minutes)...")
    results = []

    for i, model_path in enumerate(model_files, 1):
        if i % 10 == 0 or i == 1:
            print(f"  Progress: {i}/{len(model_files)} models evaluated...")

        metrics = evaluate_model_fast(model_path, df_test)

        if metrics:
            results.append(metrics)

    print(f"\n  Successfully evaluated: {len(results)}/{len(model_files)} models")

    if not results:
        print("\n❌ No models successfully evaluated")
        return

    # Convert to DataFrame and sort
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('score', ascending=False)

    # Save full results
    df_results.to_csv(RESULTS_DIR / "all_models_ranked.csv", index=False)

    # Print top 10
    print(f"\n" + "=" * 120)
    print("TOP 10 MODELS (Ranked by Score)")
    print("=" * 120)
    print(f"\n{'Rank':<5} {'Model':<45} {'Category':<20} {'WR%':<7} {'PF':<7} {'Profit':<10} {'Score':<7}")
    print("-" * 120)

    for i, (_, row) in enumerate(df_results.head(10).iterrows(), 1):
        marker = " 🏆" if i == 1 else " 🥈" if i == 2 else " 🥉" if i == 3 else ""
        profit_str = f"+{row['avg_return_after_costs_bps']:.1f}" if row['profitable'] else f"{row['avg_return_after_costs_bps']:.1f}"

        print(f"{i:<5} {row['model_name']:<45} {row['model_category']:<20} "
              f"{row['win_rate']*100:<6.1f}% {row['profit_factor']:<6.2f}  "
              f"{profit_str:<9} bps {row['score']:<6.0f}{marker}")

    # Compare top 3 to baseline
    print(f"\n" + "=" * 120)
    print("TOP 3 vs CMF+MACD v4 BASELINE")
    print("=" * 120)

    top3 = df_results.head(3)

    comparison_data = []

    # Add baseline
    comparison_data.append({
        'Model': 'CMF+MACD v4 (Baseline)',
        'WR%': BASELINE_WR * 100,
        'PF': BASELINE_PF,
        'Profit_bps': BASELINE_PROFIT,
        'Status': 'Baseline'
    })

    # Add top 3
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        comparison_data.append({
            'Model': row['model_name'],
            'WR%': row['win_rate'] * 100,
            'PF': row['profit_factor'],
            'Profit_bps': row['avg_return_after_costs_bps'],
            'Status': f"Rank #{i}"
        })

    df_comp = pd.DataFrame(comparison_data)
    print(f"\n{df_comp.to_string(index=False)}")

    # Best model analysis
    best = df_results.iloc[0]

    print(f"\n" + "=" * 120)
    print("🏆 BEST MODEL")
    print("=" * 120)
    print(f"\nModel: {best['model_name']}")
    print(f"Category: {best['model_category']}")
    print(f"\nPerformance:")
    print(f"  Win Rate:       {best['win_rate']*100:.2f}%  {'✅' if best['win_rate'] >= BASELINE_WR else '❌'}")
    print(f"  Profit Factor:  {best['profit_factor']:.2f}  {'✅' if best['profit_factor'] >= BASELINE_PF else '❌'}")
    print(f"  Profit/Trade:   {best['avg_return_after_costs_bps']:+.2f} bps  {'✅' if best['avg_return_after_costs_bps'] >= BASELINE_PROFIT else '❌'}")
    print(f"  Sharpe:         {best['sharpe']:.4f}")
    print(f"  Trades:         {best['n_trades']:,}")
    print(f"  Max DD:         {best['max_dd']:.2f}R")
    print(f"  Profitable:     {'✅ YES' if best['profitable'] else '❌ NO'}")

    # Comparison to baseline
    wr_diff = (best['win_rate'] - BASELINE_WR) * 100
    pf_diff = ((best['profit_factor'] / BASELINE_PF) - 1) * 100
    profit_diff = best['avg_return_after_costs_bps'] - BASELINE_PROFIT

    print(f"\nVs CMF+MACD v4 Baseline:")
    print(f"  Win Rate:       {wr_diff:+.1f}% {'better' if wr_diff > 0 else 'worse'}")
    print(f"  Profit Factor:  {pf_diff:+.1f}% {'better' if pf_diff > 0 else 'worse'}")
    print(f"  Profit/Trade:   {profit_diff:+.2f} bps {'better' if profit_diff > 0 else 'worse'}")

    if best['avg_return_after_costs_bps'] > BASELINE_PROFIT:
        print(f"\n🎉 FOUND BETTER MODEL! This beats CMF+MACD v4!")
    elif abs(profit_diff) < 5:
        print(f"\n✅ COMPARABLE TO BASELINE (within 5 bps)")
    else:
        print(f"\n⚠️ CMF+MACD v4 remains the best choice")

    # Category breakdown
    print(f"\n" + "=" * 120)
    print("PERFORMANCE BY CATEGORY")
    print("=" * 120)

    cat_summary = df_results.groupby('model_category').agg({
        'win_rate': 'mean',
        'profit_factor': 'mean',
        'avg_return_after_costs_bps': 'mean',
        'profitable': 'sum',
        'model_name': 'count'
    }).round(3)

    cat_summary.columns = ['Avg_WR', 'Avg_PF', 'Avg_Profit_bps', 'Profitable_Count', 'Total_Models']
    cat_summary = cat_summary.sort_values('Avg_Profit_bps', ascending=False)

    print(f"\n{cat_summary.to_string()}")

    print(f"\n\nResults saved to:")
    print(f"  - {RESULTS_DIR / 'all_models_ranked.csv'}")
    print()


if __name__ == "__main__":
    main()
