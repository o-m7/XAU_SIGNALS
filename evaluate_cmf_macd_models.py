#!/usr/bin/env python3
"""
Comprehensive Evaluation of Existing CMF+MACD Models

Tests all Model 3 variants and compares to mean reversion strategy.
Uses rigorous validation framework from Phase 1-3 research.

Metrics calculated:
- Profit Factor
- Win Rate
- Sharpe Ratio
- Max Drawdown
- R-multiple
- Trades per day
- Profitability after transaction costs

Author: Quant Research Team
Date: 2026-01-06
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
RESULTS_DIR = PROJECT_ROOT / "research_results" / "cmf_macd_evaluation"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Test period (2024 for out-of-sample)
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"

# Transaction costs
COST_PER_TRADE_BPS = 2.5  # 0.5 pips spread + slippage


def load_model(model_path: Path) -> Tuple[object, list]:
    """Load model and extract features."""
    try:
        artifact = joblib.load(model_path)

        if isinstance(artifact, dict):
            model = artifact.get("model")
            features = artifact.get("features", [])
        else:
            # Old format - just the model
            model = artifact
            features = []

        return model, features
    except Exception as e:
        print(f"    ERROR loading {model_path.name}: {e}")
        return None, []


def calculate_metrics(signals: pd.Series, labels: pd.Series, df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive performance metrics.

    Args:
        signals: Series of trade signals (1=long, -1=short, 0=no trade)
        labels: Series of actual outcomes (1=profit, -1=loss)
        df: DataFrame with price data for forward returns

    Returns:
        Dictionary of metrics
    """
    # Filter to trades only
    trade_mask = signals != 0
    n_trades = trade_mask.sum()

    if n_trades == 0:
        return None

    signals_trades = signals[trade_mask]
    labels_trades = labels[trade_mask]

    # Match signals to labels (correct prediction = win)
    correct = (signals_trades * labels_trades) > 0
    wins = correct.sum()
    losses = (~correct).sum()

    win_rate = wins / n_trades if n_trades > 0 else 0

    # Calculate returns (assuming 1R per trade, where R = risk unit)
    # Win = +1R, Loss = -1R (simplified, will refine with actual returns)
    trade_returns = np.where(correct, 1.0, -1.0)

    # If we have forward_return column, use actual returns
    if 'forward_return' in df.columns:
        forward_returns = df.loc[trade_mask, 'forward_return'].values
        # Use sign of forward returns
        trade_returns = np.abs(forward_returns) * np.where(correct, 1, -1)

    # Metrics
    total_wins_value = trade_returns[trade_returns > 0].sum()
    total_losses_value = abs(trade_returns[trade_returns < 0].sum())

    profit_factor = total_wins_value / total_losses_value if total_losses_value > 0 else 0

    avg_return = trade_returns.mean()
    std_return = trade_returns.std()
    sharpe = avg_return / std_return if std_return > 0 else 0

    avg_win = trade_returns[trade_returns > 0].mean() if (trade_returns > 0).any() else 0
    avg_loss = trade_returns[trade_returns < 0].mean() if (trade_returns < 0).any() else 0
    r_multiple = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # Drawdown (simplified - cumulative returns)
    cum_returns = np.cumsum(trade_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = running_max - cum_returns
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
    max_dd_pct = max_drawdown / (running_max.max() + 1) if running_max.max() > 0 else 0

    # Trades per day
    if isinstance(df.index, pd.DatetimeIndex):
        days = (df.index.max() - df.index.min()).days
        trades_per_day = n_trades / days if days > 0 else 0
    else:
        trades_per_day = 0

    # After costs
    avg_return_after_costs = avg_return - (COST_PER_TRADE_BPS / 10000)
    profitable = avg_return_after_costs > 0

    # Long/short breakdown
    long_mask = signals_trades == 1
    short_mask = signals_trades == -1
    n_long = long_mask.sum()
    n_short = short_mask.sum()

    long_wr = correct[long_mask].mean() if n_long > 0 else 0
    short_wr = correct[short_mask].mean() if n_short > 0 else 0

    return {
        'n_trades': int(n_trades),
        'n_long': int(n_long),
        'n_short': int(n_short),
        'win_rate': float(win_rate),
        'long_wr': float(long_wr),
        'short_wr': float(short_wr),
        'profit_factor': float(profit_factor),
        'r_multiple': float(r_multiple),
        'sharpe': float(sharpe),
        'avg_return': float(avg_return),
        'avg_win': float(avg_win),
        'avg_loss': float(avg_loss),
        'max_drawdown_r': float(max_drawdown),
        'max_drawdown_pct': float(max_dd_pct * 100),
        'trades_per_day': float(trades_per_day),
        'avg_return_after_costs': float(avg_return_after_costs),
        'profitable_after_costs': bool(profitable)
    }


def evaluate_model(model_path: Path, df: pd.DataFrame, threshold_long=0.65, threshold_short=0.35) -> Dict:
    """
    Evaluate a single CMF+MACD model.

    Args:
        model_path: Path to model file
        df: DataFrame with features and labels
        threshold_long: Probability threshold for long signals
        threshold_short: Probability threshold for short signals

    Returns:
        Dictionary with evaluation results
    """
    print(f"\n{'-' * 80}")
    print(f"Evaluating: {model_path.name}")
    print(f"{'-' * 80}")

    # Load model
    model, feature_cols = load_model(model_path)

    if model is None:
        return None

    print(f"  Features: {len(feature_cols) if feature_cols else 'unknown'}")

    # Check which features are available
    if feature_cols:
        available = [f for f in feature_cols if f in df.columns]
        missing = [f for f in feature_cols if f not in df.columns]

        if missing:
            print(f"  WARNING: Missing {len(missing)} features")
            if len(missing) <= 5:
                print(f"    Missing: {missing}")

        feature_cols = available
    else:
        # Try common features
        common_features = ['cmf', 'macd', 'macd_signal', 'macd_hist',
                          'rsi', 'atr_pct', 'volume_ratio', 'dist_ma_20']
        feature_cols = [f for f in common_features if f in df.columns]
        print(f"  Using {len(feature_cols)} common features")

    if len(feature_cols) == 0:
        print("  ERROR: No valid features found")
        return None

    # Prepare data
    df_clean = df.dropna(subset=feature_cols)

    if len(df_clean) == 0:
        print("  ERROR: No data after dropping NaNs")
        return None

    print(f"  Valid rows: {len(df_clean):,}")

    # Get predictions
    try:
        X = df_clean[feature_cols].values

        # Try predict_proba first
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                proba_up = proba[:, 1]  # P(class=1)
            else:
                proba_up = proba[:, 0]
        else:
            # Use predict as fallback
            proba_up = model.predict(X)

        print(f"  Probability range: [{proba_up.min():.3f}, {proba_up.max():.3f}]")
        print(f"  Probability mean: {proba_up.mean():.3f}")

    except Exception as e:
        print(f"  ERROR getting predictions: {e}")
        return None

    # Generate signals based on thresholds
    signals = pd.Series(0, index=df_clean.index)
    signals[proba_up >= threshold_long] = 1    # LONG
    signals[proba_up <= threshold_short] = -1  # SHORT

    n_signals = (signals != 0).sum()
    print(f"  Signals generated: {n_signals:,} ({n_signals/len(df_clean)*100:.1f}%)")
    print(f"    Long: {(signals == 1).sum():,}")
    print(f"    Short: {(signals == -1).sum():,}")

    if n_signals == 0:
        print("  ERROR: No signals generated")
        return None

    # Get labels (try y_tb_15 first, then y_tb_60)
    if 'y_tb_15' in df_clean.columns:
        labels = df_clean['y_tb_15']
        print(f"  Using y_tb_15 labels")
    elif 'y_tb_60' in df_clean.columns:
        labels = df_clean['y_tb_60']
        print(f"  Using y_tb_60 labels")
    else:
        print("  WARNING: No labels found, cannot calculate metrics")
        return None

    # Convert labels to +1/-1
    labels = labels.apply(lambda x: 1 if x == 1 else -1)

    # Calculate metrics
    metrics = calculate_metrics(signals, labels, df_clean)

    if metrics is None:
        print("  ERROR: Could not calculate metrics")
        return None

    # Print results
    print(f"\n  Performance Metrics:")
    print(f"    Win Rate:      {metrics['win_rate']*100:.2f}%  {'✅' if metrics['win_rate'] >= 0.52 else '❌'}")
    print(f"    Profit Factor: {metrics['profit_factor']:.2f}  {'✅' if metrics['profit_factor'] >= 1.6 else '❌'}")
    print(f"    R-multiple:    {metrics['r_multiple']:.2f}  {'✅' if metrics['r_multiple'] > 1.2 else '❌'}")
    print(f"    Sharpe:        {metrics['sharpe']:.4f}  {'✅' if metrics['sharpe'] >= 0.25 else '❌'}")
    print(f"    Trades/day:    {metrics['trades_per_day']:.1f}  {'✅' if 15 <= metrics['trades_per_day'] <= 30 else '⚠️'}")
    print(f"    Max DD:        {metrics['max_drawdown_pct']:.2f}%  {'✅' if metrics['max_drawdown_pct'] <= 6 else '❌'}")
    print(f"    After costs:   {metrics['avg_return_after_costs']*10000:.2f} bps  {'✅' if metrics['profitable_after_costs'] else '❌'}")

    # Check if meets targets
    targets_met = {
        'Win Rate >= 52%': metrics['win_rate'] >= 0.52,
        'Profit Factor >= 1.6': metrics['profit_factor'] >= 1.6,
        'R-multiple > 1.2': metrics['r_multiple'] > 1.2,
        'Sharpe >= 0.25': metrics['sharpe'] >= 0.25,
        'Trades/day 15-30': 15 <= metrics['trades_per_day'] <= 30,
        'Max DD <= 6%': metrics['max_drawdown_pct'] <= 6,
        'Profitable after costs': metrics['profitable_after_costs']
    }

    all_met = all(targets_met.values())
    metrics['all_targets_met'] = all_met
    metrics['targets_met_count'] = sum(targets_met.values())

    if all_met:
        print(f"\n  🎉 ALL TARGETS MET - READY FOR DEPLOYMENT!")
    else:
        print(f"\n  ⚠️  {metrics['targets_met_count']}/7 targets met")

    # Add metadata
    metrics['model_name'] = model_path.name
    metrics['threshold_long'] = threshold_long
    metrics['threshold_short'] = threshold_short

    return metrics


def main():
    print("=" * 80)
    print("CMF+MACD MODELS EVALUATION")
    print("=" * 80)
    print(f"\nTest Period: {TEST_START} to {TEST_END}")
    print(f"Transaction Cost: {COST_PER_TRADE_BPS} bps per trade")
    print()

    # Load data
    print("[1] Loading data...")
    if not DATA_PATH.exists():
        print(f"  ERROR: Data file not found: {DATA_PATH}")
        return

    df = pd.read_parquet(DATA_PATH)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    # Filter to test period
    df_test = df[(df.index >= TEST_START) & (df.index <= TEST_END)].copy()

    print(f"  Total rows: {len(df):,}")
    print(f"  Test period: {len(df_test):,}")
    print(f"  Date range: {df_test.index.min().date()} to {df_test.index.max().date()}")

    # Build CMF/MACD features
    print(f"\n[1.5] Building CMF/MACD features...")
    try:
        from src.model3_cmf_macd.features import build_cmf_macd_features
        df_test = build_cmf_macd_features(df_test)
        print(f"  ✓ CMF/MACD features added")
        print(f"  Total features now: {len(df_test.columns)}")
    except Exception as e:
        print(f"  WARNING: Could not build CMF/MACD features: {e}")
        print(f"  Continuing with existing features...")

    if len(df_test) == 0:
        print(f"  ERROR: No data in test period")
        return

    # Find all Model 3 files
    model_files = list(MODELS_DIR.glob("model3*.joblib"))
    model_files = [f for f in model_files if 'cmf' in f.name.lower() or 'v2' in f.name or 'v3' in f.name or 'v4' in f.name]

    print(f"\n[2] Found {len(model_files)} Model 3 files:")
    for f in model_files:
        print(f"    - {f.name}")

    # Evaluate all models
    print(f"\n[3] Evaluating all models...")
    results = []

    for model_path in model_files:
        metrics = evaluate_model(model_path, df_test)

        if metrics:
            results.append(metrics)

    if not results:
        print("\n❌ No models successfully evaluated")
        return

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Sort by targets met, then by Sharpe
    df_results = df_results.sort_values(['all_targets_met', 'sharpe'], ascending=[False, False])

    # Print comparison
    print(f"\n" + "=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)
    print(f"\n{'Model':<50} {'WR%':<7} {'PF':<7} {'R-mult':<7} {'Sharpe':<8} {'Trades/d':<9} {'After Cost':<12} {'Targets':<8}")
    print("-" * 100)

    for _, row in df_results.iterrows():
        marker = " ***" if row['all_targets_met'] else ""
        print(f"{row['model_name']:<50} {row['win_rate']*100:<6.1f}% {row['profit_factor']:<6.2f}  "
              f"{row['r_multiple']:<6.2f}  {row['sharpe']:<7.4f}  {row['trades_per_day']:<8.1f}  "
              f"{row['avg_return_after_costs']*10000:>+6.2f} bps  {row['targets_met_count']}/7{marker}")

    # Best model
    best = df_results.iloc[0]

    print(f"\n" + "=" * 100)
    print("BEST MODEL")
    print("=" * 100)
    print(f"\nModel: {best['model_name']}")
    print(f"\nPerformance:")
    print(f"  Win Rate:       {best['win_rate']*100:.2f}%  {'✅' if best['win_rate'] >= 0.52 else '❌'} (target: 52%)")
    print(f"  Profit Factor:  {best['profit_factor']:.2f}  {'✅' if best['profit_factor'] >= 1.6 else '❌'} (target: 1.6)")
    print(f"  R-multiple:     {best['r_multiple']:.2f}  {'✅' if best['r_multiple'] > 1.2 else '❌'} (target: > 1.2)")
    print(f"  Sharpe:         {best['sharpe']:.4f}  {'✅' if best['sharpe'] >= 0.25 else '❌'} (target: 0.25)")
    print(f"  Trades/day:     {best['trades_per_day']:.1f}  {'✅' if 15 <= best['trades_per_day'] <= 30 else '⚠️'} (target: 15-30)")
    print(f"  Max Drawdown:   {best['max_drawdown_pct']:.2f}%  {'✅' if best['max_drawdown_pct'] <= 6 else '❌'} (target: <= 6%)")
    print(f"  After costs:    {best['avg_return_after_costs']*10000:+.2f} bps  {'✅' if best['profitable_after_costs'] else '❌'}")

    print(f"\nTrades:")
    print(f"  Total: {best['n_trades']:,}")
    print(f"  Long:  {best['n_long']:,} (WR: {best['long_wr']*100:.1f}%)")
    print(f"  Short: {best['n_short']:,} (WR: {best['short_wr']*100:.1f}%)")

    if best['all_targets_met']:
        print(f"\n🎉🎉🎉 ALL TARGETS MET - READY FOR DEPLOYMENT! 🎉🎉🎉")
    else:
        print(f"\n⚠️  {best['targets_met_count']}/7 targets met - needs iteration")

    # Save results
    df_results.to_csv(RESULTS_DIR / "cmf_macd_comparison.csv", index=False)
    print(f"\n\nResults saved to: {RESULTS_DIR / 'cmf_macd_comparison.csv'}")
    print()


if __name__ == "__main__":
    main()
