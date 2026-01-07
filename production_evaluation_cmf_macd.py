#!/usr/bin/env python3
"""
PRODUCTION EVALUATION: model3_cmf_macd_v4 @ 0.70 threshold

Walk-forward analysis across 2020-2025:
- Train: 2020-2023 (model already trained on this)
- OOS Test: 2024 (validation year)
- OOS Test: 2025 (true out-of-sample)

Tests for:
1. Consistency across years
2. Performance in different market regimes
3. Drawdown behavior
4. Month-by-month stability
5. Deployment readiness
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
MODEL_PATH = PROJECT_ROOT / "models" / "model3_cmf_macd_v4.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
RESULTS_DIR = PROJECT_ROOT / "research_results" / "production_evaluation"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Configuration
THRESHOLD_LONG = 0.70
THRESHOLD_SHORT = 0.30
COST_PER_TRADE_BPS = 2.5

# Targets
TARGET_WR = 0.52
TARGET_PF = 1.6
TARGET_SHARPE = 0.25
TARGET_MIN_TRADES_DAY = 2


def calculate_metrics(signals: pd.Series, labels: pd.Series, period_name: str = "") -> Dict:
    """Calculate comprehensive metrics for a period."""
    trade_mask = signals != 0
    n_trades = trade_mask.sum()

    if n_trades == 0:
        return None

    signals_trades = signals[trade_mask]
    labels_trades = labels[trade_mask]

    # Win rate
    correct = (signals_trades * labels_trades) > 0
    wins = correct.sum()
    losses = (~correct).sum()
    win_rate = wins / n_trades

    # Returns
    trade_returns = np.where(correct, 1.0, -1.0)

    total_wins = trade_returns[trade_returns > 0].sum()
    total_losses = abs(trade_returns[trade_returns < 0].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else 0

    avg_return = trade_returns.mean()
    std_return = trade_returns.std()
    sharpe = avg_return / std_return if std_return > 0 else 0

    # After costs
    avg_return_after_costs = avg_return - (COST_PER_TRADE_BPS / 10000)
    profitable = avg_return_after_costs > 0

    # Drawdown
    cum_returns = np.cumsum(trade_returns)
    running_max = np.maximum.accumulate(cum_returns)
    drawdown = running_max - cum_returns
    max_dd = drawdown.max()
    max_dd_pct = (max_dd / (running_max.max() + 1e-10)) * 100

    # Calculate days
    timestamps = signals_trades.index
    if len(timestamps) > 0:
        days = (timestamps.max() - timestamps.min()).days + 1
        trades_per_day = n_trades / days if days > 0 else 0
    else:
        days = 0
        trades_per_day = 0

    # Win/loss distribution
    n_long = (signals_trades == 1).sum()
    n_short = (signals_trades == -1).sum()

    if n_long > 0:
        long_wr = correct[signals_trades == 1].mean()
    else:
        long_wr = 0

    if n_short > 0:
        short_wr = correct[signals_trades == -1].mean()
    else:
        short_wr = 0

    # Total R
    total_r = avg_return_after_costs * n_trades

    return {
        'period': period_name,
        'n_trades': int(n_trades),
        'n_long': int(n_long),
        'n_short': int(n_short),
        'days': days,
        'trades_per_day': float(trades_per_day),
        'win_rate': float(win_rate),
        'wins': int(wins),
        'losses': int(losses),
        'long_wr': float(long_wr),
        'short_wr': float(short_wr),
        'profit_factor': float(profit_factor),
        'sharpe': float(sharpe),
        'avg_return_bps': float(avg_return * 10000),
        'avg_return_after_costs_bps': float(avg_return_after_costs * 10000),
        'total_r': float(total_r),
        'max_dd_r': float(max_dd),
        'max_dd_pct': float(max_dd_pct),
        'profitable': bool(profitable),
        'std_return': float(std_return)
    }


def evaluate_period(df: pd.DataFrame, model, features: list,
                   start_date: str, end_date: str, period_name: str) -> Dict:
    """Evaluate model on a specific time period."""

    # Filter to period
    df_period = df[(df.index >= start_date) & (df.index < end_date)].copy()

    if len(df_period) == 0:
        return None

    # Prepare data
    df_clean = df_period.dropna(subset=features)

    if len(df_clean) < 100:
        return None

    # Predict
    X = df_clean[features].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        proba = model.predict_proba(X)[:, 1]
    except:
        return None

    # Generate signals
    signals = pd.Series(0, index=df_clean.index)
    signals[proba >= THRESHOLD_LONG] = 1
    signals[proba <= THRESHOLD_SHORT] = -1

    # Get labels
    if 'y_tb_60' in df_clean.columns:
        labels = df_clean['y_tb_60']
    elif 'y_tb_15' in df_clean.columns:
        labels = df_clean['y_tb_15']
    else:
        return None

    labels = labels.apply(lambda x: 1 if x == 1 else -1)

    # Calculate metrics
    return calculate_metrics(signals, labels, period_name)


def analyze_monthly_consistency(df: pd.DataFrame, model, features: list) -> pd.DataFrame:
    """Analyze performance month by month."""

    monthly_results = []

    # Get all year-month combinations
    df['year_month'] = df.index.to_period('M')

    for ym in df['year_month'].unique():
        df_month = df[df['year_month'] == ym].copy()

        if len(df_month) < 50:
            continue

        # Prepare data
        df_clean = df_month.dropna(subset=features)

        if len(df_clean) < 50:
            continue

        # Predict
        X = df_clean[features].values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        try:
            proba = model.predict_proba(X)[:, 1]
        except:
            continue

        # Generate signals
        signals = pd.Series(0, index=df_clean.index)
        signals[proba >= THRESHOLD_LONG] = 1
        signals[proba <= THRESHOLD_SHORT] = -1

        if (signals != 0).sum() < 10:
            continue

        # Get labels
        if 'y_tb_60' in df_clean.columns:
            labels = df_clean['y_tb_60']
        elif 'y_tb_15' in df_clean.columns:
            labels = df_clean['y_tb_15']
        else:
            continue

        labels = labels.apply(lambda x: 1 if x == 1 else -1)

        # Calculate metrics
        metrics = calculate_metrics(signals, labels, str(ym))
        if metrics:
            monthly_results.append(metrics)

    return pd.DataFrame(monthly_results)


def main():
    print("=" * 100)
    print("PRODUCTION EVALUATION: model3_cmf_macd_v4 @ 0.70/0.30 threshold")
    print("=" * 100)
    print(f"\nConfiguration:")
    print(f"  Model: model3_cmf_macd_v4.joblib")
    print(f"  Long Threshold: {THRESHOLD_LONG}")
    print(f"  Short Threshold: {THRESHOLD_SHORT}")
    print(f"  Transaction Costs: {COST_PER_TRADE_BPS} bps")
    print()

    # Load model
    print("[1] Loading model...")
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    features = model_data['features']
    print(f"  ✓ Model loaded ({len(features)} features)")

    # Load data
    print("\n[2] Loading data...")
    df = pd.read_parquet(DATA_PATH)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    print(f"  ✓ Data loaded: {len(df):,} rows ({df.index.min()} to {df.index.max()})")

    # Add CMF/MACD features
    print("\n[3] Building CMF/MACD features...")
    from src.model3_cmf_macd.features import build_cmf_macd_features
    df = build_cmf_macd_features(df)
    print(f"  ✓ Features added")

    # Check which features are available
    available_features = [f for f in features if f in df.columns]
    print(f"  ✓ Available features: {len(available_features)}/{len(features)}")

    # Walk-forward evaluation
    print("\n[4] Walk-Forward Evaluation...")
    print("=" * 100)

    periods = [
        ("2020-01-01", "2021-01-01", "2020"),
        ("2021-01-01", "2022-01-01", "2021"),
        ("2022-01-01", "2023-01-01", "2022"),
        ("2023-01-01", "2024-01-01", "2023 (Train End)"),
        ("2024-01-01", "2025-01-01", "2024 (Validation)"),
        ("2025-01-01", "2026-01-01", "2025 (OOS)"),
    ]

    yearly_results = []

    for start, end, name in periods:
        print(f"\n  Evaluating {name}...")
        metrics = evaluate_period(df, model, available_features, start, end, name)

        if metrics:
            yearly_results.append(metrics)
            print(f"    Trades: {metrics['n_trades']:,} ({metrics['trades_per_day']:.1f}/day)")
            print(f"    Win Rate: {metrics['win_rate']:.1%}")
            print(f"    Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"    Profit/Trade: {metrics['avg_return_after_costs_bps']:+.1f} bps")
            print(f"    Total R: {metrics['total_r']:+.1f}R")
        else:
            print(f"    ⚠️ No results")

    df_yearly = pd.DataFrame(yearly_results)

    # Summary table
    print("\n" + "=" * 100)
    print("YEARLY PERFORMANCE SUMMARY")
    print("=" * 100)
    print(f"\n{'Period':<20} {'Trades':<10} {'T/Day':<8} {'WR%':<7} {'PF':<7} {'Profit/Tr':<12} {'Total R':<10} {'Status':<10}")
    print("-" * 100)

    for _, row in df_yearly.iterrows():
        status = "✅" if row['profitable'] else "❌"
        print(f"{row['period']:<20} {row['n_trades']:<10,} {row['trades_per_day']:<8.1f} "
              f"{row['win_rate']*100:<6.1f}% {row['profit_factor']:<6.2f}  "
              f"{row['avg_return_after_costs_bps']:>+10.1f} bps {row['total_r']:>9.1f}R {status:<10}")

    # Overall statistics
    print("\n" + "=" * 100)
    print("OVERALL STATISTICS (2020-2025)")
    print("=" * 100)

    total_trades = df_yearly['n_trades'].sum()
    avg_wr = df_yearly['win_rate'].mean()
    avg_pf = df_yearly['profit_factor'].mean()
    avg_profit_bps = df_yearly['avg_return_after_costs_bps'].mean()
    total_r_all = df_yearly['total_r'].sum()
    profitable_years = (df_yearly['profitable'] == True).sum()

    print(f"\nTotal Trades (6 years): {total_trades:,}")
    print(f"Average WR: {avg_wr:.1%}")
    print(f"Average PF: {avg_pf:.2f}")
    print(f"Average Profit/Trade: {avg_profit_bps:+.1f} bps")
    print(f"Total R (6 years): {total_r_all:+.1f}R")
    print(f"Profitable Years: {profitable_years}/6")
    print(f"Avg Trades/Day: {df_yearly['trades_per_day'].mean():.1f}")

    # Consistency metrics
    print("\n" + "=" * 100)
    print("CONSISTENCY ANALYSIS")
    print("=" * 100)

    wr_std = df_yearly['win_rate'].std()
    pf_std = df_yearly['profit_factor'].std()

    print(f"\nWin Rate Consistency:")
    print(f"  Mean: {avg_wr:.1%}")
    print(f"  Std Dev: {wr_std:.1%}")
    print(f"  Range: {df_yearly['win_rate'].min():.1%} - {df_yearly['win_rate'].max():.1%}")

    print(f"\nProfit Factor Consistency:")
    print(f"  Mean: {avg_pf:.2f}")
    print(f"  Std Dev: {pf_std:.2f}")
    print(f"  Range: {df_yearly['profit_factor'].min():.2f} - {df_yearly['profit_factor'].max():.2f}")

    # Monthly consistency
    print("\n[5] Analyzing monthly consistency...")
    df_monthly = analyze_monthly_consistency(df, model, available_features)

    if len(df_monthly) > 0:
        profitable_months = (df_monthly['profitable'] == True).sum()
        total_months = len(df_monthly)

        print(f"\nMonthly Results:")
        print(f"  Total Months: {total_months}")
        print(f"  Profitable Months: {profitable_months} ({profitable_months/total_months:.1%})")
        print(f"  Avg Monthly WR: {df_monthly['win_rate'].mean():.1%}")
        print(f"  Avg Monthly Profit/Trade: {df_monthly['avg_return_after_costs_bps'].mean():+.1f} bps")

        # Save monthly results
        df_monthly.to_csv(RESULTS_DIR / "monthly_performance.csv", index=False)

    # Target achievement
    print("\n" + "=" * 100)
    print("TARGET ACHIEVEMENT (Out-of-Sample Years Only: 2024-2025)")
    print("=" * 100)

    oos_years = df_yearly[df_yearly['period'].isin(['2024 (Validation)', '2025 (OOS)'])]

    if len(oos_years) > 0:
        avg_oos_wr = oos_years['win_rate'].mean()
        avg_oos_pf = oos_years['profit_factor'].mean()
        avg_oos_sharpe = oos_years['sharpe'].mean()
        avg_oos_trades_day = oos_years['trades_per_day'].mean()

        print(f"\nOut-of-Sample Performance (2024-2025):")
        print(f"  Win Rate:       {avg_oos_wr:.1%}  (Target: ≥{TARGET_WR:.0%})  {'✅' if avg_oos_wr >= TARGET_WR else '❌'}")
        print(f"  Profit Factor:  {avg_oos_pf:.2f}  (Target: ≥{TARGET_PF:.1f})  {'✅' if avg_oos_pf >= TARGET_PF else '❌'}")
        print(f"  Sharpe:         {avg_oos_sharpe:.4f}  (Target: ≥{TARGET_SHARPE:.2f})  {'✅' if avg_oos_sharpe >= TARGET_SHARPE else '❌'}")
        print(f"  Trades/Day:     {avg_oos_trades_day:.1f}  (Target: ≥{TARGET_MIN_TRADES_DAY:.0f})  {'✅' if avg_oos_trades_day >= TARGET_MIN_TRADES_DAY else '❌'}")

        targets_met = sum([
            avg_oos_wr >= TARGET_WR,
            avg_oos_pf >= TARGET_PF,
            avg_oos_sharpe >= TARGET_SHARPE,
            avg_oos_trades_day >= TARGET_MIN_TRADES_DAY
        ])

        print(f"\n  Targets Met: {targets_met}/4")

    # Deployment recommendation
    print("\n" + "=" * 100)
    print("DEPLOYMENT RECOMMENDATION")
    print("=" * 100)

    # Calculate confidence score
    confidence_score = 0
    reasons = []

    # Check OOS profitability
    if len(oos_years) > 0 and all(oos_years['profitable']):
        confidence_score += 3
        reasons.append("✅ Profitable in all OOS years (2024-2025)")
    elif len(oos_years) > 0:
        confidence_score += 1
        reasons.append("⚠️ Mixed profitability in OOS years")

    # Check consistency
    if wr_std < 0.05:
        confidence_score += 2
        reasons.append("✅ Highly consistent win rate (low std dev)")
    elif wr_std < 0.10:
        confidence_score += 1
        reasons.append("⚠️ Moderate win rate consistency")

    # Check target achievement
    if targets_met >= 3:
        confidence_score += 2
        reasons.append(f"✅ Meets {targets_met}/4 targets")
    elif targets_met >= 2:
        confidence_score += 1
        reasons.append(f"⚠️ Meets only {targets_met}/4 targets")

    # Check trade frequency
    if avg_oos_trades_day >= 10:
        confidence_score += 2
        reasons.append("✅ Sufficient trade frequency (>10/day)")
    elif avg_oos_trades_day >= 5:
        confidence_score += 1
        reasons.append("⚠️ Moderate trade frequency (5-10/day)")

    # Check profitable months
    if len(df_monthly) > 0:
        monthly_profit_pct = profitable_months / total_months
        if monthly_profit_pct >= 0.70:
            confidence_score += 1
            reasons.append(f"✅ Strong monthly consistency ({monthly_profit_pct:.0%} profitable)")
        elif monthly_profit_pct >= 0.60:
            reasons.append(f"⚠️ Moderate monthly consistency ({monthly_profit_pct:.0%} profitable)")

    max_score = 10
    confidence_pct = (confidence_score / max_score) * 100

    print(f"\nConfidence Score: {confidence_score}/{max_score} ({confidence_pct:.0f}%)")
    print()
    for reason in reasons:
        print(f"  {reason}")

    print()
    if confidence_score >= 8:
        print("🟢 RECOMMENDATION: DEPLOY TO PRODUCTION")
        print("   Model shows strong, consistent performance across all periods.")
        print("   Begin with paper trading for 2 weeks, then scale gradually.")
    elif confidence_score >= 6:
        print("🟡 RECOMMENDATION: DEPLOY WITH CAUTION")
        print("   Model is profitable but shows some inconsistency.")
        print("   Extended paper trading (4 weeks) recommended before live.")
    else:
        print("🔴 RECOMMENDATION: DO NOT DEPLOY")
        print("   Model does not meet minimum reliability standards.")
        print("   Additional research and optimization needed.")

    # Save results
    print("\n" + "=" * 100)
    print("SAVING RESULTS")
    print("=" * 100)

    df_yearly.to_csv(RESULTS_DIR / "yearly_performance.csv", index=False)
    print(f"\n✓ Yearly results saved to: {RESULTS_DIR / 'yearly_performance.csv'}")

    if len(df_monthly) > 0:
        print(f"✓ Monthly results saved to: {RESULTS_DIR / 'monthly_performance.csv'}")

    # Summary report
    summary = {
        'model': 'model3_cmf_macd_v4',
        'threshold_long': THRESHOLD_LONG,
        'threshold_short': THRESHOLD_SHORT,
        'evaluation_date': pd.Timestamp.now().isoformat(),
        'total_trades_6yr': int(total_trades),
        'avg_win_rate': float(avg_wr),
        'avg_profit_factor': float(avg_pf),
        'avg_profit_per_trade_bps': float(avg_profit_bps),
        'total_r_6yr': float(total_r_all),
        'profitable_years': int(profitable_years),
        'confidence_score': int(confidence_score),
        'confidence_pct': float(confidence_pct),
        'recommendation': 'DEPLOY' if confidence_score >= 8 else 'CAUTION' if confidence_score >= 6 else 'NO'
    }

    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "evaluation_summary.csv", index=False)
    print(f"✓ Summary saved to: {RESULTS_DIR / 'evaluation_summary.csv'}")

    print("\n" + "=" * 100)
    print("EVALUATION COMPLETE")
    print("=" * 100)
    print()


if __name__ == "__main__":
    main()
