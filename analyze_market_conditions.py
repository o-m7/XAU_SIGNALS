#!/usr/bin/env python3
"""
MARKET CONDITION ANALYSIS

Analyze model performance across different market regimes:
1. Volatility (high/medium/low)
2. Trend (uptrend/downtrend/ranging)
3. Session (London/NY/Asia/After-hours)
4. Volume conditions
5. Drawdown periods

Goal: Understand when model works best/worst
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
MODEL_PATH = PROJECT_ROOT / "models" / "model3_cmf_macd_v4.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
RESULTS_DIR = PROJECT_ROOT / "research_results" / "market_condition_analysis"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Configuration
THRESHOLD_LONG = 0.70
THRESHOLD_SHORT = 0.30
COST_PER_TRADE_BPS = 2.5


def classify_volatility(df: pd.DataFrame) -> pd.Series:
    """Classify volatility regime."""
    # Use ATR percentile
    if 'atr_pct' in df.columns:
        vol = df['atr_pct']
    else:
        vol = df['close'].pct_change().rolling(20).std()

    low_thresh = vol.quantile(0.33)
    high_thresh = vol.quantile(0.67)

    regime = pd.Series('Medium', index=df.index)
    regime[vol < low_thresh] = 'Low'
    regime[vol > high_thresh] = 'High'

    return regime


def classify_trend(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """Classify trend regime."""
    # Calculate trend using MA slope
    ma = df['close'].rolling(window).mean()
    ma_slope = ma.diff(window)

    slope_pct = ma_slope / ma * 100

    up_thresh = slope_pct.quantile(0.60)
    down_thresh = slope_pct.quantile(0.40)

    regime = pd.Series('Ranging', index=df.index)
    regime[slope_pct > up_thresh] = 'Uptrend'
    regime[slope_pct < down_thresh] = 'Downtrend'

    return regime


def classify_session(df: pd.DataFrame) -> pd.Series:
    """Classify trading session."""
    hour = df.index.hour

    session = pd.Series('After-Hours', index=df.index)
    session[hour.isin(range(0, 7))] = 'Asia'
    session[hour.isin(range(7, 16))] = 'London'
    session[hour.isin(range(16, 22))] = 'NY'

    return session


def analyze_regime_performance(signals: pd.Series, labels: pd.Series,
                               regime: pd.Series, regime_name: str) -> pd.DataFrame:
    """Analyze performance by regime."""
    results = []

    for reg in regime.unique():
        mask = regime == reg
        signals_reg = signals[mask]
        labels_reg = labels[mask]

        trade_mask = signals_reg != 0
        n_trades = trade_mask.sum()

        if n_trades < 10:
            continue

        signals_trades = signals_reg[trade_mask]
        labels_trades = labels_reg[trade_mask]

        correct = (signals_trades * labels_trades) > 0
        win_rate = correct.mean()

        trade_returns = np.where(correct, 1.0, -1.0)
        avg_return = trade_returns.mean() - 0.00025

        total_wins = trade_returns[trade_returns > 0].sum()
        total_losses = abs(trade_returns[trade_returns < 0].sum())
        pf = total_wins / total_losses if total_losses > 0 else 0

        results.append({
            'regime_type': regime_name,
            'regime': reg,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'profit_factor': pf,
            'avg_return_bps': avg_return * 10000,
            'profitable': avg_return > 0
        })

    return pd.DataFrame(results)


def main():
    print("=" * 100)
    print("MARKET CONDITION ANALYSIS - model3_cmf_macd_v4")
    print("=" * 100)
    print()

    # Load model
    print("[1] Loading model...")
    model_data = joblib.load(MODEL_PATH)
    model = model_data['model']
    features = model_data['features']
    print(f"  ✓ Model loaded")

    # Load data
    print("\n[2] Loading data...")
    df = pd.read_parquet(DATA_PATH)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    print(f"  ✓ Data loaded: {len(df):,} rows")

    # Add CMF/MACD features
    print("\n[3] Building features...")
    from src.model3_cmf_macd.features import build_cmf_macd_features
    df = build_cmf_macd_features(df)
    print(f"  ✓ Features added")

    # Use 2024-2025 for analysis (OOS period)
    df_test = df[(df.index >= '2024-01-01') & (df.index < '2026-01-01')].copy()
    print(f"\n[4] Analyzing 2024-2025 (OOS period): {len(df_test):,} bars")

    # Get predictions
    available_features = [f for f in features if f in df_test.columns]
    df_clean = df_test.dropna(subset=available_features + ['y_tb_60'])

    X = df_clean[available_features].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    proba = model.predict_proba(X)[:, 1]

    signals = pd.Series(0, index=df_clean.index)
    signals[proba >= THRESHOLD_LONG] = 1
    signals[proba <= THRESHOLD_SHORT] = -1

    labels = df_clean['y_tb_60'].apply(lambda x: 1 if x == 1 else -1)

    print(f"  Total signals: {(signals != 0).sum():,}")

    # Classify regimes
    print("\n[5] Classifying market regimes...")
    df_clean['volatility_regime'] = classify_volatility(df_clean)
    df_clean['trend_regime'] = classify_trend(df_clean)
    df_clean['session'] = classify_session(df_clean)

    print(f"  ✓ Regimes classified")

    # Analyze by regime
    print("\n[6] Analyzing performance by regime...")
    print()

    all_results = []

    # Volatility
    print("  Analyzing volatility regimes...")
    vol_results = analyze_regime_performance(signals, labels,
                                            df_clean['volatility_regime'],
                                            'Volatility')
    all_results.append(vol_results)

    # Trend
    print("  Analyzing trend regimes...")
    trend_results = analyze_regime_performance(signals, labels,
                                              df_clean['trend_regime'],
                                              'Trend')
    all_results.append(trend_results)

    # Session
    print("  Analyzing session regimes...")
    session_results = analyze_regime_performance(signals, labels,
                                                df_clean['session'],
                                                'Session')
    all_results.append(session_results)

    # Combine results
    df_results = pd.concat(all_results, ignore_index=True)

    # Display results
    print("\n" + "=" * 100)
    print("PERFORMANCE BY MARKET CONDITION")
    print("=" * 100)

    for regime_type in ['Volatility', 'Trend', 'Session']:
        subset = df_results[df_results['regime_type'] == regime_type]

        print(f"\n{regime_type.upper()} REGIMES:")
        print("-" * 100)
        print(f"{'Regime':<20} {'Trades':<10} {'Win Rate':<12} {'PF':<8} {'Profit/Trade':<15} {'Status':<10}")
        print("-" * 100)

        for _, row in subset.iterrows():
            status = "✅" if row['profitable'] else "❌"
            print(f"{row['regime']:<20} {row['n_trades']:<10,} {row['win_rate']:<11.1%} "
                  f"{row['profit_factor']:<7.2f}  {row['avg_return_bps']:>+13.1f} bps {status:<10}")

    # Best/worst conditions
    print("\n" + "=" * 100)
    print("BEST & WORST CONDITIONS")
    print("=" * 100)

    best = df_results.nlargest(3, 'avg_return_bps')
    worst = df_results.nsmallest(3, 'avg_return_bps')

    print("\n🏆 BEST CONDITIONS (Highest Profit/Trade):")
    print("-" * 100)
    for _, row in best.iterrows():
        print(f"  {row['regime_type']}: {row['regime']}")
        print(f"    {row['n_trades']:,} trades, {row['win_rate']:.1%} WR, "
              f"{row['profit_factor']:.2f} PF, {row['avg_return_bps']:+.1f} bps/trade")
        print()

    print("⚠️ WORST CONDITIONS (Lowest Profit/Trade):")
    print("-" * 100)
    for _, row in worst.iterrows():
        print(f"  {row['regime_type']}: {row['regime']}")
        print(f"    {row['n_trades']:,} trades, {row['win_rate']:.1%} WR, "
              f"{row['profit_factor']:.2f} PF, {row['avg_return_bps']:+.1f} bps/trade")
        print()

    # Check if any conditions are unprofitable
    unprofitable = df_results[~df_results['profitable']]

    if len(unprofitable) > 0:
        print("🔴 UNPROFITABLE CONDITIONS:")
        print("-" * 100)
        for _, row in unprofitable.iterrows():
            print(f"  {row['regime_type']}: {row['regime']}")
            print(f"    {row['n_trades']:,} trades, {row['win_rate']:.1%} WR, {row['avg_return_bps']:+.1f} bps/trade")
            print()
    else:
        print("✅ ALL conditions are profitable!")
        print()

    # Monthly breakdown
    print("=" * 100)
    print("MONTHLY BREAKDOWN (2024-2025)")
    print("=" * 100)

    df_clean['year_month'] = df_clean.index.to_period('M')

    monthly_results = []

    for ym in df_clean['year_month'].unique():
        df_month = df_clean[df_clean['year_month'] == ym]
        signals_month = signals.loc[df_month.index]
        labels_month = labels.loc[df_month.index]

        trade_mask = signals_month != 0
        n_trades = trade_mask.sum()

        if n_trades < 5:
            continue

        signals_trades = signals_month[trade_mask]
        labels_trades = labels_month[trade_mask]

        correct = (signals_trades * labels_trades) > 0
        win_rate = correct.mean()

        trade_returns = np.where(correct, 1.0, -1.0)
        avg_return = trade_returns.mean() - 0.00025

        total_wins = trade_returns[trade_returns > 0].sum()
        total_losses = abs(trade_returns[trade_returns < 0].sum())
        pf = total_wins / total_losses if total_losses > 0 else 0

        # Get dominant regime
        vol_mode = df_month['volatility_regime'].mode()[0] if len(df_month['volatility_regime'].mode()) > 0 else 'Unknown'
        trend_mode = df_month['trend_regime'].mode()[0] if len(df_month['trend_regime'].mode()) > 0 else 'Unknown'

        monthly_results.append({
            'month': str(ym),
            'n_trades': n_trades,
            'win_rate': win_rate,
            'profit_factor': pf,
            'avg_return_bps': avg_return * 10000,
            'total_r': avg_return * n_trades,
            'volatility': vol_mode,
            'trend': trend_mode
        })

    df_monthly = pd.DataFrame(monthly_results)

    print()
    print(f"{'Month':<12} {'Trades':<8} {'WR%':<8} {'PF':<7} {'$/Trade':<12} {'Total R':<10} {'Vol':<8} {'Trend':<12}")
    print("-" * 100)

    for _, row in df_monthly.iterrows():
        status = "✅" if row['avg_return_bps'] > 0 else "❌"
        print(f"{row['month']:<12} {row['n_trades']:<8,} {row['win_rate']:<7.1%} "
              f"{row['profit_factor']:<6.2f}  {row['avg_return_bps']:>+10.1f} bps "
              f"{row['total_r']:>8.1f}R  {row['volatility']:<8} {row['trend']:<12} {status}")

    # Save results
    print("\n" + "=" * 100)
    print("SAVING RESULTS")
    print("=" * 100)

    df_results.to_csv(RESULTS_DIR / "regime_performance.csv", index=False)
    df_monthly.to_csv(RESULTS_DIR / "monthly_breakdown_with_regimes.csv", index=False)

    print(f"\n✓ Regime analysis saved to: {RESULTS_DIR / 'regime_performance.csv'}")
    print(f"✓ Monthly breakdown saved to: {RESULTS_DIR / 'monthly_breakdown_with_regimes.csv'}")

    # Recommendations
    print("\n" + "=" * 100)
    print("RECOMMENDATIONS")
    print("=" * 100)

    print("\n1. REGIME FILTERS TO CONSIDER:")

    # Find conditions with <55% WR or negative profit
    weak_conditions = df_results[
        (df_results['win_rate'] < 0.55) |
        (df_results['avg_return_bps'] < 0)
    ]

    if len(weak_conditions) > 0:
        print("\n   Consider AVOIDING these conditions:")
        for _, row in weak_conditions.iterrows():
            print(f"   - {row['regime_type']}: {row['regime']} "
                  f"({row['win_rate']:.1%} WR, {row['avg_return_bps']:+.0f} bps)")
    else:
        print("\n   ✅ All conditions profitable - no filters needed!")

    print("\n2. MARKET REGIMES THE MODEL LIKES:")

    strong_conditions = df_results[df_results['avg_return_bps'] > df_results['avg_return_bps'].median()]

    if len(strong_conditions) > 0:
        print("\n   Model performs BEST in:")
        for _, row in strong_conditions.iterrows():
            print(f"   - {row['regime_type']}: {row['regime']} "
                  f"({row['win_rate']:.1%} WR, {row['avg_return_bps']:+.0f} bps)")

    print("\n3. RETRAINING RECOMMENDATION:")

    # Check if recent months are degrading
    recent_months = df_monthly.tail(6)
    avg_recent_wr = recent_months['win_rate'].mean()
    avg_older_wr = df_monthly.head(6)['win_rate'].mean() if len(df_monthly) > 6 else avg_recent_wr

    degradation = avg_older_wr - avg_recent_wr

    print()
    if degradation > 0.05:
        print(f"   🔴 RETRAIN RECOMMENDED")
        print(f"      Recent 6-month WR: {avg_recent_wr:.1%}")
        print(f"      Older WR: {avg_older_wr:.1%}")
        print(f"      Degradation: {degradation:.1%}")
        print(f"      → Include 2025 data in retraining")
    elif degradation > 0.02:
        print(f"   🟡 MONITOR CLOSELY")
        print(f"      Recent 6-month WR: {avg_recent_wr:.1%}")
        print(f"      Slight degradation: {degradation:.1%}")
        print(f"      → Consider retraining if continues")
    else:
        print(f"   🟢 NO IMMEDIATE RETRAINING NEEDED")
        print(f"      Recent performance stable: {avg_recent_wr:.1%}")
        print(f"      Model still generalizing well")

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)
    print()


if __name__ == "__main__":
    main()
