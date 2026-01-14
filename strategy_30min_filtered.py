#!/usr/bin/env python3
"""
30-Minute Filtered Strategy - Phase 3 (Option A)

Hypothesis: The mean reversion edge is real but too noisy at 15-min frequency.
Solution: Use 30-min bars + strict entry filters to improve signal-to-noise ratio.

Key Changes from Phase 2:
1. 30-minute bars (vs 15-min) → smoother price action
2. STRICT FILTERS to only trade high-probability setups:
   - Filter 1: Extreme mean reversion (|dist_ma_50| > 1.5σ)
   - Filter 2: London or NY session only (exclude Asia/after-hours)
   - Filter 3: Calm volatility (vol < 80th percentile)

Target: 5-10 trades/day with 55%+ WR, PF >= 1.6

Author: Quant Research Team
Date: 2026-01-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path("/Users/omar/Desktop/ML/xauusd_signals")
DATA_DIR = PROJECT_ROOT / "Raw Data"
MINUTE_DIR = DATA_DIR / "ohlcv_minute"
RESULTS_DIR = PROJECT_ROOT / "research_results"
STRATEGY_30MIN_DIR = RESULTS_DIR / "strategy_30min"
STRATEGY_30MIN_DIR.mkdir(exist_ok=True)

np.random.seed(42)


def load_and_resample_30min(years=None):
    """Load minute data and resample to 30-min bars."""
    if years is None:
        years = [2020, 2021, 2022, 2023, 2024]

    print("=" * 80)
    print("LOADING AND RESAMPLING TO 30-MIN BARS")
    print("=" * 80)
    print(f"\nYears: {years}")

    dfs = []
    for year in years:
        path = MINUTE_DIR / f"XAUUSD_minute_{year}.parquet"
        if not path.exists():
            print(f"  WARNING: {path.name} not found, skipping")
            continue

        df = pd.read_parquet(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp').sort_index()
        dfs.append(df)
        print(f"  {year}: {len(df):,} bars")

    # Combine
    combined = pd.concat(dfs, axis=0).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]

    print(f"\nTotal 1-min bars: {len(combined):,}")

    # Resample to 30-min
    resampled = combined.resample('30T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'vwap': 'mean',
        'trades': 'sum'
    }).dropna(subset=['close'])

    print(f"Resampled to 30-min: {len(resampled):,} bars")
    print(f"Date range: {resampled.index.min()} to {resampled.index.max()}")
    print(f"Avg bars/day: {len(resampled) / len(years) / 252:.1f}")

    # Add returns
    resampled['returns'] = resampled['close'].pct_change()
    resampled['log_returns'] = np.log(resampled['close'] / resampled['close'].shift(1))

    return resampled


def quick_eda_30min(df):
    """Quick EDA to verify edge still exists at 30-min."""
    print("\n" + "=" * 80)
    print("QUICK EDA - 30-MIN BARS")
    print("=" * 80)

    returns = df['returns'].dropna()

    # Basic stats
    print(f"\nReturns Statistics:")
    print(f"  Count: {len(returns):,}")
    print(f"  Mean:  {returns.mean() * 10000:.4f} bps")
    print(f"  Std:   {returns.std() * 10000:.4f} bps")
    print(f"  Skew:  {stats.skew(returns):.4f}")
    print(f"  Kurt:  {stats.kurtosis(returns):.4f}")

    # Mean reversion test
    acf_vals = acf(returns, nlags=20, fft=False)
    print(f"\nMean Reversion Test (ACF):")
    print(f"  Lag-1 ACF: {acf_vals[1]:.4f}")

    if acf_vals[1] < -0.01:
        print(f"  ✅ MEAN REVERSION DETECTED (negative ACF)")
    else:
        print(f"  ⚠️  Weak or no mean reversion")

    # Stationarity
    adf = adfuller(returns, regression='c', autolag='AIC')
    print(f"\nStationarity Test (ADF):")
    print(f"  p-value: {adf[1]:.6f}")
    print(f"  Status: {'✅ STATIONARY' if adf[1] < 0.05 else '❌ NON-STATIONARY'}")

    return acf_vals


def add_features_30min(df):
    """Add features optimized for 30-min timeframe."""
    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING - 30-MIN")
    print("=" * 80)

    df = df.copy()

    # Price features (mean reversion focus)
    for lb in [10, 20, 50]:
        df[f'ma_{lb}'] = df['close'].rolling(lb).mean()
        df[f'dist_ma_{lb}'] = (df['close'] - df[f'ma_{lb}']) / df[f'ma_{lb}'] * 100
        rolling_std = df['close'].rolling(lb).std()
        df[f'zscore_ma_{lb}'] = (df['close'] - df[f'ma_{lb}']) / rolling_std
        df[f'roc_{lb}'] = (df['close'] / df['close'].shift(lb) - 1) * 100

    # Volatility features
    for window in [10, 20, 50]:
        df[f'rvol_{window}'] = df['returns'].rolling(window).std()

    df['atr_20'] = df['close'].rolling(20).apply(
        lambda x: np.mean([
            max(x.iloc[i] - x.iloc[i-1], x.iloc[i] - x.iloc[i], x.iloc[i-1] - x.iloc[i])
            for i in range(1, len(x))
        ])
    )
    df['atr_pct_20'] = df['atr_20'] / df['close'] * 100

    # Session features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    def get_session(hour):
        if 0 <= hour < 7:
            return 0  # Asia
        elif 7 <= hour < 16:
            return 1  # London
        elif 16 <= hour < 22:
            return 2  # NY
        else:
            return 3  # After-hours

    df['session'] = df['hour'].apply(get_session)
    df['session_london'] = (df['session'] == 1).astype(int)
    df['session_ny'] = (df['session'] == 2).astype(int)

    # Volume features
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio_20'] = df['volume'] / df['volume_ma_20']

    # Microstructure
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['close_position'] = df['close_position'].fillna(0.5)

    # Momentum
    for lag in [1, 2, 3, 5]:
        df[f'ret_lag{lag}'] = df['returns'].shift(lag)

    print(f"  Features added: {df.shape[1] - 10}")  # Rough count

    return df


def apply_strict_filters(df):
    """
    Apply 3 strict filters to identify high-probability setups only.

    Filters:
    1. Extreme mean reversion: |z-score_ma_50| > 1.5
    2. Session: London or NY only (exclude Asia/after-hours)
    3. Volatility: Below 80th percentile (calm markets)

    Returns:
        df with 'trade_eligible' column
    """
    print("\n" + "=" * 80)
    print("APPLYING STRICT ENTRY FILTERS")
    print("=" * 80)

    df = df.copy()

    # Filter 1: Extreme mean reversion
    # Only trade when price is far from MA (strong reversion signal)
    df['filter_reversion'] = df['zscore_ma_50'].abs() > 1.5

    # Filter 2: Session (London or NY only)
    df['filter_session'] = (df['session'] == 1) | (df['session'] == 2)

    # Filter 3: Volatility (calm markets only - below 80th percentile)
    vol_80th = df['rvol_20'].quantile(0.80)
    df['filter_volatility'] = df['rvol_20'] < vol_80th

    # Combined: ALL filters must pass
    df['trade_eligible'] = (
        df['filter_reversion'] &
        df['filter_session'] &
        df['filter_volatility']
    )

    # Statistics
    total = len(df)
    eligible = df['trade_eligible'].sum()

    print(f"\nFilter Statistics:")
    print(f"  Total bars: {total:,}")
    print(f"  Filter 1 (Extreme Reversion |z| > 1.5): {df['filter_reversion'].sum():,} ({df['filter_reversion'].mean()*100:.1f}%)")
    print(f"  Filter 2 (London/NY session): {df['filter_session'].sum():,} ({df['filter_session'].mean()*100:.1f}%)")
    print(f"  Filter 3 (Vol < 80th percentile): {df['filter_volatility'].sum():,} ({df['filter_volatility'].mean()*100:.1f}%)")
    print(f"\n  ALL FILTERS PASS: {eligible:,} ({eligible/total*100:.1f}%)")

    # Trades per day estimate
    days = (df.index.max() - df.index.min()).days
    trades_per_day = eligible / days if days > 0 else 0
    print(f"  Estimated trades/day: {trades_per_day:.1f}")

    if trades_per_day < 5:
        print(f"  ⚠️  WARNING: Very few trades ({trades_per_day:.1f}/day). May need to relax filters.")
    elif 5 <= trades_per_day <= 10:
        print(f"  ✅ GOOD: {trades_per_day:.1f} trades/day is in target range (5-10)")
    else:
        print(f"  ⚠️  High trade count ({trades_per_day:.1f}/day). Filters may not be strict enough.")

    return df


def create_labels_30min(df):
    """
    Create labels for 30-min strategy.

    Use fixed pip targets (simpler than volatility-scaled):
    - Profit target: +20 pips (~10 bps on XAUUSD)
    - Stop loss: -12 pips (~6 bps)
    - Max hold: 10 bars (5 hours)

    This gives R-multiple of 20/12 = 1.67
    """
    print("\n" + "=" * 80)
    print("GENERATING LABELS - 30-MIN")
    print("=" * 80)
    print("\nFixed Pip Targets:")
    print("  Profit: +20 pips (~10 bps)")
    print("  Stop:   -12 pips (~6 bps)")
    print("  R-multiple: 1.67")
    print("  Max hold: 10 bars (5 hours)")

    df = df.copy()

    # Convert to pips (approximate for XAUUSD)
    pip_value = 0.0005  # $0.50 move = 1 pip on XAUUSD
    profit_target_pct = 20 * pip_value / df['close'].median()
    stop_loss_pct = 12 * pip_value / df['close'].median()
    max_hold = 10

    # Initialize
    df['label'] = 0
    df['forward_return'] = 0.0
    df['hold_bars'] = 0
    df['barrier_hit'] = 0

    prices = df['close'].values

    for i in range(len(df) - max_hold):
        entry_price = prices[i]

        profit_price = entry_price * (1 + profit_target_pct)
        stop_price = entry_price * (1 - stop_loss_pct)

        hit = False
        for j in range(1, max_hold + 1):
            if i + j >= len(prices):
                break

            future_price = prices[i + j]
            forward_ret = (future_price - entry_price) / entry_price

            # Check profit
            if future_price >= profit_price:
                df.iloc[i, df.columns.get_loc('label')] = 1
                df.iloc[i, df.columns.get_loc('forward_return')] = forward_ret
                df.iloc[i, df.columns.get_loc('hold_bars')] = j
                df.iloc[i, df.columns.get_loc('barrier_hit')] = 1
                hit = True
                break

            # Check stop
            if future_price <= stop_price:
                df.iloc[i, df.columns.get_loc('label')] = -1
                df.iloc[i, df.columns.get_loc('forward_return')] = forward_ret
                df.iloc[i, df.columns.get_loc('hold_bars')] = j
                df.iloc[i, df.columns.get_loc('barrier_hit')] = -1
                hit = True
                break

        # Time exit
        if not hit and i + max_hold < len(prices):
            future_price = prices[i + max_hold]
            forward_ret = (future_price - entry_price) / entry_price
            df.iloc[i, df.columns.get_loc('forward_return')] = forward_ret
            df.iloc[i, df.columns.get_loc('hold_bars')] = max_hold

            if forward_ret > 0:
                df.iloc[i, df.columns.get_loc('label')] = 1
            elif forward_ret < 0:
                df.iloc[i, df.columns.get_loc('label')] = -1

    # Label statistics (on eligible trades only)
    eligible = df[df['trade_eligible'] == True]
    if len(eligible) > 0:
        wins = (eligible['label'] == 1).sum()
        losses = (eligible['label'] == -1).sum()
        total = wins + losses

        if total > 0:
            wr = wins / total
            print(f"\nLabel Statistics (eligible trades only):")
            print(f"  Profitable: {wins:,} ({wr*100:.2f}%)")
            print(f"  Unprofitable: {losses:,} ({(1-wr)*100:.2f}%)")
            print(f"  Baseline WR: {wr*100:.2f}%")

            if wr >= 0.55:
                print(f"  ✅ Exceeds 55% target!")
            elif wr >= 0.50:
                print(f"  🟡 Above 50% but below 55% target")
            else:
                print(f"  ❌ Below 50% - filters may need adjustment")

    return df


def train_model_30min(df, feature_cols):
    """Train model on filtered 30-min data."""
    print("\n" + "=" * 80)
    print("TRAINING MODEL - 30-MIN FILTERED")
    print("=" * 80)

    # Only use eligible trades
    df_train = df[df['trade_eligible'] == True].copy()
    df_train = df_train[df_train['label'] != 0]  # Binary: win vs loss

    print(f"\nTraining set (eligible trades only): {len(df_train):,}")

    if len(df_train) < 1000:
        print(f"  ⚠️  WARNING: Very few samples ({len(df_train)}). Model may overfit.")

    # Convert to binary
    df_train['label_binary'] = (df_train['label'] == 1).astype(int)

    # Time split
    split_idx = int(len(df_train) * 0.8)
    train = df_train.iloc[:split_idx]
    test = df_train.iloc[split_idx:]

    X_train = train[feature_cols]
    y_train = train['label_binary']
    X_test = test[feature_cols]
    y_test = test['label_binary']

    print(f"  Train: {len(X_train):,} ({train.index.min()} to {train.index.max()})")
    print(f"  Test:  {len(X_test):,} ({test.index.min()} to {test.index.max()})")
    print(f"  Train WR: {y_train.mean()*100:.2f}%")
    print(f"  Test WR:  {y_test.mean()*100:.2f}%")

    # Train
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 15,  # Smaller tree to avoid overfit
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'min_data_in_leaf': 50,
        'max_depth': 5,
        'verbose': -1,
        'seed': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=50)
        ]
    )

    # Evaluate
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nModel Performance:")
    print(f"  Accuracy: {acc*100:.2f}%")
    print(f"  AUC: {auc:.4f}")

    return model, X_test, y_test, test


def backtest_30min_strategy(df, model, feature_cols):
    """Backtest the full 30-min filtered strategy."""
    print("\n" + "=" * 80)
    print("BACKTESTING 30-MIN STRATEGY")
    print("=" * 80)

    # Test set only (last 20%)
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].copy()

    # Only eligible trades
    trades = df_test[df_test['trade_eligible'] == True].copy()

    print(f"\nTest period: {df_test.index.min()} to {df_test.index.max()}")
    print(f"Total bars: {len(df_test):,}")
    print(f"Eligible trades: {len(trades):,}")

    if len(trades) == 0:
        print("  ❌ No eligible trades in test set!")
        return None, None

    # Get model predictions on eligible trades
    X = trades[feature_cols]
    proba = model.predict(X)

    # Trade if model confidence > 0.5
    trades['model_proba'] = proba
    trades['model_signal'] = (proba > 0.5).astype(int)

    # Filter by model
    trades_taken = trades[trades['model_signal'] == 1].copy()

    print(f"Trades after model filter: {len(trades_taken):,}")

    if len(trades_taken) == 0:
        print("  ❌ Model filtered out all trades!")
        return trades, None

    # Calculate performance
    wins = (trades_taken['forward_return'] > 0).sum()
    losses = (trades_taken['forward_return'] < 0).sum()
    total = len(trades_taken)

    win_rate = wins / total if total > 0 else 0

    avg_return = trades_taken['forward_return'].mean()
    avg_win = trades_taken[trades_taken['forward_return'] > 0]['forward_return'].mean()
    avg_loss = trades_taken[trades_taken['forward_return'] < 0]['forward_return'].mean()

    r_multiple = abs(avg_win / avg_loss) if avg_loss != 0 and not np.isnan(avg_loss) else 0

    total_wins_value = trades_taken[trades_taken['forward_return'] > 0]['forward_return'].sum()
    total_losses_value = abs(trades_taken[trades_taken['forward_return'] < 0]['forward_return'].sum())
    profit_factor = total_wins_value / total_losses_value if total_losses_value > 0 else 0

    sharpe = avg_return / trades_taken['forward_return'].std() if trades_taken['forward_return'].std() > 0 else 0

    # Trades per day
    days = (trades_taken.index.max() - trades_taken.index.min()).days
    trades_per_day = len(trades_taken) / days if days > 0 else 0

    # Transaction costs
    cost_per_trade_bps = 2.5  # 0.5 pips = ~2.5 bps
    avg_return_after_costs = avg_return - (cost_per_trade_bps / 10000)

    print(f"\n" + "=" * 80)
    print("PERFORMANCE METRICS")
    print("=" * 80)
    print(f"\nWin Rate:      {win_rate*100:.2f}%  {'✅' if win_rate >= 0.55 else '🟡' if win_rate >= 0.52 else '❌'} (target: 55%)")
    print(f"R-multiple:    {r_multiple:.2f}  {'✅' if r_multiple > 1.2 else '❌'} (target: > 1.2)")
    print(f"Profit Factor: {profit_factor:.2f}  {'✅' if profit_factor >= 1.6 else '❌'} (target: >= 1.6)")
    print(f"Sharpe/trade:  {sharpe:.4f}  {'✅' if sharpe >= 0.25 else '❌'} (target: >= 0.25)")
    print(f"Trades/day:    {trades_per_day:.1f}  {'✅' if 5 <= trades_per_day <= 10 else '⚠️'} (target: 5-10)")

    print(f"\nReturns:")
    print(f"  Avg Return (before costs): {avg_return*10000:.2f} bps")
    print(f"  Transaction costs:         -{cost_per_trade_bps:.2f} bps")
    print(f"  Avg Return (after costs):  {avg_return_after_costs*10000:.2f} bps")

    if avg_return_after_costs > 0:
        print(f"  ✅ PROFITABLE after costs!")
    else:
        print(f"  ❌ UNPROFITABLE after costs")

    print(f"\nTrade Details:")
    print(f"  Total trades: {total:,}")
    print(f"  Wins:  {wins:,} ({win_rate*100:.1f}%)")
    print(f"  Losses: {losses:,} ({(1-win_rate)*100:.1f}%)")
    print(f"  Avg Win:  {avg_win*10000:.2f} bps")
    print(f"  Avg Loss: {avg_loss*10000:.2f} bps")

    # Target validation
    print(f"\n" + "=" * 80)
    print("TARGET VALIDATION")
    print("=" * 80)

    targets_met = {
        'Win Rate >= 55%': win_rate >= 0.55,
        'R-multiple > 1.2': r_multiple > 1.2,
        'Profit Factor >= 1.6': profit_factor >= 1.6,
        'Sharpe >= 0.25': sharpe >= 0.25,
        'Trades/day 5-10': 5 <= trades_per_day <= 10,
        'Profitable after costs': avg_return_after_costs > 0
    }

    for target, met in targets_met.items():
        status = "✅ PASS" if met else "❌ FAIL"
        print(f"  {target}: {status}")

    all_met = all(targets_met.values())
    print(f"\n{'='*80}")
    print(f"OVERALL: {'🎉 ALL TARGETS MET - READY FOR DEPLOYMENT' if all_met else '❌ SOME TARGETS MISSED - NEEDS ITERATION'}")
    print(f"{'='*80}")

    metrics = {
        'win_rate': win_rate,
        'r_multiple': r_multiple,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'trades_per_day': trades_per_day,
        'avg_return_bps': avg_return * 10000,
        'avg_return_after_costs_bps': avg_return_after_costs * 10000,
        'total_trades': total,
        'targets_met': all_met
    }

    return trades_taken, metrics


if __name__ == "__main__":
    print("=" * 80)
    print("30-MINUTE FILTERED STRATEGY - PHASE 3")
    print("=" * 80)
    print()

    # Step 1: Load and resample
    df = load_and_resample_30min(years=[2020, 2021, 2022, 2023, 2024])

    # Step 2: Quick EDA
    acf_vals = quick_eda_30min(df)

    # Step 3: Add features
    df = add_features_30min(df)

    # Step 4: Apply strict filters
    df = apply_strict_filters(df)

    # Step 5: Create labels
    df = create_labels_30min(df)

    # Drop NaN
    df = df.dropna()

    # Save intermediate data
    df.to_parquet(STRATEGY_30MIN_DIR / "data_30min_filtered.parquet")
    print(f"\nSaved: {STRATEGY_30MIN_DIR / 'data_30min_filtered.parquet'}")

    # Step 6: Get feature columns
    exclude = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trades',
               'returns', 'log_returns', 'label', 'label_binary',
               'forward_return', 'hold_bars', 'barrier_hit',
               'ma_10', 'ma_20', 'ma_50',
               'filter_reversion', 'filter_session', 'filter_volatility', 'trade_eligible']

    feature_cols = [c for c in df.columns if c not in exclude]
    print(f"\nFeatures: {len(feature_cols)}")

    # Step 7: Train model
    model, X_test, y_test, test_df = train_model_30min(df, feature_cols)

    # Save model
    joblib.dump(model, STRATEGY_30MIN_DIR / "model_30min_filtered.joblib")
    print(f"\nSaved model: {STRATEGY_30MIN_DIR / 'model_30min_filtered.joblib'}")

    # Step 8: Backtest
    trades, metrics = backtest_30min_strategy(df, model, feature_cols)

    if trades is not None and metrics is not None:
        # Save results
        trades.to_parquet(STRATEGY_30MIN_DIR / "backtest_trades.parquet")

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(STRATEGY_30MIN_DIR / "performance_metrics.csv", index=False)

        print(f"\n\nResults saved to: {STRATEGY_30MIN_DIR}/")
        print()
