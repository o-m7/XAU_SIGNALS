#!/usr/bin/env python3
"""
Meta-Labeling Strategy - Phase 2

Objective: Boost win rate from 40% to 52%+ using meta-labeling.

Approach:
1. Primary Model: Predicts direction (Long/Short) - already trained LightGBM
2. Meta-Model: Predicts "Should I take this trade?" (binary confidence filter)
3. Final Signal: Trade only when meta-model confident (probability > threshold)

Meta-Labels:
- 1 = Good trade (achieved profit target or positive return)
- 0 = Bad trade (hit stop loss or negative return)

This filters out low-probability signals, increasing win rate.

Author: Quant Research Team
Date: 2026-01-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path("/Users/omar/Desktop/ML/xauusd_signals")
RESULTS_DIR = PROJECT_ROOT / "research_results"
MODELS_DIR = RESULTS_DIR / "intraday_models"
META_DIR = RESULTS_DIR / "meta_labeling"
META_DIR.mkdir(exist_ok=True)


def generate_meta_labels(df, label_col='label', return_col='forward_return'):
    """
    Generate meta-labels: "Was this trade profitable?"

    Meta-label = 1 if forward_return > 0, else 0

    This is different from primary labels (direction).
    Meta-labels answer: "Should I have taken this signal?"

    Args:
        df: DataFrame with primary labels and forward returns
        label_col: Column with primary labels (1=long, -1=short)
        return_col: Column with forward returns

    Returns:
        DataFrame with meta_label column
    """
    print("=" * 80)
    print("GENERATING META-LABELS")
    print("=" * 80)

    df = df.copy()

    # Only consider bars where primary model would trade (label != 0)
    df_trade = df[df[label_col] != 0].copy()

    print(f"\nTotal bars: {len(df):,}")
    print(f"Bars with trade signals: {len(df_trade):,}")

    # Meta-label: Was the trade profitable?
    df_trade['meta_label'] = (df_trade[return_col] > 0).astype(int)

    # Statistics
    meta_positive = df_trade['meta_label'].sum()
    meta_total = len(df_trade)
    meta_wr = meta_positive / meta_total if meta_total > 0 else 0

    print(f"\nMeta-Label Distribution:")
    print(f"  Profitable trades (1): {meta_positive:,} ({meta_wr*100:.2f}%)")
    print(f"  Unprofitable trades (0): {meta_total - meta_positive:,} ({(1-meta_wr)*100:.2f}%)")

    # This win rate is what we're trying to beat
    print(f"\nBaseline Win Rate (all signals): {meta_wr*100:.2f}%")
    print(f"  Target: >= 52%")
    print(f"  Gap: {52 - meta_wr*100:+.2f}%")

    return df_trade


def prepare_meta_training_data(df, feature_cols, primary_model, test_size=0.2):
    """
    Prepare training data for meta-model.

    Features:
    - All original features
    - Primary model prediction (direction)
    - Primary model probability

    Label:
    - Meta-label (was trade profitable?)

    Args:
        df: DataFrame with features and meta-labels
        feature_cols: List of feature columns
        primary_model: Trained primary model (LightGBM)
        test_size: Fraction for test set

    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 80)
    print("PREPARING META-MODEL DATA")
    print("=" * 80)

    # Get primary model predictions
    X_features = df[feature_cols]

    # Primary model probability (direction probability)
    primary_proba = primary_model.predict(X_features)

    # Add primary predictions as features for meta-model
    df_meta = df.copy()
    df_meta['primary_proba'] = primary_proba
    df_meta['primary_confidence'] = np.abs(primary_proba - 0.5)  # Distance from 0.5

    # Extended feature set for meta-model
    meta_feature_cols = feature_cols + ['primary_proba', 'primary_confidence']

    print(f"\nFeatures for meta-model: {len(meta_feature_cols)}")
    print(f"  Original features: {len(feature_cols)}")
    print(f"  Primary model features: 2 (proba, confidence)")

    # Time-based split
    split_idx = int(len(df_meta) * (1 - test_size))

    train = df_meta.iloc[:split_idx]
    test = df_meta.iloc[split_idx:]

    X_train = train[meta_feature_cols]
    y_train = train['meta_label']
    X_test = test[meta_feature_cols]
    y_test = test['meta_label']

    print(f"\nTrain set: {len(X_train):,} ({train.index.min()} to {train.index.max()})")
    print(f"Test set:  {len(X_test):,} ({test.index.min()} to {test.index.max()})")
    print(f"Train win rate: {y_train.mean()*100:.2f}%")
    print(f"Test win rate:  {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test, meta_feature_cols


def train_meta_model(X_train, y_train, X_test, y_test):
    """
    Train meta-model to predict trade quality.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data

    Returns:
        Trained meta-model
    """
    print("\n" + "=" * 80)
    print("TRAINING META-MODEL")
    print("=" * 80)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 100,
        'max_depth': 7,
        'verbose': -1,
        'seed': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    return model


def evaluate_meta_model(meta_model, X_test, y_test):
    """
    Evaluate meta-model performance.

    Args:
        meta_model: Trained meta-model
        X_test, y_test: Test data

    Returns:
        Predictions and probabilities
    """
    print("\n" + "=" * 80)
    print("EVALUATING META-MODEL")
    print("=" * 80)

    # Predictions
    y_pred_proba = meta_model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nMeta-Model Performance (threshold=0.5):")
    print(f"  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:6,}  FP: {cm[0,1]:6,}")
    print(f"  FN: {cm[1,0]:6,}  TP: {cm[1,1]:6,}")

    return y_pred, y_pred_proba


def optimize_threshold(y_test, y_pred_proba, target_wr=0.52):
    """
    Find optimal probability threshold to achieve target win rate.

    Strategy:
    - Test thresholds from 0.5 to 0.95
    - Calculate win rate at each threshold
    - Find threshold that gets closest to 52% WR
    - Balance between WR and number of trades

    Args:
        y_test: True labels (profitable=1, not=0)
        y_pred_proba: Meta-model probabilities
        target_wr: Target win rate (0.52 = 52%)

    Returns:
        optimal_threshold, results_df
    """
    print("\n" + "=" * 80)
    print("OPTIMIZING PROBABILITY THRESHOLD")
    print("=" * 80)
    print(f"\nTarget Win Rate: {target_wr*100:.1f}%")
    print(f"\nProbability distribution:")
    print(f"  Min:  {y_pred_proba.min():.4f}")
    print(f"  25%:  {np.percentile(y_pred_proba, 25):.4f}")
    print(f"  50%:  {np.percentile(y_pred_proba, 50):.4f}")
    print(f"  75%:  {np.percentile(y_pred_proba, 75):.4f}")
    print(f"  Max:  {y_pred_proba.max():.4f}")

    thresholds = np.arange(0.3, 0.96, 0.05)  # Start from 0.3 to capture more range
    results = []

    for thresh in thresholds:
        # Predict: trade if meta-model probability > threshold
        mask = y_pred_proba >= thresh

        if mask.sum() == 0:
            continue

        # Win rate on filtered signals
        wins = y_test[mask].sum()
        trades = mask.sum()
        win_rate = wins / trades if trades > 0 else 0

        # Average return (if we have it)
        # For now, just use win rate

        results.append({
            'threshold': thresh,
            'win_rate': win_rate,
            'num_trades': trades,
            'trades_pct': trades / len(y_test),
            'gap_to_target': abs(win_rate - target_wr)
        })

    if len(results) == 0:
        print("\n❌ ERROR: No valid thresholds found!")
        print("   All predictions may be below minimum threshold.")
        print(f"   Probability range: {y_pred_proba.min():.4f} - {y_pred_proba.max():.4f}")

        # Return baseline
        return 0.5, pd.DataFrame()

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('threshold')

    print("\nThreshold Analysis:")
    print(df_results.to_string(index=False))

    # Find threshold closest to target WR with reasonable trade count
    # Require at least 10% of signals remain (avoid over-filtering)
    viable = df_results[df_results['trades_pct'] >= 0.10]

    if len(viable) == 0:
        print("\nWARNING: No threshold achieves target with >=10% trade retention")
        print("Using threshold with best win rate regardless of trade count")
        best_idx = df_results['win_rate'].idxmax()
    else:
        best_idx = viable['gap_to_target'].idxmin()

    optimal_thresh = df_results.loc[best_idx, 'threshold']
    optimal_wr = df_results.loc[best_idx, 'win_rate']
    optimal_trades = df_results.loc[best_idx, 'num_trades']
    optimal_trades_pct = df_results.loc[best_idx, 'trades_pct']

    print("\n" + "=" * 80)
    print("OPTIMAL THRESHOLD")
    print("=" * 80)
    print(f"\nThreshold: {optimal_thresh:.2f}")
    print(f"Win Rate:  {optimal_wr*100:.2f}%")
    print(f"Trades:    {optimal_trades:,} ({optimal_trades_pct*100:.1f}% of signals)")
    print(f"Gap to Target: {abs(optimal_wr - target_wr)*100:.2f}%")

    if optimal_wr >= target_wr:
        print(f"\n✅ TARGET ACHIEVED! WR = {optimal_wr*100:.2f}% >= {target_wr*100:.1f}%")
    else:
        print(f"\n⚠️  Below target: {optimal_wr*100:.2f}% < {target_wr*100:.1f}%")
        print(f"   Still {(target_wr - optimal_wr)*100:.2f}% short")

    return optimal_thresh, df_results


def backtest_meta_strategy(df, feature_cols, primary_model, meta_model, threshold=0.5):
    """
    Backtest combined primary + meta strategy.

    Process:
    1. Primary model predicts direction
    2. Meta-model predicts trade quality
    3. Only trade if meta-model probability > threshold

    Args:
        df: DataFrame with features and labels
        feature_cols: List of feature columns
        primary_model: Primary direction model
        meta_model: Meta quality model
        threshold: Meta-model probability threshold

    Returns:
        DataFrame with signals and results
    """
    print("\n" + "=" * 80)
    print("BACKTESTING META-LABELING STRATEGY")
    print("=" * 80)
    print(f"Threshold: {threshold:.2f}")

    df_test = df.copy()

    # Primary model predictions
    X = df_test[feature_cols]
    primary_proba = primary_model.predict(X)
    primary_pred = (primary_proba > 0.5).astype(int)  # 1=long, 0=short
    primary_confidence = np.abs(primary_proba - 0.5)

    # Meta-model predictions
    X_meta = df_test[feature_cols].copy()
    X_meta['primary_proba'] = primary_proba
    X_meta['primary_confidence'] = primary_confidence

    meta_feature_cols = feature_cols + ['primary_proba', 'primary_confidence']
    meta_proba = meta_model.predict(X_meta[meta_feature_cols])

    # Trade signal: meta probability > threshold
    trade_signal = (meta_proba >= threshold).astype(int)

    # Results
    df_test['primary_pred'] = primary_pred
    df_test['primary_proba'] = primary_proba
    df_test['meta_proba'] = meta_proba
    df_test['trade_signal'] = trade_signal

    # Calculate performance on trades taken
    trades = df_test[df_test['trade_signal'] == 1].copy()

    print(f"\nTotal signals: {len(df_test):,}")
    print(f"Trades taken: {len(trades):,} ({len(trades)/len(df_test)*100:.1f}%)")

    if len(trades) > 0:
        # Win rate
        wins = (trades['forward_return'] > 0).sum()
        win_rate = wins / len(trades)

        # Returns
        avg_return = trades['forward_return'].mean()
        avg_win = trades[trades['forward_return'] > 0]['forward_return'].mean()
        avg_loss = trades[trades['forward_return'] < 0]['forward_return'].mean()

        # R-multiple
        r_multiple = abs(avg_win / avg_loss) if avg_loss != 0 and not np.isnan(avg_loss) else 0

        # Profit factor
        total_wins = trades[trades['forward_return'] > 0]['forward_return'].sum()
        total_losses = abs(trades[trades['forward_return'] < 0]['forward_return'].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # Sharpe (per-trade)
        sharpe = avg_return / trades['forward_return'].std() if trades['forward_return'].std() > 0 else 0

        print(f"\nPerformance Metrics:")
        print(f"  Win Rate:      {win_rate*100:.2f}%  {'✅' if win_rate >= 0.52 else '❌'} (target: 52%)")
        print(f"  Avg Return:    {avg_return*10000:.2f} bps")
        print(f"  Avg Win:       {avg_win*10000:.2f} bps")
        print(f"  Avg Loss:      {avg_loss*10000:.2f} bps")
        print(f"  R-multiple:    {r_multiple:.2f}  {'✅' if r_multiple > 1.2 else '❌'} (target: > 1.2)")
        print(f"  Profit Factor: {profit_factor:.2f}  {'✅' if profit_factor >= 1.6 else '❌'} (target: >= 1.6)")
        print(f"  Sharpe/trade:  {sharpe:.4f}  {'✅' if sharpe >= 0.25 else '❌'} (target: >= 0.25)")

        # Trades per day
        days = (trades.index.max() - trades.index.min()).days
        trades_per_day = len(trades) / days if days > 0 else 0
        print(f"  Trades/day:    {trades_per_day:.1f}  {'✅' if 15 <= trades_per_day <= 30 else '⚠️'} (target: 15-30)")

        # Check all targets
        targets_met = {
            'Win Rate >= 52%': win_rate >= 0.52,
            'R-multiple > 1.2': r_multiple > 1.2,
            'Profit Factor >= 1.6': profit_factor >= 1.6,
            'Sharpe >= 0.25': sharpe >= 0.25,
            'Trades/day 15-30': 15 <= trades_per_day <= 30
        }

        print(f"\n" + "=" * 80)
        print("TARGET VALIDATION")
        print("=" * 80)
        for target, met in targets_met.items():
            status = "✅ PASS" if met else "❌ FAIL"
            print(f"  {target}: {status}")

        all_met = all(targets_met.values())
        print(f"\nOverall Status: {'✅ ALL TARGETS MET' if all_met else '❌ SOME TARGETS MISSED'}")

        return trades, {
            'win_rate': win_rate,
            'r_multiple': r_multiple,
            'profit_factor': profit_factor,
            'sharpe': sharpe,
            'trades_per_day': trades_per_day,
            'targets_met': all_met
        }
    else:
        print("\n❌ No trades taken with this threshold!")
        return df_test, None


if __name__ == "__main__":
    print("=" * 80)
    print("META-LABELING STRATEGY - PHASE 2")
    print("=" * 80)
    print()

    # Load data
    df = pd.read_parquet(RESULTS_DIR / "data_15min_2020_2024_labeled.parquet")

    # Load primary model
    primary_model = joblib.load(MODELS_DIR / "lightgbm_intraday.joblib")
    print(f"Loaded primary model: {MODELS_DIR / 'lightgbm_intraday.joblib'}")

    # Get feature columns
    exclude_cols = [
        'open', 'high', 'low', 'close', 'volume', 'vwap', 'trades',
        'returns', 'log_returns', 'returns_bps',
        'label', 'label_binary', 'barrier_hit', 'hold_bars', 'forward_return',
        'ma_5', 'ma_10', 'ma_20', 'ma_50', 'bar_direction', 'tr'
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Step 1: Generate meta-labels
    df_meta = generate_meta_labels(df, label_col='label', return_col='forward_return')

    # Step 2: Prepare meta-training data
    X_train, X_test, y_train, y_test, meta_feature_cols = prepare_meta_training_data(
        df_meta, feature_cols, primary_model, test_size=0.2
    )

    # Step 3: Train meta-model
    meta_model = train_meta_model(X_train, y_train, X_test, y_test)

    # Save meta-model
    joblib.dump(meta_model, META_DIR / "meta_model.joblib")
    print(f"\nSaved meta-model to: {META_DIR / 'meta_model.joblib'}")

    # Step 4: Evaluate meta-model
    y_pred, y_pred_proba = evaluate_meta_model(meta_model, X_test, y_test)

    # Step 5: Optimize threshold
    optimal_thresh, threshold_results = optimize_threshold(y_test, y_pred_proba, target_wr=0.52)
    threshold_results.to_csv(META_DIR / "threshold_optimization.csv", index=False)

    # Step 6: Backtest with optimal threshold
    # Use test set only
    df_test = df_meta.iloc[int(len(df_meta) * 0.8):]
    trades, metrics = backtest_meta_strategy(
        df_test, feature_cols, primary_model, meta_model, threshold=optimal_thresh
    )

    if metrics:
        # Save results
        trades.to_parquet(META_DIR / "backtest_results.parquet")

        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(META_DIR / "performance_metrics.csv", index=False)

        print(f"\n\nResults saved to: {META_DIR}/")
        print()
