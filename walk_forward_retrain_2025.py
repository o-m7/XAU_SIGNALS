#!/usr/bin/env python3
"""
WALK-FORWARD RETRAINING - 2025

Retrain model month-by-month on expanding window:
- Start: Train on 2020-2024, test on Jan 2025
- Month 2: Train on 2020-Jan 2025, test on Feb 2025
- Month 3: Train on 2020-Feb 2025, test on Mar 2025
- etc.

Compare retrained model vs original model (trained 2020-2023).
Goal: Determine if including 2025 data improves forward performance.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Paths
ORIGINAL_MODEL_PATH = PROJECT_ROOT / "models" / "model3_cmf_macd_v4.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
RESULTS_DIR = PROJECT_ROOT / "research_results" / "walk_forward_2025"
MODELS_RETRAINED_DIR = PROJECT_ROOT / "models" / "retrained_2025"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
MODELS_RETRAINED_DIR.mkdir(exist_ok=True, parents=True)

# Configuration
THRESHOLD_LONG = 0.70
THRESHOLD_SHORT = 0.30
COST_PER_TRADE_BPS = 2.5

# Training configuration (same as original)
TRAIN_CONFIG = {
    'max_depth': 4,
    'learning_rate': 0.03,
    'max_iter': 400,
    'min_samples_leaf': 200,
    'l2_regularization': 0.1,
    'early_stopping': True,
    'n_iter_no_change': 10,
    'validation_fraction': 0.1,
    'random_state': 42
}


def evaluate_month(model, df_month: pd.DataFrame, features: list) -> dict:
    """Evaluate model on a single month."""
    df_clean = df_month.dropna(subset=features + ['y_tb_60'])

    if len(df_clean) < 50:
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

    trade_mask = signals != 0
    n_trades = trade_mask.sum()

    if n_trades < 10:
        return None

    # Get labels
    labels = df_clean['y_tb_60'].apply(lambda x: 1 if x == 1 else -1)

    signals_trades = signals[trade_mask]
    labels_trades = labels[trade_mask]

    # Calculate metrics
    correct = (signals_trades * labels_trades) > 0
    win_rate = correct.mean()

    trade_returns = np.where(correct, 1.0, -1.0)
    avg_return = trade_returns.mean() - (COST_PER_TRADE_BPS / 10000)

    total_wins = trade_returns[trade_returns > 0].sum()
    total_losses = abs(trade_returns[trade_returns < 0].sum())
    pf = total_wins / total_losses if total_losses > 0 else 0

    total_r = avg_return * n_trades

    return {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'profit_factor': pf,
        'avg_return_bps': avg_return * 10000,
        'total_r': total_r
    }


def train_model(df_train: pd.DataFrame, features: list) -> HistGradientBoostingClassifier:
    """Train a new model on given data."""
    # Prepare training data
    df_clean = df_train.dropna(subset=features + ['y_tb_60'])
    df_clean = df_clean[df_clean['y_tb_60'] != 0]

    X_train = df_clean[features].values
    y_train = (df_clean['y_tb_60'] == 1).astype(int).values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)

    # Train
    model = HistGradientBoostingClassifier(**TRAIN_CONFIG)
    model.fit(X_train, y_train)

    return model


def main():
    print("=" * 100)
    print("WALK-FORWARD RETRAINING - 2025")
    print("=" * 100)
    print(f"\nStrategy: Train on expanding window, validate on next month")
    print(f"Original Model: Trained on 2020-2023")
    print(f"Comparison: Original vs Retrained performance")
    print()

    # Load original model
    print("[1] Loading original model...")
    original_model_data = joblib.load(ORIGINAL_MODEL_PATH)
    original_model = original_model_data['model']
    features = original_model_data['features']
    print(f"  ✓ Original model loaded ({len(features)} features)")
    print(f"  ✓ Trained on: 2020-2023")

    # Load data
    print("\n[2] Loading data...")
    df = pd.read_parquet(DATA_PATH)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')

    print(f"  ✓ Data loaded: {len(df):,} rows")

    # Add features
    print("\n[3] Building CMF/MACD features...")
    from src.model3_cmf_macd.features import build_cmf_macd_features
    df = build_cmf_macd_features(df)
    print(f"  ✓ Features added")

    # Get available features
    available_features = [f for f in features if f in df.columns]
    print(f"  ✓ Available: {len(available_features)}/{len(features)} features")

    # Define 2025 months
    months_2025 = pd.period_range('2025-01', '2025-12', freq='M')

    print("\n[4] Walk-Forward Evaluation...")
    print("=" * 100)

    results = []

    for i, test_month in enumerate(months_2025, 1):
        test_start = test_month.to_timestamp().tz_localize('UTC')
        test_end = (test_month + 1).to_timestamp().tz_localize('UTC')

        # Get test data
        df_test = df[(df.index >= test_start) & (df.index < test_end)].copy()

        if len(df_test) < 100:
            print(f"\n  [{i}/12] {test_month}: Skipping (insufficient data)")
            continue

        print(f"\n  [{i}/12] {test_month}:")

        # Evaluate original model
        orig_metrics = evaluate_month(original_model, df_test, available_features)

        if orig_metrics is None:
            print(f"    ⚠️ Insufficient trades")
            continue

        print(f"    Original Model:")
        print(f"      Trades: {orig_metrics['n_trades']:,}, WR: {orig_metrics['win_rate']:.1%}, "
              f"PF: {orig_metrics['profit_factor']:.2f}, Total R: {orig_metrics['total_r']:+.1f}R")

        # Train new model on data up to previous month
        train_end = test_start
        train_start_date = pd.Timestamp('2020-01-01').tz_localize('UTC')
        df_train = df[(df.index >= train_start_date) & (df.index < train_end)].copy()

        if len(df_train) < 10000:
            print(f"    ⚠️ Insufficient training data")
            continue

        print(f"    Training new model on data up to {train_end.strftime('%Y-%m')}...")
        retrained_model = train_model(df_train, available_features)

        # Evaluate retrained model
        retrained_metrics = evaluate_month(retrained_model, df_test, available_features)

        if retrained_metrics is None:
            print(f"    ⚠️ Retrained model failed")
            continue

        print(f"    Retrained Model:")
        print(f"      Trades: {retrained_metrics['n_trades']:,}, WR: {retrained_metrics['win_rate']:.1%}, "
              f"PF: {retrained_metrics['profit_factor']:.2f}, Total R: {retrained_metrics['total_r']:+.1f}R")

        # Compare
        wr_improvement = retrained_metrics['win_rate'] - orig_metrics['win_rate']
        pf_improvement = retrained_metrics['profit_factor'] - orig_metrics['profit_factor']
        r_improvement = retrained_metrics['total_r'] - orig_metrics['total_r']

        if wr_improvement > 0 and r_improvement > 0:
            status = "✅ BETTER"
        elif wr_improvement < -0.02 or r_improvement < -10:
            status = "❌ WORSE"
        else:
            status = "≈ SIMILAR"

        print(f"    Improvement: WR {wr_improvement:+.1%}, PF {pf_improvement:+.2f}, R {r_improvement:+.1f}R {status}")

        # Save model if better
        if r_improvement > 0:
            model_path = MODELS_RETRAINED_DIR / f"model3_cmf_macd_retrained_{test_month}.joblib"
            artifact = {
                'model': retrained_model,
                'features': available_features,
                'threshold_long': THRESHOLD_LONG,
                'threshold_short': THRESHOLD_SHORT,
                'trained_on': f"2020-01-01 to {train_end.strftime('%Y-%m-%d')}",
                'saved_at': datetime.now().isoformat()
            }
            joblib.dump(artifact, model_path)
            print(f"    💾 Saved improved model: {model_path.name}")

        # Store results
        results.append({
            'month': str(test_month),
            'original_trades': orig_metrics['n_trades'],
            'original_wr': orig_metrics['win_rate'],
            'original_pf': orig_metrics['profit_factor'],
            'original_total_r': orig_metrics['total_r'],
            'retrained_trades': retrained_metrics['n_trades'],
            'retrained_wr': retrained_metrics['win_rate'],
            'retrained_pf': retrained_metrics['profit_factor'],
            'retrained_total_r': retrained_metrics['total_r'],
            'wr_improvement': wr_improvement,
            'pf_improvement': pf_improvement,
            'r_improvement': r_improvement,
            'status': status
        })

    # Summary
    print("\n" + "=" * 100)
    print("WALK-FORWARD SUMMARY")
    print("=" * 100)

    df_results = pd.DataFrame(results)

    print()
    print(f"{'Month':<12} {'Orig WR':<10} {'Retrain WR':<12} {'Orig PF':<10} {'Retrain PF':<12} "
          f"{'R Improve':<12} {'Status':<10}")
    print("-" * 100)

    for _, row in df_results.iterrows():
        print(f"{row['month']:<12} {row['original_wr']:<9.1%} {row['retrained_wr']:<11.1%} "
              f"{row['original_pf']:<9.2f} {row['retrained_pf']:<11.2f} "
              f"{row['r_improvement']:>+10.1f}R  {row['status']:<10}")

    # Overall statistics
    print("\n" + "=" * 100)
    print("OVERALL COMPARISON")
    print("=" * 100)

    total_orig_r = df_results['original_total_r'].sum()
    total_retrained_r = df_results['retrained_total_r'].sum()
    total_improvement = total_retrained_r - total_orig_r

    avg_orig_wr = df_results['original_wr'].mean()
    avg_retrained_wr = df_results['retrained_wr'].mean()

    avg_orig_pf = df_results['original_pf'].mean()
    avg_retrained_pf = df_results['retrained_pf'].mean()

    months_better = (df_results['r_improvement'] > 0).sum()
    months_worse = (df_results['r_improvement'] < 0).sum()
    months_similar = (df_results['r_improvement'] == 0).sum()

    print(f"\nOriginal Model (2020-2023 training):")
    print(f"  Total R (2025): {total_orig_r:+.1f}R")
    print(f"  Avg Win Rate: {avg_orig_wr:.1%}")
    print(f"  Avg Profit Factor: {avg_orig_pf:.2f}")

    print(f"\nRetrained Model (expanding window):")
    print(f"  Total R (2025): {total_retrained_r:+.1f}R")
    print(f"  Avg Win Rate: {avg_retrained_wr:.1%}")
    print(f"  Avg Profit Factor: {avg_retrained_pf:.2f}")

    print(f"\nImprovement:")
    print(f"  Total R: {total_improvement:+.1f}R ({total_improvement/abs(total_orig_r)*100:+.1f}%)")
    print(f"  Win Rate: {avg_retrained_wr - avg_orig_wr:+.1%}")
    print(f"  Profit Factor: {avg_retrained_pf - avg_orig_pf:+.2f}")

    print(f"\nMonths Comparison:")
    print(f"  Better: {months_better}/{len(df_results)}")
    print(f"  Worse: {months_worse}/{len(df_results)}")
    print(f"  Similar: {months_similar}/{len(df_results)}")

    # Recommendation
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)

    improvement_pct = (total_improvement / abs(total_orig_r)) * 100 if total_orig_r != 0 else 0

    print()
    if improvement_pct > 10 and months_better > months_worse * 1.5:
        print("🟢 RECOMMEND RETRAINING")
        print(f"   Retrained model shows {improvement_pct:+.1f}% improvement")
        print(f"   Better performance in {months_better}/{len(df_results)} months")
        print(f"   → Use expanding-window retraining going forward")
        print(f"   → Retrain quarterly with latest data")
    elif improvement_pct > 5:
        print("🟡 MARGINAL BENEFIT")
        print(f"   Retrained model shows {improvement_pct:+.1f}% improvement")
        print(f"   Better in {months_better} months, worse in {months_worse}")
        print(f"   → Consider retraining if performance degrades")
        print(f"   → Monitor original model closely")
    else:
        print("🔴 NO SIGNIFICANT BENEFIT")
        print(f"   Retrained model improvement: {improvement_pct:+.1f}%")
        print(f"   Original model generalizes well")
        print(f"   → Continue using original model")
        print(f"   → No need for frequent retraining")

    # Save results
    print("\n" + "=" * 100)
    print("SAVING RESULTS")
    print("=" * 100)

    df_results.to_csv(RESULTS_DIR / "walk_forward_comparison.csv", index=False)
    print(f"\n✓ Results saved to: {RESULTS_DIR / 'walk_forward_comparison.csv'}")

    if months_better > 0:
        print(f"✓ {months_better} improved models saved to: {MODELS_RETRAINED_DIR}/")

    # Final summary
    summary = {
        'evaluation_date': datetime.now().isoformat(),
        'original_model_training': '2020-2023',
        'test_period': '2025',
        'months_evaluated': len(df_results),
        'original_total_r': float(total_orig_r),
        'retrained_total_r': float(total_retrained_r),
        'improvement_pct': float(improvement_pct),
        'months_better': int(months_better),
        'months_worse': int(months_worse),
        'avg_original_wr': float(avg_orig_wr),
        'avg_retrained_wr': float(avg_retrained_wr),
        'recommendation': 'RETRAIN' if improvement_pct > 10 else 'MONITOR' if improvement_pct > 5 else 'KEEP_ORIGINAL'
    }

    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "retraining_summary.csv", index=False)
    print(f"✓ Summary saved to: {RESULTS_DIR / 'retraining_summary.csv'}")

    print("\n" + "=" * 100)
    print("WALK-FORWARD EVALUATION COMPLETE")
    print("=" * 100)
    print()


if __name__ == "__main__":
    main()
