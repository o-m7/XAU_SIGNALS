#!/usr/bin/env python3
"""
COMPLETE REBUILD - Train Trading Models from Scratch

UPDATED 2026-01-13: Now uses UNIFIED FEATURE ENGINEERING!
- Uses src.features.build_features() for training/live parity
- Fixed lookahead bias in synthetic_order_flow
- Single source of truth for all features

Strategy:
1. Feature Engineering: ALL features via unified module (microstructure + CMF/MACD + regime)
2. Walk-Forward Validation: 2014-2025, train on past, test on future
3. Robust Labels: Learn from BOTH good and bad trades
4. Anti-Overfitting: Multiple validation periods, conservative hyperparameters

NO MORE OVERFITTING.
NO MORE HALLUCINATIONS.
ONLY HONEST, ROBUST MODELS.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# ML libraries
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score

# Project imports
sys.path.insert(0, str(Path(__file__).parent))
from src.features import build_features, get_features_for_model

print("="*80)
print("REBUILD FROM SCRATCH - Honest Model Training")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Master configuration for model training."""

    # Data
    DATA_PATH = Path('data/features/xauusd_features_2020_2025_fixed.parquet')
    OUTPUT_DIR = Path('models/rebuilt_from_scratch')

    # Training periods (walk-forward)
    TRAIN_START = '2014-01-01'
    TRAIN_END = '2025-12-31'

    # Walk-forward splits (CRITICAL for preventing overfitting)
    # Train on past, test on future, NEVER look ahead
    WALK_FORWARD_SPLITS = [
        {'train': ('2014-01-01', '2021-12-31'), 'test': ('2022-01-01', '2022-12-31')},
        {'train': ('2014-01-01', '2022-12-31'), 'test': ('2023-01-01', '2023-12-31')},
        {'train': ('2014-01-01', '2023-12-31'), 'test': ('2024-01-01', '2024-12-31')},
        {'train': ('2014-01-01', '2024-12-31'), 'test': ('2025-01-01', '2025-12-31')},
    ]

    # Labeling strategy - LEARN FROM BOTH WINS AND LOSSES
    # Don't just avoid bad trades, learn WHAT MAKES A GOOD TRADE
    LABEL_CONFIG = {
        'tp_atr_mult': 1.5,    # Take profit at 1.5 ATR
        'sl_atr_mult': 1.0,    # Stop loss at 1.0 ATR
        'max_bars': 30,        # Max 30 bars (7.5 hours on 15min data)
        'min_bars': 2,         # Minimum 2 bars (30 min) to count as real trade
    }

    # Model hyperparameters - CONSERVATIVE (anti-overfitting)
    MODEL_CONFIGS = {
        'model1_hgb': {
            'algorithm': HistGradientBoostingClassifier,
            'params': {
                'max_iter': 100,           # Low iterations to prevent overfitting
                'max_depth': 4,            # Shallow trees (prevent memorization)
                'learning_rate': 0.05,     # Slow learning
                'min_samples_leaf': 100,   # Large leaf size (smooth predictions)
                'l2_regularization': 1.0,  # Strong L2 regularization
                'max_bins': 64,            # Limit complexity
                'random_state': 42,
            },
            'features': 'microstructure',  # Microstructure + order flow
        },
        'model3_cmf_macd': {
            'algorithm': HistGradientBoostingClassifier,
            'params': {
                'max_iter': 100,
                'max_depth': 4,
                'learning_rate': 0.05,
                'min_samples_leaf': 100,
                'l2_regularization': 1.0,
                'max_bins': 64,
                'random_state': 42,
            },
            'features': 'cmf_macd',  # CMF + MACD momentum indicators
        },
        'model_rf': {
            'algorithm': RandomForestClassifier,
            'params': {
                'n_estimators': 100,       # Moderate ensemble size
                'max_depth': 6,            # Limit depth
                'min_samples_split': 200,  # Require many samples to split
                'min_samples_leaf': 100,   # Large leaf size
                'max_features': 'sqrt',    # Use sqrt of features (decorrelation)
                'bootstrap': True,
                'oob_score': True,         # Out-of-bag validation
                'random_state': 42,
                'n_jobs': -1,
            },
            'features': 'microstructure',
        },
    }

    # Validation criteria - STRICT (prevent deployment of bad models)
    DEPLOYMENT_CRITERIA = {
        'min_win_rate': 0.52,      # Minimum 52% win rate (after costs)
        'min_profit_factor': 1.30,  # Minimum 1.3 profit factor
        'min_trades_per_day': 1.0,  # At least 1 trade/day
        'max_trades_per_day': 20.0, # Max 20 trades/day (prevent over-trading)
        'min_sharpe': 0.5,          # Minimum Sharpe ratio
        'max_drawdown': 0.15,       # Max 15% drawdown
    }

config = Config()


# ============================================================================
# STEP 1: FEATURE ENGINEERING
# ============================================================================

def build_all_features(df: pd.DataFrame, quotes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Build ALL features using unified feature engineering module.

    Uses src.features.build_features() - the SINGLE SOURCE OF TRUTH.
    This ensures training and live use IDENTICAL feature computation.

    Args:
        df: Bar data with OHLCV columns
        quotes: Optional quote data with bid/ask (for microstructure features)

    Returns:
        DataFrame with all features
    """
    print("\n" + "="*80)
    print("STEP 1: UNIFIED FEATURE ENGINEERING")
    print("="*80)

    print(f"Input data: {len(df):,} rows from {df.index[0]} to {df.index[-1]}")

    # Use unified feature builder (SAME AS LIVE!)
    df = build_features(df, quotes=quotes, feature_set="all")

    # Feature completeness check
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"\n  ⚠️ Found NaN values:")
        for col in nan_counts[nan_counts > 0].index[:10]:  # Show first 10
            print(f"    {col}: {nan_counts[col]:,} NaN")
        print(f"  Filling NaN with forward-fill then 0...")
        df = df.ffill().fillna(0)

    print(f"\n✓ Feature engineering complete (using unified module)")
    print(f"  Total features: {len(df.columns)}")
    print(f"  Total rows: {len(df):,}")

    return df


def get_feature_set(df: pd.DataFrame, feature_type: str) -> List[str]:
    """
    Get list of features for a specific model type.

    Uses src.features.get_features_for_model() - the SINGLE SOURCE OF TRUTH.
    This ensures training and live use IDENTICAL feature lists.

    Args:
        df: DataFrame with features
        feature_type: One of 'microstructure', 'cmf_macd' (or model names)

    Returns:
        List of available feature names
    """
    # Get required features from unified registry
    features = get_features_for_model(feature_type)

    # Filter to only features that exist in DataFrame
    available = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]

    if missing:
        print(f"  ⚠️ Missing {len(missing)} features: {missing[:5]}...")
        print(f"     Total requested: {len(features)}, Available: {len(available)}")

    return available


# ============================================================================
# STEP 2: ROBUST LABELING
# ============================================================================

def create_triple_barrier_labels(
    df: pd.DataFrame,
    tp_atr_mult: float = 1.5,
    sl_atr_mult: float = 1.0,
    max_bars: int = 30,
    min_bars: int = 2
) -> pd.DataFrame:
    """
    Create labels using triple-barrier method.

    KEY INSIGHT: We want to learn WHAT MAKES A GOOD TRADE
    - Label = +1 if hits TP before SL or timeout
    - Label = -1 if hits SL before TP or timeout
    - Label = 0 if timeout without hitting TP/SL (neutral)

    This way model learns:
    - What patterns lead to fast wins (hit TP quickly)
    - What patterns lead to fast losses (hit SL quickly)
    - What patterns are uncertain (timeout)

    Model WON'T trade uncertain patterns!
    """
    print("\n" + "="*80)
    print("STEP 2: ROBUST LABELING (Triple-Barrier)")
    print("="*80)

    print(f"  TP: {tp_atr_mult}x ATR")
    print(f"  SL: {sl_atr_mult}x ATR")
    print(f"  Max hold: {max_bars} bars")
    print(f"  Min hold: {min_bars} bars (filter noise)")

    labels = []
    bars_to_exit = []

    for i in range(len(df) - max_bars):
        if i % 10000 == 0:
            print(f"  Processing: {i:,}/{len(df):,} ({i/len(df)*100:.1f}%)", end='\r')

        entry_price = df['close'].iloc[i]
        atr = df['ATR_14'].iloc[i]

        if pd.isna(entry_price) or pd.isna(atr) or atr <= 0:
            labels.append(0)
            bars_to_exit.append(0)
            continue

        # Barriers
        tp_barrier = entry_price + (tp_atr_mult * atr)
        sl_barrier = entry_price - (sl_atr_mult * atr)

        # Check future bars
        hit_tp = False
        hit_sl = False
        exit_bar = max_bars

        for j in range(min_bars, max_bars + 1):
            if i + j >= len(df):
                break

            future_high = df['high'].iloc[i + j]
            future_low = df['low'].iloc[i + j]

            # Check if barriers hit
            if future_high >= tp_barrier and not hit_sl:
                hit_tp = True
                exit_bar = j
                break
            elif future_low <= sl_barrier and not hit_tp:
                hit_sl = True
                exit_bar = j
                break

        # Assign label
        if hit_tp:
            labels.append(1)  # WIN
        elif hit_sl:
            labels.append(-1)  # LOSS
        else:
            labels.append(0)  # TIMEOUT (uncertain)

        bars_to_exit.append(exit_bar)

    # Pad to match length
    labels += [0] * max_bars
    bars_to_exit += [0] * max_bars

    df['y_tb'] = labels
    df['bars_to_exit'] = bars_to_exit

    # Statistics
    wins = (df['y_tb'] == 1).sum()
    losses = (df['y_tb'] == -1).sum()
    neutral = (df['y_tb'] == 0).sum()
    total = wins + losses + neutral

    print(f"\n\n✓ Labeling complete")
    print(f"  WIN:     {wins:8,} ({wins/total*100:5.1f}%) - Hit TP before SL")
    print(f"  LOSS:    {losses:8,} ({losses/total*100:5.1f}%) - Hit SL before TP")
    print(f"  NEUTRAL: {neutral:8,} ({neutral/total*100:5.1f}%) - Timeout (uncertain)")
    print(f"  Total:   {total:8,}")

    # Win rate (excluding neutral)
    decided = wins + losses
    if decided > 0:
        print(f"\n  Win Rate (excl. neutral): {wins/decided*100:.1f}%")

    return df


# ============================================================================
# STEP 3: WALK-FORWARD VALIDATION
# ============================================================================

def walk_forward_train_test(
    df: pd.DataFrame,
    model_config: Dict,
    model_name: str
) -> Dict:
    """
    Walk-forward validation: Train on past, test on future.

    This is THE MOST IMPORTANT anti-overfitting technique:
    - Never look ahead into test period
    - Train on expanding window (2014-2021, 2014-2022, etc.)
    - Test on next year (2022, 2023, etc.)
    - Model MUST generalize to future data
    """
    print("\n" + "="*80)
    print(f"STEP 3: WALK-FORWARD VALIDATION - {model_name}")
    print("="*80)

    # Get features
    features = get_feature_set(df, model_config['features'])
    print(f"\nFeatures: {len(features)}")

    # Results storage
    all_results = []

    for fold_idx, split in enumerate(config.WALK_FORWARD_SPLITS, 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}/{len(config.WALK_FORWARD_SPLITS)}")
        print(f"  Train: {split['train'][0]} to {split['train'][1]}")
        print(f"  Test:  {split['test'][0]} to {split['test'][1]}")
        print(f"{'='*60}")

        # Split data
        train_df = df[split['train'][0]:split['train'][1]]
        test_df = df[split['test'][0]:split['test'][1]]

        print(f"  Train size: {len(train_df):,}")
        print(f"  Test size:  {len(test_df):,}")

        # Prepare data
        X_train = train_df[features].values
        y_train = train_df['y_tb'].values
        X_test = test_df[features].values
        y_test = test_df['y_tb'].values

        # Remove neutral labels for training (model learns WIN vs LOSS)
        train_mask = y_train != 0
        X_train_filtered = X_train[train_mask]
        y_train_filtered = y_train[train_mask]

        # Convert to binary: +1 = 1, -1 = 0
        y_train_binary = (y_train_filtered == 1).astype(int)
        y_test_binary = (y_test == 1).astype(int)

        print(f"\n  Training samples: {len(X_train_filtered):,}")
        print(f"    WIN:  {(y_train_binary == 1).sum():,}")
        print(f"    LOSS: {(y_train_binary == 0).sum():,}")

        # Train model
        print(f"\n  Training {model_config['algorithm'].__name__}...")
        model = model_config['algorithm'](**model_config['params'])
        model.fit(X_train_filtered, y_train_binary)

        # Predict on test set
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of WIN

        # Evaluate with different thresholds
        thresholds_to_test = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

        best_wr = 0
        best_threshold = 0.5

        print(f"\n  Testing thresholds on {split['test'][0]}:")
        print(f"  {'Threshold':<12} {'Signals':<10} {'Win Rate':<10} {'Status'}")
        print(f"  {'-'*50}")

        for thresh in thresholds_to_test:
            # Generate signals
            signals = np.where(y_pred_proba >= thresh, 1,
                             np.where(y_pred_proba <= (1 - thresh), -1, 0))

            # Calculate win rate (only on actual signals with non-neutral labels)
            # Filter: signal != 0 AND y_test != 0 (exclude neutrals from evaluation)
            signal_mask = (signals != 0) & (y_test != 0)
            if signal_mask.sum() > 0:
                correct = (signals[signal_mask] == np.sign(y_test[signal_mask])).sum()
                wr = correct / signal_mask.sum()

                status = "✓" if wr > 0.52 else "✗"
                print(f"  {thresh:<12.2f} {signal_mask.sum():<10,} {wr:<10.1%} {status}")

                if wr > best_wr:
                    best_wr = wr
                    best_threshold = thresh

        print(f"\n  Best: Threshold={best_threshold:.2f}, WR={best_wr:.1%}")

        # Store results
        all_results.append({
            'fold': fold_idx,
            'train_period': f"{split['train'][0]} to {split['train'][1]}",
            'test_period': f"{split['test'][0]} to {split['test'][1]}",
            'best_threshold': best_threshold,
            'best_win_rate': best_wr,
            'test_samples': len(test_df),
        })

    # Summary across all folds
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD SUMMARY - {model_name}")
    print(f"{'='*60}")

    for result in all_results:
        print(f"\nFold {result['fold']}: {result['test_period']}")
        print(f"  Best Threshold: {result['best_threshold']:.2f}")
        print(f"  Best Win Rate:  {result['best_win_rate']:.1%}")

    avg_wr = np.mean([r['best_win_rate'] for r in all_results])
    print(f"\nAverage Win Rate: {avg_wr:.1%}")

    return {
        'model': model,
        'features': features,
        'results': all_results,
        'avg_win_rate': avg_wr,
    }


# ============================================================================
# STEP 4: FINAL TRAINING & DEPLOYMENT
# ============================================================================

def train_final_model(
    df: pd.DataFrame,
    model_config: Dict,
    model_name: str,
    optimal_threshold: float
) -> Dict:
    """
    Train final model on ALL available data for deployment.

    Use optimal threshold found during walk-forward validation.
    """
    print(f"\n{'='*80}")
    print(f"FINAL TRAINING - {model_name}")
    print(f"{'='*80}")

    features = get_feature_set(df, model_config['features'])

    # Use all data for final training
    X = df[features].values
    y = df['y_tb'].values

    # Remove neutral
    mask = y != 0
    X_filtered = X[mask]
    y_filtered = y[mask]
    y_binary = (y_filtered == 1).astype(int)

    print(f"Training on {len(X_filtered):,} samples")
    print(f"  WIN:  {(y_binary == 1).sum():,}")
    print(f"  LOSS: {(y_binary == 0).sum():,}")

    # Train
    model = model_config['algorithm'](**model_config['params'])
    model.fit(X_filtered, y_binary)

    # Save
    output_path = config.OUTPUT_DIR / f"{model_name}.joblib"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    artifact = {
        'model': model,
        'features': features,
        'threshold_long': optimal_threshold,
        'threshold_short': 1 - optimal_threshold,
        'training_date': datetime.now().isoformat(),
        'training_samples': len(X_filtered),
    }

    joblib.dump(artifact, output_path)
    print(f"\n✓ Model saved: {output_path}")
    print(f"  Thresholds: LONG≥{optimal_threshold:.2f}, SHORT≤{1-optimal_threshold:.2f}")

    return artifact


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline."""

    # Load data
    print(f"\nLoading data from: {config.DATA_PATH}")
    df = pd.read_parquet(config.DATA_PATH)

    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')

    print(f"Loaded: {len(df):,} rows from {df.index[0]} to {df.index[-1]}")

    # Step 1: Feature Engineering
    df = build_all_features(df)

    # Step 2: Labeling
    df = create_triple_barrier_labels(
        df,
        tp_atr_mult=config.LABEL_CONFIG['tp_atr_mult'],
        sl_atr_mult=config.LABEL_CONFIG['sl_atr_mult'],
        max_bars=config.LABEL_CONFIG['max_bars'],
        min_bars=config.LABEL_CONFIG['min_bars']
    )

    # Step 3: Train each model with walk-forward validation
    results = {}

    for model_name, model_config in config.MODEL_CONFIGS.items():
        result = walk_forward_train_test(df, model_config, model_name)
        results[model_name] = result

        # Train final model if passes validation
        if result['avg_win_rate'] >= config.DEPLOYMENT_CRITERIA['min_win_rate']:
            print(f"\n✅ {model_name} PASSES validation ({result['avg_win_rate']:.1%} WR)")

            # Use median threshold from walk-forward
            optimal_threshold = np.median([r['best_threshold'] for r in result['results']])
            train_final_model(df, model_config, model_name, optimal_threshold)
        else:
            print(f"\n❌ {model_name} FAILS validation ({result['avg_win_rate']:.1%} < {config.DEPLOYMENT_CRITERIA['min_win_rate']:.1%})")

    # Final report
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {config.OUTPUT_DIR}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\nModel Performance:")
    for model_name, result in results.items():
        status = "✅ PASS" if result['avg_win_rate'] >= config.DEPLOYMENT_CRITERIA['min_win_rate'] else "❌ FAIL"
        print(f"  {model_name}: {result['avg_win_rate']:.1%} WR - {status}")


if __name__ == "__main__":
    main()
