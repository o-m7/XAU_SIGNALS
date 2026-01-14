#!/usr/bin/env python3
"""
Model 8: Momentum LONG-Only

Strategy: Carhart Momentum Factor adapted for intraday XAUUSD
ONLY PREDICTS LONGS - No short signals

Mathematical Foundation:
- Momentum Factor: Winners tend to keep winning (short-term)
- Entry when:
  1. momentum_score > 1.96 (95% statistically significant uptrend)
  2. close > VWAP (price above fair value)
  3. volume_ratio > 1.0 (above average volume confirms)

This is a proven momentum strategy based on academic research.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.labels_v2 import create_validated_labels

# Paths
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "model8_momentum_long.joblib"

# Config
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-06-30"
VAL_START = "2023-07-01"
VAL_END = "2024-06-30"
TEST_START = "2024-07-01"
TEST_END = "2025-12-31"

LABEL_HORIZON = 30  # 30 minutes


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (intraday).
    
    VWAP = sum(Price * Volume) / sum(Volume)
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cumulative_pv = (typical_price * df['volume']).rolling(30).sum()
    cumulative_vol = df['volume'].rolling(30).sum()
    
    return cumulative_pv / (cumulative_vol + 1e-10)


def calculate_momentum_zscore(series: pd.Series, lookback: int = 30) -> pd.Series:
    """
    Calculate momentum z-score (statistically standardized return).
    
    z = (P_t - P_{t-n}) / (sigma * sqrt(n))
    
    |z| > 1.96 means 95% confidence the trend is real
    """
    price_change = series - series.shift(lookback)
    rolling_std = series.pct_change().rolling(lookback).std() * series
    
    zscore = price_change / (rolling_std * np.sqrt(lookback) + 1e-10)
    
    return zscore


def build_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build momentum-focused features for LONG-only signals."""
    df = df.copy()
    
    print("  Building momentum features (LONG-only focus)...")
    
    # ========================
    # 1. MOMENTUM Z-SCORES
    # ========================
    for period in [10, 20, 30, 60]:
        df[f'momentum_z_{period}'] = calculate_momentum_zscore(df['close'], period)
    
    # Statistically significant momentum (|z| > 1.96)
    df['momentum_significant_10'] = (df['momentum_z_10'] > 1.96).astype(int)
    df['momentum_significant_30'] = (df['momentum_z_30'] > 1.96).astype(int)
    
    # Strong momentum (|z| > 2.58, 99% confidence)
    df['momentum_strong_30'] = (df['momentum_z_30'] > 2.58).astype(int)
    
    # ========================
    # 2. VWAP
    # ========================
    df['vwap'] = calculate_vwap(df)
    df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap'] * 100
    
    # Above VWAP (bullish)
    df['above_vwap'] = (df['close'] > df['vwap']).astype(int)
    df['far_above_vwap'] = (df['vwap_deviation'] > 0.1).astype(int)  # 0.1% above
    
    # ========================
    # 3. VOLUME CONFIRMATION
    # ========================
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1e-10)
    
    df['volume_above_avg'] = (df['volume_ratio'] > 1.0).astype(int)
    df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)
    
    # On-balance volume trend
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    df['obv_trend'] = (df['obv'] > df['obv'].rolling(20).mean()).astype(int)
    
    # ========================
    # 4. PRICE TREND CONFIRMATION
    # ========================
    # Higher highs and higher lows (uptrend structure)
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
    
    df['uptrend_bars_5'] = (df['higher_high'] + df['higher_low']).rolling(5).sum()
    
    # Price above moving averages
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_30'] = df['close'].rolling(30).mean()
    
    df['above_sma_10'] = (df['close'] > df['sma_10']).astype(int)
    df['above_sma_30'] = (df['close'] > df['sma_30']).astype(int)
    df['sma_10_above_30'] = (df['sma_10'] > df['sma_30']).astype(int)  # Bullish crossover
    
    # ========================
    # 5. ACCELERATION
    # ========================
    # Momentum of momentum (is trend strengthening?)
    df['momentum_accel'] = df['momentum_z_30'] - df['momentum_z_30'].shift(5)
    df['accelerating'] = (df['momentum_accel'] > 0).astype(int)
    
    # ========================
    # 6. RISK METRICS
    # ========================
    # Volatility (avoid entering during extreme vol)
    df['volatility_30'] = df['close'].pct_change().rolling(30).std() * 100
    df['volatility_percentile'] = df['volatility_30'].rolling(200).rank(pct=True)
    df['normal_volatility'] = ((df['volatility_percentile'] > 0.2) & 
                               (df['volatility_percentile'] < 0.8)).astype(int)
    
    # ========================
    # 7. COMBINED ENTRY SIGNALS
    # ========================
    # Primary entry signal (all conditions)
    df['entry_signal_primary'] = (
        (df['momentum_significant_30'] == 1) &
        (df['above_vwap'] == 1) &
        (df['volume_above_avg'] == 1)
    ).astype(int)
    
    # Strong entry signal (stricter)
    df['entry_signal_strong'] = (
        (df['momentum_strong_30'] == 1) &
        (df['far_above_vwap'] == 1) &
        (df['volume_spike'] == 1) &
        (df['accelerating'] == 1)
    ).astype(int)
    
    # Conservative entry
    df['entry_signal_conservative'] = (
        (df['momentum_significant_30'] == 1) &
        (df['above_vwap'] == 1) &
        (df['volume_above_avg'] == 1) &
        (df['above_sma_30'] == 1) &
        (df['normal_volatility'] == 1)
    ).astype(int)
    
    # Simple returns for context
    for period in [1, 5, 15, 30]:
        df[f'return_{period}'] = df['close'].pct_change(period)
    
    return df


def main():
    print("=" * 80)
    print("MODEL 8: MOMENTUM LONG-ONLY")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now()}")
    print(f"Train: {TRAIN_START} to {TRAIN_END}")
    print(f"Validation: {VAL_START} to {VAL_END}")
    print(f"Test: {TEST_START} to {TEST_END}")
    
    # Load data
    print("\n[1] Loading data...")
    df = pd.read_parquet(FEATURES_PATH)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    print(f"  Loaded {len(df):,} rows")
    
    # Build features
    print("\n[2] Building momentum features...")
    df = build_momentum_features(df)
    
    # Create LONG-ONLY labels (only positive price movements)
    print("\n[3] Creating LONG-ONLY labels...")
    
    # Calculate future return
    future_return = df['close'].shift(-LABEL_HORIZON) / df['close'] - 1
    
    # LONG-only: 1 = price goes UP, 0 = price stays flat or goes DOWN
    # We want to predict if a LONG trade will be profitable
    min_profit = 0.001  # Minimum 0.1% move to count as "up"
    df['label'] = (future_return > min_profit).astype(int)
    
    # Distribution check
    label_dist = df['label'].value_counts(normalize=True)
    print(f"  Label distribution:")
    print(f"    UP (1):   {label_dist.get(1, 0):.1%}")
    print(f"    DOWN (0): {label_dist.get(0, 0):.1%}")
    
    # Split data
    print("\n[4] Splitting data...")
    train_df = df[TRAIN_START:TRAIN_END].copy()
    val_df = df[VAL_START:VAL_END].copy()
    test_df = df[TEST_START:TEST_END].copy()
    
    print(f"  Train: {len(train_df):,}")
    print(f"  Validation: {len(val_df):,}")
    print(f"  Test: {len(test_df):,}")
    
    # Select features
    momentum_features = [
        # Momentum z-scores
        'momentum_z_10', 'momentum_z_20', 'momentum_z_30', 'momentum_z_60',
        'momentum_significant_10', 'momentum_significant_30', 'momentum_strong_30',
        
        # VWAP
        'vwap_deviation', 'above_vwap', 'far_above_vwap',
        
        # Volume
        'volume_ratio', 'volume_above_avg', 'volume_spike', 'obv_trend',
        
        # Trend
        'higher_high', 'higher_low', 'uptrend_bars_5',
        'above_sma_10', 'above_sma_30', 'sma_10_above_30',
        
        # Acceleration
        'momentum_accel', 'accelerating',
        
        # Risk
        'volatility_30', 'volatility_percentile', 'normal_volatility',
        
        # Combined signals
        'entry_signal_primary', 'entry_signal_strong', 'entry_signal_conservative',
        
        # Returns
        'return_1', 'return_5', 'return_15', 'return_30'
    ]
    
    # Filter to existing columns
    feature_cols = [f for f in momentum_features if f in train_df.columns]
    print(f"\n[5] Using {len(feature_cols)} features")
    
    # Prepare training data
    train_clean = train_df.dropna(subset=['label'] + feature_cols)
    
    X_train = train_clean[feature_cols].values
    y_train = train_clean['label'].values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    val_clean = val_df.dropna(subset=['label'] + feature_cols)
    
    X_val = val_clean[feature_cols].values
    y_val = val_clean['label'].values
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Train samples: {len(X_train):,} (up={y_train.mean():.1%})")
    print(f"  Val samples: {len(X_val):,}")
    
    # Train model (Neural Network)
    print("\n[6] Training Neural Network (MLP) model...")
    
    # Scale features for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 regularization
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=False
    )
    model.fit(X_train_scaled, y_train)
    
    # Store scaler with model
    model.scaler_ = scaler
    X_val = X_val_scaled  # Use scaled data for evaluation
    
    # Evaluate
    print("\n[7] Evaluation...")
    
    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]
    
    print(f"\n  Validation Results:")
    print(f"    Accuracy:  {accuracy_score(y_val, val_pred):.4f}")
    print(f"    ROC-AUC:   {roc_auc_score(y_val, val_proba):.4f}")
    print(f"    Precision: {precision_score(y_val, val_pred):.4f}")
    print(f"    Recall:    {recall_score(y_val, val_pred):.4f}")
    
    # Signal distribution with HIGH threshold (for high win rate)
    threshold = 0.60  # Higher threshold for LONG-only
    signals_long = (val_proba >= threshold).sum()
    print(f"\n  LONG signals (threshold={threshold}):")
    print(f"    Count: {signals_long:,} ({signals_long/len(val_proba):.1%} of bars)")
    
    # Win rate of high-confidence signals
    high_conf_mask = val_proba >= threshold
    if high_conf_mask.sum() > 0:
        high_conf_win_rate = y_val[high_conf_mask].mean()
        print(f"    Expected Win Rate: {high_conf_win_rate:.1%}")
    
    # Feature importance
    print("\n[8] Top features:")
    if hasattr(model, 'feature_importances_'):
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for _, row in importances.head(10).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")
    else:
        print("    (Feature importances not available for this model type)")
    
    # Save model
    print(f"\n[9] Saving model to {MODEL_OUTPUT_PATH}")
    artifact = {
        'model': model,
        'model_type': 'NeuralNetwork',
        'scaler': scaler,  # Store scaler for inference
        'features': feature_cols,
        'label_horizon': LABEL_HORIZON,
        'threshold': threshold,
        'strategy': 'momentum_long_only',
        'direction': 'LONG_ONLY',
        'math_basis': 'carhart_momentum_factor',
        'train_period': f"{TRAIN_START} to {TRAIN_END}",
        'val_auc': roc_auc_score(y_val, val_proba),
        'saved_at': datetime.now().isoformat()
    }
    
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    print("  Saved!")
    
    print("\n" + "=" * 80)
    print("MODEL 8 TRAINING COMPLETE (LONG-ONLY)")
    print("=" * 80)


if __name__ == "__main__":
    main()

