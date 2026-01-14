#!/usr/bin/env python3
"""
Model 9: Liquidity Rejection SHORT-Only

Strategy: Price rejection at liquidity/resistance levels
ONLY PREDICTS SHORTS - No long signals

Mathematical Foundation:
- Liquidity Levels: Areas where price previously reversed
- Rejection: Price touches level but fails to break through
- Entry when:
  1. Price touched resistance in last 5 bars
  2. rejection_score > 0.7 (strong rejection candle pattern)
  3. volume_spike = True (volume > 1.5x average, institutions present)

This exploits the tendency of prices to reverse at key levels.
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
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "model9_rejection_short.joblib"

# Config
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-06-30"
VAL_START = "2023-07-01"
VAL_END = "2024-06-30"
TEST_START = "2024-07-01"
TEST_END = "2025-12-31"

LABEL_HORIZON = 30  # 30 minutes


def identify_resistance_levels(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """
    Identify recent resistance levels (rolling highs).
    """
    # Use rolling max as resistance proxy
    return df['high'].rolling(lookback).max()


def identify_support_levels(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """
    Identify recent support levels (rolling lows).
    """
    return df['low'].rolling(lookback).min()


def calculate_rejection_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate rejection candle score (0 to 1).
    
    A strong bearish rejection:
    - Price went up (touched high) but closed low
    - Long upper wick, small body, close near low
    - Higher score = stronger rejection
    """
    range_size = df['high'] - df['low']
    
    # Upper wick (rejection from above)
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    
    # Body at bottom of range
    body_position = (df['close'] - df['low']) / (range_size + 1e-10)  # 0 = at low, 1 = at high
    
    # Upper wick ratio (larger = more rejection)
    upper_wick_ratio = upper_wick / (range_size + 1e-10)
    
    # Rejection score combines:
    # 1. Large upper wick (0.5 weight)
    # 2. Close near the low (0.5 weight)
    rejection = upper_wick_ratio * 0.5 + (1 - body_position) * 0.5
    
    # Bearish candle bonus
    bearish = (df['close'] < df['open']).astype(float) * 0.2
    
    return (rejection + bearish).clip(0, 1)


def build_rejection_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build rejection-focused features for SHORT-only signals."""
    df = df.copy()
    
    print("  Building rejection features (SHORT-only focus)...")
    
    # ========================
    # 1. RESISTANCE LEVELS
    # ========================
    for lookback in [20, 50, 100]:
        df[f'resistance_{lookback}'] = identify_resistance_levels(df, lookback)
        df[f'support_{lookback}'] = identify_support_levels(df, lookback)
    
    # Distance to resistance (negative = at/above resistance)
    df['dist_to_resistance'] = (df['resistance_50'] - df['high']) / df['close'] * 100
    df['dist_to_support'] = (df['low'] - df['support_50']) / df['close'] * 100
    
    # At resistance (within 0.1% of level)
    df['at_resistance'] = (abs(df['dist_to_resistance']) < 0.1).astype(int)
    df['touched_resistance_5'] = (df['at_resistance'].rolling(5).sum() > 0).astype(int)
    
    # Broke resistance then failed
    df['new_high'] = (df['high'] > df['high'].shift(1).rolling(20).max()).astype(int)
    df['failed_breakout'] = ((df['new_high'] == 1) & 
                              (df['close'] < df['open'])).astype(int)
    
    # ========================
    # 2. REJECTION SCORE
    # ========================
    df['rejection_score'] = calculate_rejection_score(df)
    
    # Strong rejection (threshold > 0.7)
    df['strong_rejection'] = (df['rejection_score'] > 0.7).astype(int)
    df['very_strong_rejection'] = (df['rejection_score'] > 0.8).astype(int)
    
    # Consecutive rejections (bearish)
    df['rejection_streak'] = df['strong_rejection'].rolling(3).sum()
    
    # ========================
    # 3. VOLUME ANALYSIS
    # ========================
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
    
    df['volume_above_avg'] = (df['volume_ratio'] > 1.0).astype(int)
    df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)
    df['volume_climax'] = (df['volume_ratio'] > 2.0).astype(int)  # Potential exhaustion
    
    # ========================
    # 4. MOMENTUM EXHAUSTION
    # ========================
    # Price momentum slowing
    df['momentum_10'] = df['close'].pct_change(10)
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_slowing'] = ((df['momentum_10'] > 0) & 
                               (df['momentum_5'] < df['momentum_10'] / 2)).astype(int)
    
    # Price extended (far from mean)
    df['mean_50'] = df['close'].rolling(50).mean()
    df['std_50'] = df['close'].rolling(50).std()
    df['zscore_50'] = (df['close'] - df['mean_50']) / (df['std_50'] + 1e-10)
    
    df['overbought_zscore'] = (df['zscore_50'] > 2.0).astype(int)
    df['extremely_overbought'] = (df['zscore_50'] > 2.5).astype(int)
    
    # ========================
    # 5. CANDLE PATTERNS
    # ========================
    # Shooting star pattern
    range_size = df['high'] - df['low']
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    body = abs(df['close'] - df['open'])
    
    df['shooting_star'] = ((upper_wick > 2 * body) & 
                            (df['close'] < df['open'])).astype(int)
    
    # Bearish engulfing
    prev_body = abs(df['close'].shift(1) - df['open'].shift(1))
    bearish_engulf = ((body > prev_body * 1.5) & 
                      (df['close'] < df['open']) & 
                      (df['close'].shift(1) > df['open'].shift(1)))
    df['bearish_engulfing'] = bearish_engulf.astype(int)
    
    # Evening star (simplified)
    df['evening_star'] = ((df['close'].shift(2) > df['open'].shift(2)) &  # First: bullish
                          (abs(df['close'].shift(1) - df['open'].shift(1)) < body.shift(1) * 0.3) &  # Second: doji
                          (df['close'] < df['open'])).astype(int)  # Third: bearish
    
    # ========================
    # 6. TREND REVERSAL SIGNALS
    # ========================
    # Lower high (bearish)
    df['lower_high'] = (df['high'] < df['high'].shift(1)).astype(int)
    df['lower_close'] = (df['close'] < df['close'].shift(1)).astype(int)
    
    df['downtrend_bars_3'] = df['lower_close'].rolling(3).sum()
    
    # Price below VWAP (bearish)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).rolling(30).sum() / df['volume'].rolling(30).sum()
    df['below_vwap'] = (df['close'] < df['vwap']).astype(int)
    
    # ========================
    # 7. RISK METRICS
    # ========================
    df['volatility_30'] = df['close'].pct_change().rolling(30).std() * 100
    
    # ========================
    # 8. COMBINED SHORT SIGNALS
    # ========================
    # Primary short signal
    df['short_signal_primary'] = (
        (df['touched_resistance_5'] == 1) &
        (df['strong_rejection'] == 1) &
        (df['volume_above_avg'] == 1)
    ).astype(int)
    
    # Strong short signal
    df['short_signal_strong'] = (
        (df['touched_resistance_5'] == 1) &
        (df['very_strong_rejection'] == 1) &
        (df['volume_spike'] == 1) &
        (df['overbought_zscore'] == 1)
    ).astype(int)
    
    # Conservative short signal (multiple confirmations)
    df['short_signal_conservative'] = (
        (df['touched_resistance_5'] == 1) &
        (df['strong_rejection'] == 1) &
        (df['volume_spike'] == 1) &
        (df['below_vwap'] == 1) &
        (df['momentum_slowing'] == 1)
    ).astype(int)
    
    # Simple returns for context
    for period in [1, 5, 15, 30]:
        df[f'return_{period}'] = df['close'].pct_change(period)
    
    return df


def main():
    print("=" * 80)
    print("MODEL 9: LIQUIDITY REJECTION SHORT-ONLY")
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
    print("\n[2] Building rejection features...")
    df = build_rejection_features(df)
    
    # Create SHORT-ONLY labels (only negative price movements)
    print("\n[3] Creating SHORT-ONLY labels...")
    
    # Calculate future return
    future_return = df['close'].shift(-LABEL_HORIZON) / df['close'] - 1
    
    # SHORT-only: 1 = price goes DOWN, 0 = price stays flat or goes UP
    # We want to predict if a SHORT trade will be profitable
    min_profit = 0.001  # Minimum 0.1% move to count as "down"
    df['label'] = (future_return < -min_profit).astype(int)
    
    # Distribution check
    label_dist = df['label'].value_counts(normalize=True)
    print(f"  Label distribution:")
    print(f"    DOWN (1): {label_dist.get(1, 0):.1%}")
    print(f"    UP (0):   {label_dist.get(0, 0):.1%}")
    
    # Split data
    print("\n[4] Splitting data...")
    train_df = df[TRAIN_START:TRAIN_END].copy()
    val_df = df[VAL_START:VAL_END].copy()
    test_df = df[TEST_START:TEST_END].copy()
    
    print(f"  Train: {len(train_df):,}")
    print(f"  Validation: {len(val_df):,}")
    print(f"  Test: {len(test_df):,}")
    
    # Select features
    rejection_features = [
        # Resistance
        'dist_to_resistance', 'at_resistance', 'touched_resistance_5',
        'failed_breakout',
        
        # Rejection
        'rejection_score', 'strong_rejection', 'very_strong_rejection',
        'rejection_streak',
        
        # Volume
        'volume_ratio', 'volume_above_avg', 'volume_spike', 'volume_climax',
        
        # Momentum exhaustion
        'momentum_10', 'momentum_5', 'momentum_slowing',
        'zscore_50', 'overbought_zscore', 'extremely_overbought',
        
        # Candle patterns
        'shooting_star', 'bearish_engulfing', 'evening_star',
        
        # Trend reversal
        'lower_high', 'lower_close', 'downtrend_bars_3',
        'below_vwap',
        
        # Risk
        'volatility_30',
        
        # Combined signals
        'short_signal_primary', 'short_signal_strong', 'short_signal_conservative',
        
        # Returns
        'return_1', 'return_5', 'return_15', 'return_30'
    ]
    
    # Filter to existing columns
    feature_cols = [f for f in rejection_features if f in train_df.columns]
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
    
    print(f"  Train samples: {len(X_train):,} (down={y_train.mean():.1%})")
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
    threshold = 0.60  # Higher threshold for SHORT-only
    signals_short = (val_proba >= threshold).sum()
    print(f"\n  SHORT signals (threshold={threshold}):")
    print(f"    Count: {signals_short:,} ({signals_short/len(val_proba):.1%} of bars)")
    
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
        'strategy': 'rejection_short_only',
        'direction': 'SHORT_ONLY',
        'math_basis': 'liquidity_rejection_reversal',
        'train_period': f"{TRAIN_START} to {TRAIN_END}",
        'val_auc': roc_auc_score(y_val, val_proba),
        'saved_at': datetime.now().isoformat()
    }
    
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    print("  Saved!")
    
    print("\n" + "=" * 80)
    print("MODEL 9 TRAINING COMPLETE (SHORT-ONLY)")
    print("=" * 80)


if __name__ == "__main__":
    main()

