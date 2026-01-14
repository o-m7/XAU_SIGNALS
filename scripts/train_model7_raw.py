#!/usr/bin/env python3
"""
Model 7: Raw Price Predictor

Strategy: Pure price action using ONLY raw OHLCV data (no indicators)

Mathematical Foundation:
- Log returns: r_t = log(P_t / P_{t-1})
- Range ratios: body/range, upper_wick/range, lower_wick/range
- Candle patterns expressed mathematically
- Volume dynamics

This model tests if raw price information alone can predict direction.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.labels_v2 import create_validated_labels

# Paths
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "model7_raw_price.joblib"

# Config
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-06-30"
VAL_START = "2023-07-01"
VAL_END = "2024-06-30"
TEST_START = "2024-07-01"
TEST_END = "2025-12-31"

LABEL_HORIZON = 30  # 30 minutes


def build_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build features using ONLY raw OHLCV data."""
    df = df.copy()
    
    print("  Building raw price features (no indicators)...")
    
    # ========================
    # 1. LOG RETURNS
    # ========================
    for period in [1, 2, 3, 5, 10, 15, 30]:
        df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
    
    # Cumulative return
    df['cum_return_5'] = df['log_return_1'].rolling(5).sum()
    df['cum_return_15'] = df['log_return_1'].rolling(15).sum()
    
    # ========================
    # 2. CANDLE ANATOMY (RATIOS)
    # ========================
    range_size = df['high'] - df['low']
    body = abs(df['close'] - df['open'])
    
    # Body ratio (0 = doji, 1 = full body)
    df['body_ratio'] = body / (range_size + 1e-10)
    
    # Body direction (1 = bullish, -1 = bearish)
    df['body_direction'] = np.sign(df['close'] - df['open'])
    
    # Upper wick ratio
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    df['upper_wick_ratio'] = upper_wick / (range_size + 1e-10)
    
    # Lower wick ratio
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    df['lower_wick_ratio'] = lower_wick / (range_size + 1e-10)
    
    # Wick asymmetry (positive = more upper wick)
    df['wick_asymmetry'] = df['upper_wick_ratio'] - df['lower_wick_ratio']
    
    # Range relative to price
    df['range_pct'] = range_size / df['close'] * 100
    
    # ========================
    # 3. PRICE POSITION (WHERE IS CLOSE IN RANGE?)
    # ========================
    df['close_position'] = (df['close'] - df['low']) / (range_size + 1e-10)  # 0 to 1
    
    # Rolling close position
    for window in [5, 15]:
        rolling_high = df['high'].rolling(window).max()
        rolling_low = df['low'].rolling(window).min()
        df[f'price_position_{window}'] = (df['close'] - rolling_low) / \
                                          (rolling_high - rolling_low + 1e-10)
    
    # ========================
    # 4. GAP ANALYSIS
    # ========================
    df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    df['gap_filled'] = ((df['gap'] > 0) & (df['low'] <= df['close'].shift(1))).astype(int) | \
                       ((df['gap'] < 0) & (df['high'] >= df['close'].shift(1))).astype(int)
    
    # ========================
    # 5. HIGHER HIGHS / LOWER LOWS
    # ========================
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
    df['higher_close'] = (df['close'] > df['close'].shift(1)).astype(int)
    
    # Consecutive patterns
    df['consecutive_up'] = df['higher_close'].rolling(3).sum()
    df['consecutive_down'] = (1 - df['higher_close']).rolling(3).sum()
    
    # ========================
    # 6. VOLATILITY (RAW, NO INDICATORS)
    # ========================
    df['raw_volatility_5'] = df['log_return_1'].rolling(5).std()
    df['raw_volatility_15'] = df['log_return_1'].rolling(15).std()
    df['raw_volatility_30'] = df['log_return_1'].rolling(30).std()
    
    # Volatility change
    df['vol_change'] = df['raw_volatility_15'] / (df['raw_volatility_30'] + 1e-10)
    
    # ========================
    # 7. VOLUME FEATURES (RAW)
    # ========================
    if 'volume' in df.columns:
        # Volume change
        df['volume_change'] = df['volume'] / (df['volume'].shift(1) + 1e-10)
        
        # Volume vs moving window
        for window in [5, 15, 30]:
            df[f'volume_vs_{window}'] = df['volume'] / (df['volume'].rolling(window).mean() + 1e-10)
        
        # Volume-price correlation (is volume confirming move?)
        df['vol_price_sign'] = np.sign(df['log_return_1']) * np.sign(df['volume_change'] - 1)
        df['vol_price_corr_5'] = df['log_return_1'].rolling(5).corr(df['volume'].pct_change())
    
    # ========================
    # 8. PATTERN SCORES
    # ========================
    # Engulfing pattern score
    prev_body = abs(df['close'].shift(1) - df['open'].shift(1))
    curr_body = abs(df['close'] - df['open'])
    engulfing = (curr_body > prev_body * 1.5) & (df['body_direction'] != df['body_direction'].shift(1))
    df['engulfing'] = engulfing.astype(int)
    df['bullish_engulfing'] = (engulfing & (df['body_direction'] == 1)).astype(int)
    df['bearish_engulfing'] = (engulfing & (df['body_direction'] == -1)).astype(int)
    
    # Doji detection (small body relative to range)
    df['is_doji'] = (df['body_ratio'] < 0.1).astype(int)
    
    # Hammer/Shooting star
    df['hammer_like'] = ((df['lower_wick_ratio'] > 0.6) & (df['upper_wick_ratio'] < 0.2)).astype(int)
    df['shooting_star_like'] = ((df['upper_wick_ratio'] > 0.6) & (df['lower_wick_ratio'] < 0.2)).astype(int)
    
    # ========================
    # 9. MOMENTUM (RAW)
    # ========================
    # Simple momentum: sum of returns
    df['momentum_5'] = df['log_return_1'].rolling(5).sum()
    df['momentum_15'] = df['log_return_1'].rolling(15).sum()
    
    # Momentum change
    df['momentum_delta'] = df['momentum_5'] - df['momentum_5'].shift(5)
    
    # ========================
    # 10. TIME FEATURES
    # ========================
    if df.index.dtype == 'datetime64[ns]':
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['day_of_week'] = df.index.dayofweek
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    return df


def main():
    print("=" * 80)
    print("MODEL 7: RAW PRICE PREDICTOR (NO INDICATORS)")
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
    print("\n[2] Building raw features...")
    df = build_raw_features(df)
    
    # Create validated labels
    print("\n[3] Creating validated labels...")
    labels, validation = create_validated_labels(df, horizon=LABEL_HORIZON, method='direction')
    
    if not validation['is_valid']:
        print("  WARNING: Labels did not pass validation!")
    
    df['label'] = labels
    
    # Split data
    print("\n[4] Splitting data...")
    train_df = df[TRAIN_START:TRAIN_END].copy()
    val_df = df[VAL_START:VAL_END].copy()
    test_df = df[TEST_START:TEST_END].copy()
    
    print(f"  Train: {len(train_df):,}")
    print(f"  Validation: {len(val_df):,}")
    print(f"  Test: {len(test_df):,}")
    
    # Select RAW features only (no traditional indicators)
    raw_features = [
        # Returns
        'log_return_1', 'log_return_2', 'log_return_3', 'log_return_5',
        'log_return_10', 'log_return_15', 'log_return_30',
        'cum_return_5', 'cum_return_15',
        
        # Candle anatomy
        'body_ratio', 'body_direction',
        'upper_wick_ratio', 'lower_wick_ratio', 'wick_asymmetry',
        'range_pct', 'close_position',
        
        # Price position
        'price_position_5', 'price_position_15',
        
        # Gap
        'gap', 'gap_filled',
        
        # Higher/lower
        'higher_high', 'lower_low', 'higher_close',
        'consecutive_up', 'consecutive_down',
        
        # Volatility (raw)
        'raw_volatility_5', 'raw_volatility_15', 'raw_volatility_30',
        'vol_change',
        
        # Volume (raw)
        'volume_change', 'volume_vs_5', 'volume_vs_15', 'volume_vs_30',
        'vol_price_sign', 'vol_price_corr_5',
        
        # Patterns
        'engulfing', 'bullish_engulfing', 'bearish_engulfing',
        'is_doji', 'hammer_like', 'shooting_star_like',
        
        # Momentum (raw)
        'momentum_5', 'momentum_15', 'momentum_delta',
        
        # Time
        'hour_sin', 'hour_cos'
    ]
    
    # Filter to existing columns
    feature_cols = [f for f in raw_features if f in train_df.columns]
    print(f"\n[5] Using {len(feature_cols)} raw features")
    
    # Prepare training data
    train_clean = train_df.dropna(subset=['label'] + feature_cols)
    train_clean = train_clean[train_clean['label'] != 0]
    
    X_train = train_clean[feature_cols].values
    y_train = (train_clean['label'] == 1).astype(int).values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    val_clean = val_df.dropna(subset=['label'] + feature_cols)
    val_clean = val_clean[val_clean['label'] != 0]
    
    X_val = val_clean[feature_cols].values
    y_val = (val_clean['label'] == 1).astype(int).values
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  Train samples: {len(X_train):,} (up={y_train.mean():.1%})")
    print(f"  Val samples: {len(X_val):,}")
    
    # Train model (Random Forest)
    print("\n[6] Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=100,
        min_samples_split=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\n[7] Evaluation...")
    
    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]
    
    print(f"\n  Validation Results:")
    print(f"    Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"    ROC-AUC:  {roc_auc_score(y_val, val_proba):.4f}")
    
    # Signal distribution
    threshold = 0.55
    signals_long = (val_proba >= threshold).sum()
    signals_short = (val_proba <= 1 - threshold).sum()
    print(f"\n  Signal distribution (threshold={threshold}):")
    print(f"    LONG signals: {signals_long:,} ({signals_long/len(val_proba):.1%})")
    print(f"    SHORT signals: {signals_short:,} ({signals_short/len(val_proba):.1%})")
    
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
        'model_type': 'RandomForest',
        'features': feature_cols,
        'label_horizon': LABEL_HORIZON,
        'threshold': threshold,
        'strategy': 'raw_price',
        'math_basis': 'pure_ohlcv_no_indicators',
        'train_period': f"{TRAIN_START} to {TRAIN_END}",
        'val_auc': roc_auc_score(y_val, val_proba),
        'label_validation': validation,
        'saved_at': datetime.now().isoformat()
    }
    
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    print("  Saved!")
    
    print("\n" + "=" * 80)
    print("MODEL 7 TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

