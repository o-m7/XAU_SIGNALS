#!/usr/bin/env python3
"""
Model 3 v3: Chaikin Money Flow + MACD Strategy

Mathematical Foundation:
- CMF = sum(((close-low)-(high-close))/(high-low)*volume, n) / sum(volume, n)
  - CMF > 0: Buying pressure (accumulation)
  - CMF < 0: Selling pressure (distribution)
  
- MACD = EMA(12) - EMA(26)
  - Signal = EMA(MACD, 9)
  - Histogram = MACD - Signal
  
Entry Logic:
- LONG: CMF > 0.05 AND MACD crosses above signal AND volume > average
- SHORT: CMF < -0.05 AND MACD crosses below signal AND volume > average
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.labels_v2 import create_validated_labels

# Paths
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "model3_v3_cmf_macd.joblib"

# Config
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-06-30"
VAL_START = "2023-07-01"
VAL_END = "2024-06-30"
TEST_START = "2024-07-01"
TEST_END = "2025-12-31"

LABEL_HORIZON = 30  # 30 minutes


def ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def calculate_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Calculate Chaikin Money Flow.
    
    CMF = sum(MFV, period) / sum(Volume, period)
    MFV = ((Close - Low) - (High - Close)) / (High - Low) * Volume
    
    Interpretation:
    - CMF > 0: Buying pressure dominates (bullish)
    - CMF < 0: Selling pressure dominates (bearish)
    - CMF magnitude: Strength of pressure
    """
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # Money Flow Multiplier: ranges from -1 to +1
    # +1 when close = high, -1 when close = low
    hl_range = high - low
    hl_range = hl_range.replace(0, np.nan)  # Avoid division by zero
    
    mf_multiplier = ((close - low) - (high - close)) / hl_range
    
    # Money Flow Volume
    mf_volume = mf_multiplier * volume
    
    # CMF = sum(MFV) / sum(Volume) over period
    cmf = mf_volume.rolling(period).sum() / volume.rolling(period).sum()
    
    return cmf


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    Calculate MACD indicator.
    
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = ema(df['close'], fast)
    ema_slow = ema(df['close'], slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def build_cmf_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build CMF and MACD related features."""
    df = df.copy()
    
    print("  Building CMF/MACD features...")
    
    # CMF at different periods
    df['cmf_10'] = calculate_cmf(df, 10)
    df['cmf_20'] = calculate_cmf(df, 20)
    df['cmf_40'] = calculate_cmf(df, 40)
    
    # CMF trend (is CMF increasing?)
    df['cmf_trend'] = df['cmf_20'] - df['cmf_20'].shift(5)
    
    # CMF threshold signals (pre-computed for model)
    df['cmf_bullish'] = (df['cmf_20'] > 0.05).astype(int)
    df['cmf_bearish'] = (df['cmf_20'] < -0.05).astype(int)
    df['cmf_strong_bullish'] = (df['cmf_20'] > 0.10).astype(int)
    df['cmf_strong_bearish'] = (df['cmf_20'] < -0.10).astype(int)
    
    # MACD
    macd, signal, histogram = calculate_macd(df)
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_histogram'] = histogram
    
    # MACD normalized (relative to price)
    df['macd_normalized'] = df['macd'] / df['close'] * 100
    df['macd_histogram_normalized'] = df['macd_histogram'] / df['close'] * 100
    
    # MACD crossover detection
    df['macd_prev'] = df['macd'].shift(1)
    df['signal_prev'] = df['macd_signal'].shift(1)
    df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & 
                           (df['macd_prev'] <= df['signal_prev'])).astype(int)
    df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & 
                             (df['macd_prev'] >= df['signal_prev'])).astype(int)
    
    # MACD momentum (histogram slope)
    df['macd_momentum'] = df['macd_histogram'] - df['macd_histogram'].shift(3)
    
    # MACD above/below zero
    df['macd_positive'] = (df['macd'] > 0).astype(int)
    df['macd_histogram_positive'] = (df['macd_histogram'] > 0).astype(int)
    
    # Combined signals
    df['cmf_macd_bullish'] = ((df['cmf_bullish'] == 1) & 
                              (df['macd_histogram_positive'] == 1)).astype(int)
    df['cmf_macd_bearish'] = ((df['cmf_bearish'] == 1) & 
                              (df['macd_histogram_positive'] == 0)).astype(int)
    
    # Volume confirmation
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
    df['volume_above_avg'] = (df['volume_ratio'] > 1.0).astype(int)
    df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)
    
    # Price trend features
    df['return_5'] = df['close'].pct_change(5)
    df['return_15'] = df['close'].pct_change(15)
    df['return_30'] = df['close'].pct_change(30)
    
    # Volatility
    df['volatility_15'] = df['close'].pct_change().rolling(15).std()
    
    # Support/Resistance levels from CMF
    df['cmf_support'] = df['cmf_20'].rolling(50).min()
    df['cmf_resistance'] = df['cmf_20'].rolling(50).max()
    df['cmf_position'] = (df['cmf_20'] - df['cmf_support']) / \
                         (df['cmf_resistance'] - df['cmf_support'] + 1e-10)
    
    # Remove temp columns
    df = df.drop(columns=['macd_prev', 'signal_prev'], errors='ignore')
    
    return df


def main():
    print("=" * 80)
    print("MODEL 3 v3: CHAIKIN MONEY FLOW + MACD")
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
    print("\n[2] Building CMF/MACD features...")
    df = build_cmf_macd_features(df)
    
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
    
    # Select features
    cmf_macd_features = [
        # CMF features
        'cmf_10', 'cmf_20', 'cmf_40',
        'cmf_trend',
        'cmf_bullish', 'cmf_bearish',
        'cmf_strong_bullish', 'cmf_strong_bearish',
        'cmf_position',
        
        # MACD features
        'macd', 'macd_signal', 'macd_histogram',
        'macd_normalized', 'macd_histogram_normalized',
        'macd_cross_up', 'macd_cross_down',
        'macd_momentum',
        'macd_positive', 'macd_histogram_positive',
        
        # Combined
        'cmf_macd_bullish', 'cmf_macd_bearish',
        
        # Volume
        'volume_ratio', 'volume_above_avg', 'volume_spike',
        
        # Price
        'return_5', 'return_15', 'return_30',
        'volatility_15'
    ]
    
    # Filter to existing columns
    feature_cols = [f for f in cmf_macd_features if f in train_df.columns]
    print(f"\n[5] Using {len(feature_cols)} features")
    
    # Prepare training data (exclude neutral labels)
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
    
    # Train model (XGBoost)
    print("\n[6] Training XGBoost model...")
    
    # Calculate scale_pos_weight for imbalanced classes
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    model = XGBClassifier(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        min_child_weight=200,
        reg_lambda=0.3,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    # Evaluate
    print("\n[7] Evaluation...")
    
    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]
    
    print(f"\n  Validation Results:")
    print(f"    Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"    ROC-AUC:  {roc_auc_score(y_val, val_proba):.4f}")
    
    # Signal distribution with threshold
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
        'model_type': 'XGBoost',
        'features': feature_cols,
        'label_horizon': LABEL_HORIZON,
        'threshold': threshold,
        'strategy': 'cmf_macd',
        'math_basis': 'chaikin_money_flow_macd',
        'train_period': f"{TRAIN_START} to {TRAIN_END}",
        'val_auc': roc_auc_score(y_val, val_proba),
        'label_validation': validation,
        'saved_at': datetime.now().isoformat()
    }
    
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    print("  Saved!")
    
    print("\n" + "=" * 80)
    print("MODEL 3 v3 TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

