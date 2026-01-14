#!/usr/bin/env python3
"""
Model 5 v3: Statistical Mean Reversion

Mathematical Foundation:
- Ornstein-Uhlenbeck Process: dX = θ(μ - X)dt + σdW
  - θ (theta): Speed of mean reversion
  - μ (mu): Long-term mean
  - σ (sigma): Volatility
  
- Z-Score: (Price - Mean) / StdDev
  - |Z| > 2.0: Extreme deviation (95% confidence)
  - |Z| > 2.5: Very extreme (99% confidence)
  
Entry Logic:
- LONG: Z-score < -2.0 AND market is ranging (ADX < 20)
- SHORT: Z-score > 2.0 AND market is ranging (ADX < 20)
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
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "model5_v3_mean_reversion.joblib"

# Config
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-06-30"
VAL_START = "2023-07-01"
VAL_END = "2024-06-30"
TEST_START = "2024-07-01"
TEST_END = "2025-12-31"

LABEL_HORIZON = 30  # 30 minutes


def calculate_ou_parameters(series: pd.Series, window: int = 100) -> tuple:
    """
    Estimate Ornstein-Uhlenbeck parameters using rolling regression.
    
    dX = θ(μ - X)dt + σdW
    
    Returns:
        theta: Mean reversion speed (higher = faster reversion)
        mu: Long-term mean
        sigma: Volatility
    """
    # Simple approach: regress X_{t+1} - X_t on X_t
    dx = series.diff()
    x = series.shift(1)
    
    # Rolling regression
    xy = (x * dx).rolling(window).sum()
    x2 = (x ** 2).rolling(window).sum()
    
    # theta = -slope (since dx = -theta * x + theta * mu)
    theta = -xy / (x2 + 1e-10)
    theta = theta.clip(0, 10)  # Reasonable bounds
    
    # mu estimated as rolling mean
    mu = series.rolling(window).mean()
    
    # sigma from residual volatility
    sigma = dx.rolling(window).std()
    
    return theta, mu, sigma


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index.
    
    ADX < 20: Ranging market (good for mean reversion)
    ADX > 25: Trending market (bad for mean reversion)
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(span=period).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period).mean() / (atr + 1e-10)
    
    # DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return adx


def build_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build mean reversion features based on OU process."""
    df = df.copy()
    
    print("  Building mean reversion features...")
    
    # Z-scores at multiple lookback periods
    for window in [20, 50, 100]:
        rolling_mean = df['close'].rolling(window).mean()
        rolling_std = df['close'].rolling(window).std()
        df[f'zscore_{window}'] = (df['close'] - rolling_mean) / (rolling_std + 1e-10)
    
    # Extreme deviation signals (pre-computed)
    df['zscore_extreme_low'] = (df['zscore_50'] < -2.0).astype(int)
    df['zscore_extreme_high'] = (df['zscore_50'] > 2.0).astype(int)
    df['zscore_very_extreme_low'] = (df['zscore_50'] < -2.5).astype(int)
    df['zscore_very_extreme_high'] = (df['zscore_50'] > 2.5).astype(int)
    
    # Ornstein-Uhlenbeck parameters
    print("    Calculating OU parameters...")
    theta, mu, sigma = calculate_ou_parameters(df['close'], window=100)
    df['ou_theta'] = theta  # Mean reversion speed
    df['ou_mu'] = mu  # Long-term mean
    df['ou_sigma'] = sigma  # Volatility
    
    # Half-life of mean reversion: ln(2) / theta
    df['ou_halflife'] = np.log(2) / (df['ou_theta'] + 1e-10)
    df['ou_halflife'] = df['ou_halflife'].clip(1, 200)  # Reasonable bounds
    
    # Deviation from OU mean
    df['ou_deviation'] = df['close'] - df['ou_mu']
    df['ou_deviation_pct'] = df['ou_deviation'] / df['ou_mu'] * 100
    
    # ADX for regime detection
    print("    Calculating ADX...")
    df['adx_14'] = calculate_adx(df, 14)
    
    # Regime flags
    df['is_ranging'] = (df['adx_14'] < 20).astype(int)
    df['is_trending'] = (df['adx_14'] > 25).astype(int)
    df['is_strong_trend'] = (df['adx_14'] > 35).astype(int)
    
    # Bollinger Band position
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
    df['bb_outside_upper'] = (df['close'] > bb_upper).astype(int)
    df['bb_outside_lower'] = (df['close'] < bb_lower).astype(int)
    
    # RSI for overbought/oversold
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
    
    # Combined mean reversion signals
    df['mr_long_signal'] = ((df['zscore_extreme_low'] == 1) & 
                            (df['is_ranging'] == 1)).astype(int)
    df['mr_short_signal'] = ((df['zscore_extreme_high'] == 1) & 
                             (df['is_ranging'] == 1)).astype(int)
    
    # Price velocity (to filter out fast breakouts)
    df['price_velocity'] = df['close'].diff(5).abs() / df['close'] * 100
    df['slow_market'] = (df['price_velocity'] < df['price_velocity'].rolling(50).median()).astype(int)
    
    # Volatility features
    df['volatility_15'] = df['close'].pct_change().rolling(15).std() * 100
    df['volatility_ratio'] = df['volatility_15'] / df['close'].pct_change().rolling(100).std()
    
    # Volume
    if 'volume' in df.columns:
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
    
    return df


def main():
    print("=" * 80)
    print("MODEL 5 v3: STATISTICAL MEAN REVERSION")
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
    print("\n[2] Building mean reversion features...")
    df = build_mean_reversion_features(df)
    
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
    mr_features = [
        # Z-scores
        'zscore_20', 'zscore_50', 'zscore_100',
        'zscore_extreme_low', 'zscore_extreme_high',
        'zscore_very_extreme_low', 'zscore_very_extreme_high',
        
        # OU parameters
        'ou_theta', 'ou_halflife', 'ou_deviation_pct',
        
        # Regime
        'adx_14', 'is_ranging', 'is_trending',
        
        # Bollinger
        'bb_position', 'bb_outside_upper', 'bb_outside_lower',
        
        # RSI
        'rsi_14', 'rsi_oversold', 'rsi_overbought',
        
        # Combined signals
        'mr_long_signal', 'mr_short_signal',
        
        # Market state
        'price_velocity', 'slow_market',
        'volatility_15', 'volatility_ratio'
    ]
    
    # Add volume if available
    if 'volume_ratio' in train_df.columns:
        mr_features.append('volume_ratio')
    
    # Filter to existing columns
    feature_cols = [f for f in mr_features if f in train_df.columns]
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
        'strategy': 'mean_reversion',
        'math_basis': 'ornstein_uhlenbeck_zscore',
        'train_period': f"{TRAIN_START} to {TRAIN_END}",
        'val_auc': roc_auc_score(y_val, val_proba),
        'label_validation': validation,
        'saved_at': datetime.now().isoformat()
    }
    
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    print("  Saved!")
    
    print("\n" + "=" * 80)
    print("MODEL 5 v3 TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

