#!/usr/bin/env python3
"""
Model 6 v3: Microstructure Alpha

Mathematical Foundation:
- Kyle's Lambda (λ): Price impact per unit of order flow
  - High λ: Low liquidity, large price moves from small orders
  - Low λ: High liquidity, prices stable despite large orders
  
- Order Flow Imbalance (OFI): 
  - OFI = bid_size_change - ask_size_change
  - Positive OFI: Buying pressure
  - Negative OFI: Selling pressure
  
- VPIN (Volume-Synchronized Probability of Informed Trading):
  - Measures probability that trades are from informed traders

Entry Logic:
- LONG: Order flow imbalance significantly positive + low impact
- SHORT: Order flow imbalance significantly negative + low impact
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
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "model6_v3_microstructure.joblib"

# Config
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-06-30"
VAL_START = "2023-07-01"
VAL_END = "2024-06-30"
TEST_START = "2024-07-01"
TEST_END = "2025-12-31"

LABEL_HORIZON = 30  # 30 minutes


def calculate_kyle_lambda(df: pd.DataFrame, window: int = 50) -> pd.Series:
    """
    Estimate Kyle's Lambda: price impact per unit of signed volume.
    
    λ = Cov(ΔP, SignedVolume) / Var(SignedVolume)
    
    Higher lambda = larger price moves per unit of flow = lower liquidity
    """
    # Price change
    price_change = df['close'].diff()
    
    # Signed volume (positive if close > open, negative if close < open)
    sign = np.sign(df['close'] - df['open'])
    signed_volume = sign * df['volume']
    
    # Rolling covariance and variance
    cov_rolling = price_change.rolling(window).cov(signed_volume)
    var_rolling = signed_volume.rolling(window).var()
    
    # Kyle's Lambda
    kyle_lambda = cov_rolling / (var_rolling + 1e-10)
    
    return kyle_lambda


def calculate_order_flow_imbalance(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate Order Flow Imbalance proxy from OHLCV.
    
    Since we don't have bid/ask sizes, we use price action:
    - Close near high = buying pressure
    - Close near low = selling pressure
    """
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # Price position in range
    range_size = high - low
    position = (close - low) / (range_size + 1e-10)  # 0 to 1
    
    # Signed pressure: -1 to +1
    pressure = (position - 0.5) * 2
    
    # Volume-weighted pressure
    flow = pressure * volume
    
    # Rolling sum for imbalance
    ofi = flow.rolling(window).sum()
    
    return ofi


def calculate_vpin_proxy(df: pd.DataFrame, bucket_size: int = 50) -> pd.Series:
    """
    Calculate VPIN proxy (Volume-Synchronized Probability of Informed Trading).
    
    VPIN estimates the probability of trades being information-driven.
    Higher VPIN = more informed trading = potential price movement.
    """
    # Classify bars as buy or sell based on close vs open
    buy_volume = df['volume'].where(df['close'] > df['open'], 0)
    sell_volume = df['volume'].where(df['close'] < df['open'], 0)
    
    # Rolling sums
    total_vol = df['volume'].rolling(bucket_size).sum()
    buy_vol = buy_volume.rolling(bucket_size).sum()
    sell_vol = sell_volume.rolling(bucket_size).sum()
    
    # VPIN = |Buy - Sell| / Total
    vpin = (buy_vol - sell_vol).abs() / (total_vol + 1e-10)
    
    return vpin


def build_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build microstructure-based features."""
    df = df.copy()
    
    print("  Building microstructure features...")
    
    # Kyle's Lambda at multiple windows
    print("    Calculating Kyle's Lambda...")
    for window in [20, 50, 100]:
        df[f'kyle_lambda_{window}'] = calculate_kyle_lambda(df, window)
    
    # Normalize Kyle's Lambda
    df['kyle_lambda_zscore'] = (df['kyle_lambda_50'] - df['kyle_lambda_50'].rolling(200).mean()) / \
                               (df['kyle_lambda_50'].rolling(200).std() + 1e-10)
    
    # High/low liquidity signals
    df['low_liquidity'] = (df['kyle_lambda_zscore'] > 1.0).astype(int)
    df['high_liquidity'] = (df['kyle_lambda_zscore'] < -1.0).astype(int)
    
    # Order Flow Imbalance
    print("    Calculating Order Flow Imbalance...")
    for window in [10, 20, 50]:
        df[f'ofi_{window}'] = calculate_order_flow_imbalance(df, window)
    
    # Normalize OFI
    df['ofi_zscore'] = (df['ofi_20'] - df['ofi_20'].rolling(200).mean()) / \
                       (df['ofi_20'].rolling(200).std() + 1e-10)
    
    # OFI signals
    df['ofi_strong_buy'] = (df['ofi_zscore'] > 2.0).astype(int)
    df['ofi_strong_sell'] = (df['ofi_zscore'] < -2.0).astype(int)
    df['ofi_moderate_buy'] = (df['ofi_zscore'] > 1.0).astype(int)
    df['ofi_moderate_sell'] = (df['ofi_zscore'] < -1.0).astype(int)
    
    # VPIN proxy
    print("    Calculating VPIN proxy...")
    df['vpin_50'] = calculate_vpin_proxy(df, 50)
    df['vpin_100'] = calculate_vpin_proxy(df, 100)
    
    # High VPIN (potential informed trading)
    df['vpin_high'] = (df['vpin_50'] > 0.3).astype(int)
    df['vpin_very_high'] = (df['vpin_50'] > 0.5).astype(int)
    
    # Spread proxy (high-low relative to close)
    df['spread_proxy'] = (df['high'] - df['low']) / df['close']
    df['spread_zscore'] = (df['spread_proxy'] - df['spread_proxy'].rolling(100).mean()) / \
                          (df['spread_proxy'].rolling(100).std() + 1e-10)
    df['wide_spread'] = (df['spread_zscore'] > 1.5).astype(int)
    
    # Realized volatility (5-minute)
    df['rv_5'] = df['close'].pct_change().rolling(5).std() * np.sqrt(5) * 100
    df['rv_15'] = df['close'].pct_change().rolling(15).std() * np.sqrt(15) * 100
    
    # Volume profile
    df['volume_ma'] = df['volume'].rolling(50).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
    df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)
    df['volume_dry'] = (df['volume_ratio'] < 0.5).astype(int)
    
    # Price momentum for context
    df['return_5'] = df['close'].pct_change(5)
    df['return_15'] = df['close'].pct_change(15)
    
    # Combined signals
    df['ms_long_signal'] = ((df['ofi_moderate_buy'] == 1) & 
                            (df['high_liquidity'] == 1)).astype(int)
    df['ms_short_signal'] = ((df['ofi_moderate_sell'] == 1) & 
                             (df['high_liquidity'] == 1)).astype(int)
    
    # Adverse selection risk (high VPIN + low liquidity = danger)
    df['adverse_selection'] = ((df['vpin_high'] == 1) & 
                               (df['low_liquidity'] == 1)).astype(int)
    
    return df


def main():
    print("=" * 80)
    print("MODEL 6 v3: MICROSTRUCTURE ALPHA")
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
    print("\n[2] Building microstructure features...")
    df = build_microstructure_features(df)
    
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
    ms_features = [
        # Kyle's Lambda
        'kyle_lambda_20', 'kyle_lambda_50', 'kyle_lambda_100',
        'kyle_lambda_zscore',
        'low_liquidity', 'high_liquidity',
        
        # Order Flow Imbalance
        'ofi_10', 'ofi_20', 'ofi_50',
        'ofi_zscore',
        'ofi_strong_buy', 'ofi_strong_sell',
        'ofi_moderate_buy', 'ofi_moderate_sell',
        
        # VPIN
        'vpin_50', 'vpin_100',
        'vpin_high', 'vpin_very_high',
        
        # Spread
        'spread_proxy', 'spread_zscore', 'wide_spread',
        
        # Volatility
        'rv_5', 'rv_15',
        
        # Volume
        'volume_ratio', 'volume_spike', 'volume_dry',
        
        # Momentum context
        'return_5', 'return_15',
        
        # Combined
        'ms_long_signal', 'ms_short_signal',
        'adverse_selection'
    ]
    
    # Filter to existing columns
    feature_cols = [f for f in ms_features if f in train_df.columns]
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
        'model_type': 'RandomForest',
        'features': feature_cols,
        'label_horizon': LABEL_HORIZON,
        'threshold': threshold,
        'strategy': 'microstructure',
        'math_basis': 'kyle_lambda_ofi_vpin',
        'train_period': f"{TRAIN_START} to {TRAIN_END}",
        'val_auc': roc_auc_score(y_val, val_proba),
        'label_validation': validation,
        'saved_at': datetime.now().isoformat()
    }
    
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    print("  Saved!")
    
    print("\n" + "=" * 80)
    print("MODEL 6 v3 TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

