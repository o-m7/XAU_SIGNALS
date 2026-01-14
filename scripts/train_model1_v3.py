#!/usr/bin/env python3
"""
Model 1 v3: High-Frequency Trend Following

Mathematical Foundation:
- Hurst Exponent: H > 0.5 indicates trending (persistent) market
- Autocorrelation: corr(r_t, r_{t-1}) > 0 confirms trend continuation
- Momentum Score: Standardized price change with statistical significance

Entry: When Hurst indicates trend AND momentum is statistically significant
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

from src.labels_v2 import create_validated_labels, create_binary_labels_for_training

# Paths
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "models" / "model1_v3_trend.joblib"

# Config
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-06-30"
VAL_START = "2023-07-01"
VAL_END = "2024-06-30"
TEST_START = "2024-07-01"
TEST_END = "2025-12-31"

LABEL_HORIZON = 30  # 30 minutes


def calculate_hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """
    Calculate Hurst Exponent using R/S analysis.
    
    H > 0.5: Trending (persistent)
    H = 0.5: Random walk
    H < 0.5: Mean-reverting (anti-persistent)
    """
    lags = range(2, max_lag)
    rs_values = []
    
    for lag in lags:
        # Divide series into chunks
        chunks = len(series) // lag
        if chunks < 1:
            continue
        
        rs_list = []
        for i in range(chunks):
            chunk = series.iloc[i*lag:(i+1)*lag]
            if len(chunk) < 2:
                continue
            
            # Mean-adjusted cumulative sum
            mean_adj = chunk - chunk.mean()
            cumsum = mean_adj.cumsum()
            
            # Range
            R = cumsum.max() - cumsum.min()
            
            # Standard deviation
            S = chunk.std()
            
            if S > 0:
                rs_list.append(R / S)
        
        if rs_list:
            rs_values.append((lag, np.mean(rs_list)))
    
    if len(rs_values) < 2:
        return 0.5
    
    # Linear regression of log(R/S) vs log(n)
    lags_arr = np.array([x[0] for x in rs_values])
    rs_arr = np.array([x[1] for x in rs_values])
    
    # Filter out zeros/negatives
    mask = rs_arr > 0
    if mask.sum() < 2:
        return 0.5
    
    log_lags = np.log(lags_arr[mask])
    log_rs = np.log(rs_arr[mask])
    
    # Hurst exponent is the slope
    slope, _ = np.polyfit(log_lags, log_rs, 1)
    
    return np.clip(slope, 0, 1)


def calculate_autocorrelation(series: pd.Series, lag: int = 1) -> float:
    """Calculate autocorrelation at given lag."""
    if len(series) < lag + 2:
        return 0.0
    return series.autocorr(lag=lag)


def build_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build Hurst Exponent and trend-related features."""
    df = df.copy()
    
    print("  Building trend features...")
    
    # Log returns at various scales
    for period in [1, 5, 15, 30]:
        df[f'return_{period}'] = np.log(df['close'] / df['close'].shift(period))
    
    # Rolling Hurst Exponent (simplified - compute every 100 bars)
    print("    Calculating rolling Hurst exponent (sampled)...")
    window = 100
    sample_rate = 50  # Only compute every 50 bars for speed
    hurst_values = np.full(len(df), 0.5)
    
    for i in range(window, len(df), sample_rate):
        h = calculate_hurst_exponent(df['close'].iloc[i-window:i])
        # Forward fill for next sample_rate bars
        end_idx = min(i + sample_rate, len(df))
        hurst_values[i:end_idx] = h
    
    df['hurst_100'] = hurst_values
    
    # Trend indicator: H > 0.5
    df['is_trending'] = (df['hurst_100'] > 0.55).astype(int)
    
    # Autocorrelation of returns
    print("    Calculating autocorrelation...")
    df['autocorr_1'] = df['return_1'].rolling(50).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 2 else 0, raw=False
    )
    df['autocorr_5'] = df['return_1'].rolling(50).apply(
        lambda x: x.autocorr(lag=5) if len(x) > 6 else 0, raw=False
    )
    
    # Momentum score (standardized)
    # M_t = (P_t - P_{t-n}) / (sigma_n * sqrt(n))
    for n in [15, 30]:
        price_change = df['close'] - df['close'].shift(n)
        rolling_std = df['close'].pct_change().rolling(n).std() * df['close']
        df[f'momentum_zscore_{n}'] = price_change / (rolling_std * np.sqrt(n) + 1e-10)
    
    # Momentum significance (|z| > 1.96 = 95% confidence)
    df['momentum_significant'] = (abs(df['momentum_zscore_30']) > 1.96).astype(int)
    
    # Trend strength: R² of linear regression
    print("    Calculating trend strength...")
    def calc_r2(series):
        if len(series) < 2:
            return 0
        x = np.arange(len(series))
        try:
            slope, intercept = np.polyfit(x, series, 1)
            predicted = slope * x + intercept
            ss_res = np.sum((series - predicted) ** 2)
            ss_tot = np.sum((series - series.mean()) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        except:
            return 0
    
    df['trend_r2_30'] = df['close'].rolling(30).apply(calc_r2, raw=False)
    
    # Price position relative to recent range
    df['price_position'] = (df['close'] - df['low'].rolling(30).min()) / \
                           (df['high'].rolling(30).max() - df['low'].rolling(30).min() + 1e-10)
    
    # Volatility features
    df['volatility_30'] = df['return_1'].rolling(30).std()
    df['volatility_ratio'] = df['volatility_30'] / df['return_1'].rolling(100).std()
    
    # Volume confirmation
    if 'volume' in df.columns:
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
    
    return df


def main():
    print("=" * 80)
    print("MODEL 1 v3: HIGH-FREQUENCY TREND FOLLOWING")
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
    
    # Build trend features
    print("\n[2] Building trend features...")
    df = build_trend_features(df)
    
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
    trend_features = [
        'return_1', 'return_5', 'return_15', 'return_30',
        'hurst_100', 'is_trending',
        'autocorr_1', 'autocorr_5',
        'momentum_zscore_15', 'momentum_zscore_30', 'momentum_significant',
        'trend_r2_30', 'price_position',
        'volatility_30', 'volatility_ratio'
    ]
    
    # Add volume if available
    if 'volume_ratio' in df.columns:
        trend_features.append('volume_ratio')
    
    # Filter to existing columns
    feature_cols = [f for f in trend_features if f in train_df.columns]
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
        'strategy': 'trend_following',
        'math_basis': 'hurst_exponent_autocorrelation',
        'train_period': f"{TRAIN_START} to {TRAIN_END}",
        'val_auc': roc_auc_score(y_val, val_proba),
        'label_validation': validation,
        'saved_at': datetime.now().isoformat()
    }
    
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_OUTPUT_PATH)
    print("  Saved!")
    
    print("\n" + "=" * 80)
    print("MODEL 1 v3 TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

