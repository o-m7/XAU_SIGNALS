"""
Breakout Strategy Model

Strategy: Trade breakouts from consolidation zones
- Entry: When price breaks above/below recent range
- Filter: Only during high-volume sessions (London/NY overlap)
- R:R: 2:1 minimum (TP = 2x ATR, SL = 1x ATR)
- Confirmation: Volume spike on breakout

This is a completely different approach from mean-reversion.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = Path("/Users/omar/Desktop/ML/Data")
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_data(years):
    """Load minute data."""
    all_data = []
    for year in years:
        path = DATA_DIR / "ohlcv_second" / f"XAUUSD_second_{year}.parquet"
        if path.exists():
            print(f"  Loading {year}...")
            df = pd.read_parquet(path)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            all_data.append(df)
    
    df = pd.concat(all_data).sort_index()
    df.index = df.index.tz_localize(None)
    
    # Aggregate to minute
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    return df.resample('1min').agg(agg).dropna()


def add_features(df):
    """Add breakout-specific features."""
    df = df.copy()
    
    # ATR
    tr = np.maximum(df['high'] - df['low'],
                   np.maximum(abs(df['high'] - df['close'].shift(1)),
                             abs(df['low'] - df['close'].shift(1))))
    df['atr'] = tr.rolling(14).mean()
    
    # Range detection (consolidation)
    df['range_high_20'] = df['high'].rolling(20).max()
    df['range_low_20'] = df['low'].rolling(20).min()
    df['range_size_20'] = (df['range_high_20'] - df['range_low_20']) / df['close']
    
    df['range_high_50'] = df['high'].rolling(50).max()
    df['range_low_50'] = df['low'].rolling(50).min()
    df['range_size_50'] = (df['range_high_50'] - df['range_low_50']) / df['close']
    
    # Breakout signals
    df['breakout_up'] = (df['close'] > df['range_high_20'].shift(1)).astype(int)
    df['breakout_down'] = (df['close'] < df['range_low_20'].shift(1)).astype(int)
    
    # Volume confirmation
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
    df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)
    
    # Volatility squeeze (low vol = potential breakout coming)
    df['atr_ma'] = df['atr'].rolling(50).mean()
    df['volatility_ratio'] = df['atr'] / (df['atr_ma'] + 1e-10)
    df['squeeze'] = (df['volatility_ratio'] < 0.8).astype(int)
    df['expansion'] = (df['volatility_ratio'] > 1.2).astype(int)
    
    # Session timing (UTC)
    df['hour'] = df.index.hour
    df['is_london'] = ((df['hour'] >= 7) & (df['hour'] <= 16)).astype(int)
    df['is_ny'] = ((df['hour'] >= 12) & (df['hour'] <= 21)).astype(int)
    df['is_overlap'] = ((df['hour'] >= 12) & (df['hour'] <= 16)).astype(int)  # Best time
    
    # Momentum
    df['mom_5'] = df['close'].pct_change(5)
    df['mom_10'] = df['close'].pct_change(10)
    df['mom_20'] = df['close'].pct_change(20)
    
    # Trend filter
    df['ma_50'] = df['close'].rolling(50).mean()
    df['ma_200'] = df['close'].rolling(200).mean()
    df['trend_up'] = (df['ma_50'] > df['ma_200']).astype(int)
    df['trend_down'] = (df['ma_50'] < df['ma_200']).astype(int)
    df['price_above_ma50'] = (df['close'] > df['ma_50']).astype(int)
    
    # Consolidation tightness
    df['consolidation_bars'] = 0
    for i in range(20, len(df)):
        if df['range_size_20'].iloc[i] < 0.005:  # Tight range
            df.iloc[i, df.columns.get_loc('consolidation_bars')] = 1
    
    return df


def create_breakout_labels(df, horizon=30, tp_mult=2.0, sl_mult=1.0):
    """
    Create labels for breakout trades with 2:1 R:R.
    
    Only label bars where there's a valid breakout setup.
    """
    labels = pd.Series(index=df.index, dtype=float)
    directions = pd.Series(index=df.index, dtype=float)
    
    atr = df['atr']
    
    for i in range(50, len(df) - horizon):
        # Must be during good session
        if df.iloc[i]['is_overlap'] != 1 and df.iloc[i]['is_london'] != 1:
            continue
        
        # Must have volume confirmation
        if df.iloc[i]['volume_ratio'] < 1.2:
            continue
        
        entry_price = df.iloc[i]['close']
        current_atr = atr.iloc[i]
        
        if pd.isna(current_atr) or current_atr <= 0:
            continue
        
        # Check for breakout
        is_breakout_up = df.iloc[i]['breakout_up'] == 1
        is_breakout_down = df.iloc[i]['breakout_down'] == 1
        
        if not (is_breakout_up or is_breakout_down):
            continue
        
        # Determine direction
        if is_breakout_up and df.iloc[i]['trend_up'] == 1:
            direction = 1
        elif is_breakout_down and df.iloc[i]['trend_down'] == 1:
            direction = -1
        else:
            continue  # Only trade with trend
        
        directions.iloc[i] = direction
        
        # Define barriers (2:1 R:R)
        if direction == 1:
            tp = entry_price + current_atr * tp_mult
            sl = entry_price - current_atr * sl_mult
        else:
            tp = entry_price - current_atr * tp_mult
            sl = entry_price + current_atr * sl_mult
        
        # Check outcome
        hit_tp = False
        hit_sl = False
        
        for j in range(1, horizon + 1):
            if i + j >= len(df):
                break
            bar = df.iloc[i + j]
            
            if direction == 1:
                if bar['high'] >= tp:
                    hit_tp = True
                    break
                elif bar['low'] <= sl:
                    hit_sl = True
                    break
            else:
                if bar['low'] <= tp:
                    hit_tp = True
                    break
                elif bar['high'] >= sl:
                    hit_sl = True
                    break
        
        if hit_tp:
            labels.iloc[i] = 1
        elif hit_sl:
            labels.iloc[i] = 0
        else:
            # Time exit
            final = df.iloc[min(i + horizon, len(df) - 1)]['close']
            if direction == 1:
                labels.iloc[i] = 1 if final > entry_price else 0
            else:
                labels.iloc[i] = 1 if final < entry_price else 0
    
    return labels, directions


def train_model():
    """Train breakout model."""
    print("\n" + "="*60)
    print("BREAKOUT STRATEGY MODEL")
    print("2:1 Risk/Reward, Session Filtered")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = load_data([2020, 2021, 2022, 2023, 2024])
    print(f"Total bars: {len(df):,}")
    
    # Add features
    print("Building features...")
    df = add_features(df)
    
    # Create labels
    print("Creating breakout labels (2:1 R:R)...")
    df['label'], df['direction'] = create_breakout_labels(df, horizon=30, tp_mult=2.0, sl_mult=1.0)
    
    valid = df['label'].dropna()
    print(f"Valid setups: {len(valid):,}")
    print(f"Win rate in training data: {valid.mean():.1%}")
    
    if len(valid) < 1000:
        print("Not enough setups, adjusting parameters...")
        df['label'], df['direction'] = create_breakout_labels(df, horizon=60, tp_mult=1.5, sl_mult=1.0)
        valid = df['label'].dropna()
        print(f"After adjustment: {len(valid):,} setups, {valid.mean():.1%} WR")
    
    # Features
    feature_cols = [
        'range_size_20', 'range_size_50',
        'volume_ratio', 'volume_spike',
        'volatility_ratio', 'squeeze', 'expansion',
        'is_london', 'is_ny', 'is_overlap',
        'mom_5', 'mom_10', 'mom_20',
        'trend_up', 'trend_down', 'price_above_ma50',
        'breakout_up', 'breakout_down'
    ]
    
    # Prepare data
    valid_mask = df[feature_cols].notna().all(axis=1) & df['label'].notna()
    df_clean = df[valid_mask].copy()
    
    # Split
    split = '2024-01-01'
    train_df = df_clean[df_clean.index < split]
    val_df = df_clean[df_clean.index >= split]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['label'].values.astype(int)
    X_val = val_df[feature_cols].values
    y_val = val_df['label'].values.astype(int)
    
    print(f"\nTrain: {len(X_train):,}, {y_train.mean():.1%} winners")
    print(f"Val: {len(X_val):,}, {y_val.mean():.1%} winners")
    
    if len(X_train) < 100:
        print("ERROR: Not enough training data")
        return None
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train
    print("\nTraining XGBoost...")
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=n_neg/n_pos if n_pos > 0 else 1,
        eval_metric='auc',
        early_stopping_rounds=20,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], verbose=False)
    
    # Evaluate
    print("\nValidation Results:")
    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    acc = accuracy_score(y_val, (y_pred_proba >= 0.5).astype(int))
    auc = roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
    print(f"  Accuracy: {acc:.1%}")
    print(f"  AUC: {auc:.3f}")
    
    print("\nThreshold Analysis:")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = y_pred_proba >= thresh
        if mask.sum() > 0:
            wr = y_val[mask].mean()
            n = mask.sum()
            status = "✓" if wr >= 0.50 else ""  # 50% WR with 2:1 R:R = profitable
            print(f"  @{thresh:.0%}: {wr:.1%} WR, {n} trades {status}")
    
    # Save
    artifact = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'model_type': 'breakout',
        'horizon': 30,
        'tp_mult': 2.0,
        'sl_mult': 1.0,
        'train_date': datetime.now().isoformat()
    }
    
    save_path = MODEL_DIR / "model_breakout.joblib"
    joblib.dump(artifact, save_path)
    print(f"\nSaved to: {save_path}")
    
    return model


if __name__ == "__main__":
    train_model()

