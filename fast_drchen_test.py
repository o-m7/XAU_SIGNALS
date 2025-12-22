#!/usr/bin/env python3
"""
FAST Dr. Chen System - EXTREME SELECTIVITY VERSION.
Only trade at extreme oversold/overbought with highest confidence.
"""

import time
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

print("Starting FAST Dr. Chen System (Extreme Selectivity)...")
start_time = time.time()

# =============================================================================
# CONFIG - ULTRA SELECTIVE
# =============================================================================
MIN_EDGE_MULT = 2.5    # Very high edge required
MIN_SIGMA = 0.00008    # Higher vol floor
MAX_SIGMA = 0.0015     # Lower vol ceiling
ENV_THRESHOLD = 0.80   # Very high confidence

HORIZONS = {"5m": 5, "15m": 15, "30m": 30}
K1, K2 = 0.8, 2.0      # Tight SL, 2.5:1 R:R

DATA_DIR = Path(__file__).parent.parent / "Data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

TRAIN_END = "2023-12-31"
VAL_END = "2024-06-30"

# =============================================================================
# DATA LOADING
# =============================================================================
print("\n[1/6] Loading data...")

def load_year(year):
    mp = DATA_DIR / "ohlcv_minute" / f"XAUUSD_minute_{year}.parquet"
    qp = DATA_DIR / "quotes" / f"XAUUSD_quotes_{year}.parquet"
    if not mp.exists() or not qp.exists():
        return None
    
    m = pd.read_parquet(mp)
    q = pd.read_parquet(qp)
    
    if "timestamp" in m.columns:
        m["timestamp"] = pd.to_datetime(m["timestamp"], utc=True)
        m = m.set_index("timestamp")
    if "timestamp" in q.columns:
        q["timestamp"] = pd.to_datetime(q["timestamp"], utc=True)
        q = q.set_index("timestamp")
    
    m = m.reset_index()
    q = q.reset_index()
    
    df = pd.merge_asof(
        m.sort_values("timestamp"),
        q.sort_values("timestamp")[["timestamp", "bid_price", "ask_price"]],
        on="timestamp", direction="backward"
    ).set_index("timestamp")
    
    return df.dropna(subset=["bid_price", "ask_price"])

dfs = []
for year in [2020, 2021, 2022, 2023, 2024, 2025]:
    print(f"  {year}...", end=" ", flush=True)
    d = load_year(year)
    if d is not None:
        print(f"{len(d):,} rows")
        dfs.append(d)
    else:
        print("skip")

df = pd.concat(dfs).sort_index()
df = df[~df.index.duplicated(keep='first')]
print(f"Total: {len(df):,} rows")

# =============================================================================
# FEATURES
# =============================================================================
print("\n[2/6] Building features...")

df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
df["spread"] = df["ask_price"] - df["bid_price"]
df["spread_pct"] = df["spread"] / df["mid"]

df["log_ret"] = np.log(df["mid"] / df["mid"].shift(1))
past_ret = df["log_ret"].shift(1)

for n in [5, 15, 30, 60]:
    df[f"sigma_{n}"] = past_ret.rolling(n, min_periods=n//2).std()
df["sigma"] = df["sigma_60"]

df["sigma_slope"] = df["sigma"].diff(5)
df["sigma_increasing"] = (df["sigma_slope"] > 0).astype(int)
df["sigma_decreasing"] = (df["sigma_slope"] < 0).astype(int)

# Regime
past_sigma = df["sigma"].shift(1)
med_sigma = past_sigma.rolling(60, min_periods=30).median()
df["vol_regime_ratio"] = np.where(med_sigma > 0, past_sigma / med_sigma, 1)
df["vol_regime_high"] = (df["vol_regime_ratio"] > 1.5).astype(int)
df["vol_regime_low"] = (df["vol_regime_ratio"] < 0.7).astype(int)
df["vol_regime_normal"] = ((df["vol_regime_ratio"] >= 0.7) & (df["vol_regime_ratio"] <= 1.5)).astype(int)

past_spread = df["spread_pct"].shift(1)
med_spread = past_spread.rolling(60, min_periods=30).median()
df["spread_regime_ratio"] = np.where(med_spread > 0, past_spread / med_spread, 1)
df["liquidity_tight"] = (df["spread_regime_ratio"] < 0.8).astype(int)
df["liquidity_wide"] = (df["spread_regime_ratio"] > 1.5).astype(int)

# Mean reversion features
past_mid = df["mid"].shift(1)

for n in [10, 20, 30, 60]:
    rm = past_mid.rolling(n, min_periods=n//2).mean()
    rs = past_mid.rolling(n, min_periods=n//2).std()
    df[f"zscore_{n}"] = np.where(rs > 0, (past_mid - rm) / rs, 0)

# RSI
up_moves = df["log_ret"].clip(lower=0)
down_moves = (-df["log_ret"]).clip(lower=0)
avg_up = up_moves.shift(1).rolling(14, min_periods=7).mean()
avg_down = down_moves.shift(1).rolling(14, min_periods=7).mean()
rs = np.where(avg_down > 0, avg_up / avg_down, 1)
df["rsi"] = 100 - (100 / (1 + rs))

# Momentum
df["momentum_5"] = past_mid.pct_change(5)
df["momentum_15"] = past_mid.pct_change(15)
df["momentum_60"] = past_mid.pct_change(60)

# Session
hour = df.index.hour
df["is_london"] = ((hour >= 8) & (hour < 16)).astype(int)
df["is_ny"] = ((hour >= 13) & (hour < 22)).astype(int)
df["is_overlap"] = ((hour >= 13) & (hour < 16)).astype(int)
df["session_quality"] = df["is_overlap"] * 2 + df["is_london"] + df["is_ny"]
df["hour_of_day"] = hour
df["day_of_week"] = df.index.dayofweek

if "vwap" in df.columns:
    past_vwap = df["vwap"].shift(1)
    df["vwap_deviation"] = np.where(past_vwap > 0, (past_mid - past_vwap) / past_vwap, 0)
else:
    df["vwap_deviation"] = 0

print(f"Features: {len(df.columns)} columns")

# =============================================================================
# LABELING
# =============================================================================
print("\n[3/6] Generating labels...")

mid_arr = df["mid"].values
spread_arr = df["spread_pct"].values
sigma_arr = df["sigma"].values
n = len(df)

for horizon_name, horizon_mins in HORIZONS.items():
    print(f"  {horizon_name}...", end=" ", flush=True)
    
    labels = np.zeros(n, dtype=int)
    valid_end = n - horizon_mins
    
    for i in range(0, valid_end):
        if np.isnan(mid_arr[i]) or mid_arr[i] <= 0:
            continue
        if np.isnan(sigma_arr[i]) or sigma_arr[i] < MIN_SIGMA:
            continue
        
        future = mid_arr[i+1:i+1+horizon_mins]
        if len(future) == 0:
            continue
        
        returns = future / mid_arr[i] - 1
        mu_plus = np.nanmax(returns)
        mu_minus = np.nanmin(returns)
        movement = max(mu_plus, abs(mu_minus))
        
        cost = spread_arr[i] + 0.0001 if not np.isnan(spread_arr[i]) else 0.0002
        if cost <= 0:
            cost = 0.0001
        
        if movement / cost >= MIN_EDGE_MULT:
            labels[i] = 1
    
    df[f"env_{horizon_mins}m"] = labels
    good = labels.sum()
    print(f"Good: {good:,} ({100*good/valid_end:.1f}%)")

# =============================================================================
# TRAINING
# =============================================================================
print("\n[4/6] Training models...")

FEATURES = [
    "sigma_5", "sigma_15", "sigma_30", "sigma_60", "sigma_increasing", "sigma_decreasing",
    "vol_regime_ratio", "vol_regime_high", "vol_regime_low", "vol_regime_normal",
    "spread_regime_ratio", "liquidity_tight", "liquidity_wide",
    "spread_pct", "zscore_10", "zscore_20", "zscore_30", "zscore_60",
    "rsi", "momentum_5", "momentum_15", "momentum_60",
    "is_london", "is_ny", "is_overlap", "session_quality",
    "hour_of_day", "day_of_week", "vwap_deviation"
]
available_features = [f for f in FEATURES if f in df.columns]

train_end = pd.Timestamp(TRAIN_END, tz='UTC')
val_end = pd.Timestamp(VAL_END, tz='UTC')

train_df = df[df.index <= train_end]
val_df = df[(df.index > train_end) & (df.index <= val_end)]
test_df = df[df.index > val_end]

print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

models = {}
for horizon_name, horizon_mins in HORIZONS.items():
    label_col = f"env_{horizon_mins}m"
    
    train_work = train_df[available_features + [label_col]].dropna()
    val_work = val_df[available_features + [label_col]].dropna()
    
    X_train = train_work[available_features].values
    y_train = train_work[label_col].values.astype(int)
    X_val = val_work[available_features].values
    y_val = val_work[label_col].values.astype(int)
    
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos = n_neg / n_pos if n_pos > 0 else 1
    
    print(f"  {horizon_name}: {len(y_train):,} samples, {n_pos:,} good ({100*n_pos/len(y_train):.1f}%)")
    
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=4,
        learning_rate=0.1,
        n_estimators=200,
        min_child_weight=20,
        scale_pos_weight=scale_pos,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    print(f"    AUC: Train={train_auc:.3f}, Val={val_auc:.3f}")
    
    models[horizon_name] = model
    joblib.dump({"model": model, "features": available_features}, MODEL_DIR / f"model_{horizon_name}.pkl")

# =============================================================================
# SIGNALS - EXTREME CONDITIONS ONLY
# =============================================================================
print("\n[5/6] Generating signals (EXTREME conditions only)...")

test_work = test_df.copy()
X_test = test_work[available_features].fillna(0).values

for horizon_name in HORIZONS:
    model = models[horizon_name]
    proba = model.predict_proba(X_test)[:, 1]
    test_work[f"env_score_{horizon_name}"] = proba
    test_work[f"env_good_{horizon_name}"] = (proba >= ENV_THRESHOLD).astype(int)

# EXTREME CONDITIONS for direction
zscore_20 = test_work["zscore_20"].values
zscore_60 = test_work["zscore_60"].values
rsi = test_work["rsi"].values
vol_normal = test_work["vol_regime_normal"].values
mom_60 = test_work["momentum_60"].values

# Use longer-term mean reversion with momentum confirmation
# LONG: extreme oversold + positive momentum reversal starting
direction = np.zeros(len(test_work), dtype=int)

# EXTREME LONG: very oversold on both short and long term
extreme_oversold = (
    (zscore_20 < -2.0) &     # Very oversold short-term
    (zscore_60 < -1.5) &     # Also oversold longer-term
    (rsi < 30) &             # RSI extreme
    (vol_normal == 1)        # Normal vol regime
)
direction = np.where(extreme_oversold, 1, direction)

# EXTREME SHORT: very overbought on both short and long term
extreme_overbought = (
    (zscore_20 > 2.0) &      # Very overbought short-term
    (zscore_60 > 1.5) &      # Also overbought longer-term
    (rsi > 70) &             # RSI extreme
    (vol_normal == 1)        # Normal vol regime
)
direction = np.where(extreme_overbought, -1, direction)

test_work["direction"] = direction

# Very strict filters
sigma_test = test_work["sigma"].values
spread_test = test_work["spread_pct"].values
session_test = test_work["session_quality"].values

filter_mask = (
    (sigma_test >= MIN_SIGMA) & (sigma_test <= MAX_SIGMA) &
    (spread_test <= 0.0003) &   # Very tight spread only
    (session_test >= 3) &       # Only best sessions
    (~np.isnan(sigma_test)) & (~np.isnan(spread_test))
)

for horizon_name in HORIZONS:
    env_good = test_work[f"env_good_{horizon_name}"].values
    signal = np.where((env_good == 1) & filter_mask & (direction != 0), direction, 0)
    test_work[f"signal_{horizon_name}"] = signal

for horizon_name in HORIZONS:
    sig = test_work[f"signal_{horizon_name}"]
    n_long = (sig == 1).sum()
    n_short = (sig == -1).sum()
    total = n_long + n_short
    print(f"  {horizon_name}: {total:,} signals (L:{n_long:,}, S:{n_short:,})")

# =============================================================================
# BACKTEST
# =============================================================================
print("\n[6/6] Running backtest...")

mid_test = test_work["mid"].values
sigma_test = test_work["sigma"].values
spread_test = test_work["spread_pct"].values
n_test = len(test_work)

results = {}

for horizon_name, horizon_mins in HORIZONS.items():
    signal = test_work[f"signal_{horizon_name}"].values
    
    trade_indices = np.where(signal != 0)[0]
    trade_indices = trade_indices[trade_indices < n_test - horizon_mins - 1]
    
    if len(trade_indices) == 0:
        print(f"  {horizon_name}: No trades!")
        results[horizon_name] = {"n_trades": 0}
        continue
    
    pnl_list, r_list, directions, wins = [], [], [], []
    
    for i in trade_indices:
        entry_mid = mid_test[i]
        entry_sigma = sigma_test[i]
        entry_spread = spread_test[i]
        trade_dir = signal[i]
        
        if np.isnan(entry_mid) or np.isnan(entry_sigma) or entry_sigma <= 0:
            continue
        
        entry_price = mid_test[i + 1]
        if np.isnan(entry_price):
            continue
        
        sl_ret = K1 * entry_sigma
        tp_ret = K2 * entry_sigma + entry_spread
        
        if trade_dir == 1:
            sl_price = entry_price * (1 - sl_ret)
            tp_price = entry_price * (1 + tp_ret)
        else:
            sl_price = entry_price * (1 + sl_ret)
            tp_price = entry_price * (1 - tp_ret)
        
        exit_price = None
        for j in range(i + 2, min(i + 2 + horizon_mins, n_test)):
            p = mid_test[j]
            if np.isnan(p):
                continue
            if trade_dir == 1:
                if p <= sl_price:
                    exit_price = sl_price
                    break
                if p >= tp_price:
                    exit_price = tp_price
                    break
            else:
                if p >= sl_price:
                    exit_price = sl_price
                    break
                if p <= tp_price:
                    exit_price = tp_price
                    break
        
        if exit_price is None:
            exit_price = mid_test[min(i + 1 + horizon_mins, n_test - 1)]
        
        if np.isnan(exit_price):
            continue
        
        ret = (exit_price / entry_price - 1) if trade_dir == 1 else (entry_price / exit_price - 1)
        pnl = ret - entry_spread
        r = pnl / sl_ret if sl_ret > 0 else 0
        
        pnl_list.append(pnl)
        r_list.append(r)
        directions.append(trade_dir)
        wins.append(pnl > 0)
    
    if len(pnl_list) == 0:
        results[horizon_name] = {"n_trades": 0}
        continue
    
    pnl_arr = np.array(pnl_list)
    r_arr = np.array(r_list)
    dir_arr = np.array(directions)
    win_arr = np.array(wins)
    
    n_trades = len(pnl_arr)
    n_long = (dir_arr == 1).sum()
    n_short = (dir_arr == -1).sum()
    
    cum_pnl = pnl_arr.cumsum()
    max_dd = (np.maximum.accumulate(cum_pnl) - cum_pnl).max()
    
    results[horizon_name] = {
        "n_trades": n_trades,
        "total_r": r_arr.sum(),
        "avg_r": r_arr.mean(),
        "win_rate": win_arr.mean(),
        "long_count": n_long,
        "short_count": n_short,
        "long_pct": 100 * n_long / n_trades if n_trades > 0 else 0,
        "short_pct": 100 * n_short / n_trades if n_trades > 0 else 0,
        "long_win": win_arr[dir_arr == 1].mean() if n_long > 0 else 0,
        "short_win": win_arr[dir_arr == -1].mean() if n_short > 0 else 0,
        "total_pnl_pct": pnl_arr.sum() * 100,
        "max_dd_pct": max_dd * 100,
    }

# =============================================================================
# RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS - Extreme Selectivity")
print("=" * 70)

for h, m in results.items():
    if m["n_trades"] == 0:
        print(f"\n{h}: No trades")
        continue
    
    print(f"\n{h}:")
    print(f"  Trades: {m['n_trades']:,} (L:{m['long_count']}, S:{m['short_count']})")
    print(f"  Win Rate: {m['win_rate']:.1%}")
    print(f"  Avg R: {m['avg_r']:.3f}, Total R: {m['total_r']:.2f}")
    print(f"  P&L: {m['total_pnl_pct']:.2f}%, Max DD: {m['max_dd_pct']:.2f}%")
    print(f"  Long Win: {m['long_win']:.1%}, Short Win: {m['short_win']:.1%}")
    print(f"  Status: {'✓ PROFITABLE' if m['avg_r'] > 0 else '⚠ LOSING'}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for h, m in results.items():
    if m["n_trades"] == 0:
        print(f"  {h}: No trades")
    else:
        prof = "✓" if m['avg_r'] > 0 else "✗"
        print(f"  {h}: R={m['total_r']:+.2f}, WR={m['win_rate']:.1%}, Profit {prof}")

elapsed = time.time() - start_time
print(f"\nCompleted in {elapsed:.1f} seconds")
