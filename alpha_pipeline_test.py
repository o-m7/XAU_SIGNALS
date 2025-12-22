#!/usr/bin/env python3
"""
Alpha-Based Signal Pipeline - Academically Validated Approach.

This implements the 4 validated alpha sources:
A. Order Flow Imbalance (Hasbrouck/Bouchaud)
B. Microprice Dislocation (Biais/Hillion)
C. Liquidity Shock Breakout (Gatheral)
D. Volatility Expansion Breakout (Andersen/Bollerslev)

The model predicts WHEN edge exists (alpha active), not direction.
Direction comes from microstructure after alpha detection.
"""

import time
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score

print("=" * 70)
print("ALPHA-BASED SIGNAL PIPELINE")
print("Academically Validated Microstructure Approach")
print("=" * 70)

start_time = time.time()

# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = Path(__file__).parent.parent / "Data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

HORIZONS = {"5m": 5, "15m": 15, "30m": 30}
TRAIN_END = "2023-12-31"
VAL_END = "2024-06-30"

# Alpha thresholds
OFI_THRESHOLD = 0.3
MICRO_THRESHOLD = 0.0002
SPREAD_EXPANSION = 1.5
VOL_RATIO_THRESHOLD = 1.5
COST_FLOOR = 0.0003

# Signal generation
ALPHA_CONFIDENCE = 0.7  # Higher confidence required
K1, K2 = 0.8, 1.5  # Tighter SL, smaller TP for higher win rate

# =============================================================================
# DATA LOADING
# =============================================================================
print("\n[1/7] Loading data...")

def load_year(year):
    mp = DATA_DIR / "ohlcv_minute" / f"XAUUSD_minute_{year}.parquet"
    qp = DATA_DIR / "quotes" / f"XAUUSD_quotes_{year}.parquet"
    if not mp.exists() or not qp.exists():
        return None
    
    m = pd.read_parquet(mp)
    q = pd.read_parquet(qp)
    
    for d in [m, q]:
        if "timestamp" in d.columns:
            d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
            d.set_index("timestamp", inplace=True)
    
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
        print(f"{len(d):,}")
        dfs.append(d)
    else:
        print("skip")

df = pd.concat(dfs).sort_index()
df = df[~df.index.duplicated(keep='first')]
print(f"Total: {len(df):,} rows")

# =============================================================================
# MICROSTRUCTURE FEATURE ENGINEERING
# =============================================================================
print("\n[2/7] Engineering microstructure features...")

# Basic prices
df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
df["spread"] = df["ask_price"] - df["bid_price"]
df["spread_pct"] = df["spread"] / df["mid"]

# Log returns
df["log_ret"] = np.log(df["mid"] / df["mid"].shift(1))

# Multi-scale volatility (Andersen/Bollerslev)
past_ret = df["log_ret"].shift(1)
df["sigma_5"] = past_ret.rolling(5, min_periods=3).std()
df["sigma_15"] = past_ret.rolling(15, min_periods=8).std()
df["sigma_30"] = past_ret.rolling(30, min_periods=15).std()
df["sigma_60"] = past_ret.rolling(60, min_periods=30).std()
df["sigma"] = df["sigma_60"]

# Volatility ratio (key for Alpha D)
df["vol_ratio_5_30"] = np.where(df["sigma_30"] > 0, df["sigma_5"] / df["sigma_30"], 1)
df["vol_ratio_5_60"] = np.where(df["sigma_60"] > 0, df["sigma_5"] / df["sigma_60"], 1)
df["vol_expanding"] = (df["vol_ratio_5_30"] > 1.2).astype(int)

# Normalized spread (key for Alpha C - Gatheral)
past_spread = df["spread_pct"].shift(1)
spread_median = past_spread.rolling(60, min_periods=30).median()
df["spread_normalized"] = np.where(spread_median > 0, past_spread / spread_median, 1)
df["spread_expanding"] = (df["spread_normalized"] > SPREAD_EXPANSION).astype(int)

# OFI Proxy (key for Alpha A - Hasbrouck/Bouchaud)
# Without trade data, proxy using price momentum + spread dynamics
spread_change = df["spread_pct"].diff()
price_mom = df["log_ret"].rolling(3, min_periods=1).sum()
df["ofi_proxy_raw"] = price_mom - spread_change.shift(1) * 10

# Normalize OFI
ofi_std = df["ofi_proxy_raw"].shift(1).rolling(60, min_periods=30).std()
df["ofi_proxy"] = np.where(ofi_std > 0, df["ofi_proxy_raw"] / ofi_std, 0)
df["ofi_positive"] = (df["ofi_proxy"] > OFI_THRESHOLD).astype(int)
df["ofi_negative"] = (df["ofi_proxy"] < -OFI_THRESHOLD).astype(int)

# Fair value / Microprice proxy (key for Alpha B - Biais/Hillion)
if "vwap" in df.columns:
    df["fair_value"] = df["vwap"].shift(1)
else:
    df["fair_value"] = df["mid"].shift(1).rolling(20, min_periods=10).mean()

df["micro_dislocation"] = np.where(
    df["fair_value"] > 0,
    (df["mid"] - df["fair_value"]) / df["fair_value"],
    0
)
df["dislocated_high"] = (df["micro_dislocation"] > MICRO_THRESHOLD).astype(int)
df["dislocated_low"] = (df["micro_dislocation"] < -MICRO_THRESHOLD).astype(int)

# Regime features
past_sigma = df["sigma"].shift(1)
sigma_median = past_sigma.rolling(120, min_periods=60).median()
df["vol_regime"] = np.where(sigma_median > 0, past_sigma / sigma_median, 1)
df["vol_high"] = (df["vol_regime"] > 1.5).astype(int)
df["vol_low"] = (df["vol_regime"] < 0.7).astype(int)
df["vol_normal"] = ((df["vol_regime"] >= 0.7) & (df["vol_regime"] <= 1.5)).astype(int)

# VWAP distance and slope
if "vwap" not in df.columns:
    past_close = df["close"].shift(1) if "close" in df.columns else df["mid"].shift(1)
    past_vol = df["volume"].shift(1) if "volume" in df.columns else pd.Series(1, index=df.index)
    pv = past_close * past_vol
    df["vwap"] = pv.rolling(60, min_periods=30).sum() / past_vol.rolling(60, min_periods=30).sum()

df["vwap_dist"] = np.where(df["vwap"] > 0, (df["mid"] - df["vwap"]) / df["vwap"], 0)
df["vwap_slope"] = df["vwap"].pct_change(5)

# Session encoding
hour = df.index.hour
df["is_london"] = ((hour >= 8) & (hour < 16)).astype(int)
df["is_ny"] = ((hour >= 13) & (hour < 22)).astype(int)
df["is_overlap"] = ((hour >= 13) & (hour < 16)).astype(int)
df["session_quality"] = df["is_overlap"] * 2 + df["is_london"] + df["is_ny"]
df["hour"] = hour
df["dow"] = df.index.dayofweek

# Price variance ratio (Hurst proxy)
ret_5 = df["log_ret"].rolling(5, min_periods=3).var()
ret_20 = df["log_ret"].rolling(20, min_periods=10).var()
df["variance_ratio"] = np.where(ret_20 > 0, ret_5 / ret_20 * 4, 1)  # Should be ~1 for random walk

print(f"Features: {len(df.columns)} columns")

# =============================================================================
# ALPHA SIGNAL LABELING
# =============================================================================
print("\n[3/7] Generating alpha signal labels...")

mid_arr = df["mid"].values
spread_pct_arr = df["spread_pct"].values
sigma_arr = df["sigma"].values
ofi_arr = df["ofi_proxy"].values
fair_value_arr = df["fair_value"].values
spread_norm_arr = df["spread_normalized"].values
vol_ratio_arr = df["vol_ratio_5_30"].values
n = len(df)

for horizon_name, horizon_mins in HORIZONS.items():
    print(f"\n  {horizon_name} horizon:")
    
    alpha_a = np.zeros(n, dtype=int)  # Order Flow
    alpha_b = np.zeros(n, dtype=int)  # Microprice
    alpha_c = np.zeros(n, dtype=int)  # Liquidity Shock
    alpha_d = np.zeros(n, dtype=int)  # Vol Breakout
    
    valid_end = n - horizon_mins
    
    for i in range(60, valid_end):
        mid_t = mid_arr[i]
        if np.isnan(mid_t) or mid_t <= 0:
            continue
        
        sigma_t = sigma_arr[i]
        if np.isnan(sigma_t) or sigma_t <= 0:
            sigma_t = 0.0001
        
        future_mids = mid_arr[i+1:i+1+horizon_mins]
        if len(future_mids) == 0:
            continue
        
        future_return = future_mids[-1] / mid_t - 1
        max_up = np.nanmax(future_mids) / mid_t - 1
        max_down = 1 - np.nanmin(future_mids) / mid_t
        max_move = max(max_up, max_down)
        
        # ALPHA A: Order Flow Imbalance
        ofi = ofi_arr[i]
        if not np.isnan(ofi) and abs(ofi) > OFI_THRESHOLD:
            same_sign = (ofi > 0 and future_return > 0) or (ofi < 0 and future_return < 0)
            if same_sign and abs(future_return) > COST_FLOOR:
                alpha_a[i] = 1
        
        # ALPHA B: Microprice Reversion
        fv = fair_value_arr[i]
        if not np.isnan(fv) and fv > 0:
            dislocation = (mid_t - fv) / fv
            if abs(dislocation) > MICRO_THRESHOLD:
                # Check reversion
                current_dist = abs(mid_t - fv)
                future_dists = np.abs(future_mids - fv)
                if np.nanmin(future_dists) < current_dist * 0.6:
                    alpha_b[i] = 1
        
        # ALPHA C: Liquidity Shock
        spread_norm = spread_norm_arr[i]
        vol_ratio = vol_ratio_arr[i]
        if not np.isnan(spread_norm) and spread_norm > SPREAD_EXPANSION:
            if not np.isnan(vol_ratio) and vol_ratio > 1.2:
                if max_move > 0.5 * sigma_t:
                    alpha_c[i] = 1
        
        # ALPHA D: Volatility Breakout
        if not np.isnan(vol_ratio) and vol_ratio > VOL_RATIO_THRESHOLD:
            if max_move > 0.5 * sigma_t:
                alpha_d[i] = 1
    
    df[f"alpha_a_{horizon_mins}m"] = alpha_a
    df[f"alpha_b_{horizon_mins}m"] = alpha_b
    df[f"alpha_c_{horizon_mins}m"] = alpha_c
    df[f"alpha_d_{horizon_mins}m"] = alpha_d
    df[f"any_alpha_{horizon_mins}m"] = ((alpha_a + alpha_b + alpha_c + alpha_d) > 0).astype(int)
    
    print(f"    Alpha A (Order Flow):   {alpha_a.sum():,} ({100*alpha_a.mean():.1f}%)")
    print(f"    Alpha B (Microprice):   {alpha_b.sum():,} ({100*alpha_b.mean():.1f}%)")
    print(f"    Alpha C (Liq Shock):    {alpha_c.sum():,} ({100*alpha_c.mean():.1f}%)")
    print(f"    Alpha D (Vol Break):    {alpha_d.sum():,} ({100*alpha_d.mean():.1f}%)")
    print(f"    Any Alpha:              {df[f'any_alpha_{horizon_mins}m'].sum():,}")

# =============================================================================
# MODEL TRAINING - MULTI-LABEL CLASSIFIER
# =============================================================================
print("\n[4/7] Training multi-label alpha classifiers...")

FEATURES = [
    "sigma_5", "sigma_15", "sigma_30", "sigma_60",
    "vol_ratio_5_30", "vol_ratio_5_60", "vol_expanding",
    "spread_pct", "spread_normalized", "spread_expanding",
    "ofi_proxy", "ofi_positive", "ofi_negative",
    "micro_dislocation", "dislocated_high", "dislocated_low",
    "vol_regime", "vol_high", "vol_low", "vol_normal",
    "vwap_dist", "vwap_slope", "variance_ratio",
    "is_london", "is_ny", "is_overlap", "session_quality", "hour", "dow"
]
available = [f for f in FEATURES if f in df.columns]
print(f"Using {len(available)} features")

train_end = pd.Timestamp(TRAIN_END, tz='UTC')
val_end = pd.Timestamp(VAL_END, tz='UTC')

train_df = df[df.index <= train_end]
val_df = df[(df.index > train_end) & (df.index <= val_end)]
test_df = df[df.index > val_end]

print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

models = {}
for horizon_name, horizon_mins in HORIZONS.items():
    print(f"\n  {horizon_name} models:")
    models[horizon_name] = {}
    
    for alpha in ["a", "b", "c", "d", "any"]:
        label_col = f"alpha_{alpha}_{horizon_mins}m" if alpha != "any" else f"any_alpha_{horizon_mins}m"
        
        train_work = train_df[available + [label_col]].dropna()
        val_work = val_df[available + [label_col]].dropna()
        
        X_train = train_work[available].values
        y_train = train_work[label_col].values.astype(int)
        X_val = val_work[available].values
        y_val = val_work[label_col].values.astype(int)
        
        n_pos = (y_train == 1).sum()
        if n_pos < 100:
            print(f"    Alpha {alpha.upper()}: Skipped (too few positives)")
            continue
        
        scale_pos = (y_train == 0).sum() / n_pos if n_pos > 0 else 1
        
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            max_depth=4,
            learning_rate=0.1,
            n_estimators=150,
            min_child_weight=20,
            scale_pos_weight=scale_pos,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        
        alpha_name = {"a": "OrdFlow", "b": "MicroP", "c": "LiqShk", "d": "VolBrk", "any": "AnyAlpha"}[alpha]
        print(f"    {alpha_name}: AUC Train={train_auc:.3f}, Val={val_auc:.3f}")
        
        models[horizon_name][alpha] = model
        joblib.dump(model, MODEL_DIR / f"alpha_{alpha}_{horizon_name}.pkl")

# =============================================================================
# SIGNAL GENERATION - ALPHA + DIRECTION
# =============================================================================
print("\n[5/7] Generating signals on test data...")

test_work = test_df.copy()
X_test = test_work[available].fillna(0).values

for horizon_name, horizon_mins in HORIZONS.items():
    # Predict alpha probabilities
    for alpha, model in models[horizon_name].items():
        proba = model.predict_proba(X_test)[:, 1]
        test_work[f"p_alpha_{alpha}_{horizon_name}"] = proba
    
    # Combined alpha score (max of individual alphas)
    alpha_cols = [f"p_alpha_{a}_{horizon_name}" for a in ["a", "b", "c", "d"] if f"p_alpha_{a}_{horizon_name}" in test_work.columns]
    if alpha_cols:
        test_work[f"alpha_score_{horizon_name}"] = test_work[alpha_cols].max(axis=1)
    else:
        test_work[f"alpha_score_{horizon_name}"] = test_work[f"p_alpha_any_{horizon_name}"]

# DIRECTION FROM MICROSTRUCTURE ONLY - SYMMETRIC LOGIC
# Use OFI sign directly for direction (momentum-based)
# This is symmetric by construction

ofi = test_work["ofi_proxy"].values
vol_normal = test_work["vol_normal"].values

direction = np.zeros(len(test_work), dtype=int)

# Simple symmetric logic: follow the order flow
# Long when OFI strongly positive
# Short when OFI strongly negative
# Use HIGHER threshold for stronger signals
OFI_DIR_THRESHOLD = 1.0

direction = np.where(ofi > OFI_DIR_THRESHOLD, 1, direction)
direction = np.where(ofi < -OFI_DIR_THRESHOLD, -1, direction)

test_work["direction"] = direction

# Filters
sigma_test = test_work["sigma"].values
spread_test = test_work["spread_pct"].values
session_test = test_work["session_quality"].values

filter_mask = (
    (sigma_test >= 0.00005) & (sigma_test <= 0.002) &
    (spread_test <= 0.0008) &
    (session_test >= 2) &
    (~np.isnan(sigma_test))
)

# Final signals: alpha active + direction determined + filters pass
for horizon_name in HORIZONS:
    alpha_score = test_work[f"alpha_score_{horizon_name}"].values
    signal = np.where(
        (alpha_score >= ALPHA_CONFIDENCE) & filter_mask & (direction != 0),
        direction,
        0
    )
    test_work[f"signal_{horizon_name}"] = signal
    
    n_long = (signal == 1).sum()
    n_short = (signal == -1).sum()
    print(f"  {horizon_name}: {n_long + n_short:,} signals (L:{n_long:,}, S:{n_short:,})")

# =============================================================================
# BACKTESTING
# =============================================================================
print("\n[6/7] Running backtest...")

mid_test = test_work["mid"].values
sigma_test = test_work["sigma"].values
spread_test = test_work["spread_pct"].values
n_test = len(test_work)

results = {}

for horizon_name, horizon_mins in HORIZONS.items():
    signal = test_work[f"signal_{horizon_name}"].values
    alpha_score = test_work[f"alpha_score_{horizon_name}"].values
    
    trade_indices = np.where(signal != 0)[0]
    trade_indices = trade_indices[trade_indices < n_test - horizon_mins - 1]
    
    if len(trade_indices) == 0:
        results[horizon_name] = {"n_trades": 0}
        continue
    
    trades = {"pnl": [], "r": [], "dir": [], "alpha": [], "alpha_type": []}
    
    for i in trade_indices:
        entry_sigma = sigma_test[i]
        entry_spread = spread_test[i]
        trade_dir = signal[i]
        
        if np.isnan(mid_test[i]) or np.isnan(entry_sigma) or entry_sigma <= 0:
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
        
        trades["pnl"].append(pnl)
        trades["r"].append(r)
        trades["dir"].append(trade_dir)
        trades["alpha"].append(alpha_score[i])
    
    if len(trades["pnl"]) == 0:
        results[horizon_name] = {"n_trades": 0}
        continue
    
    pnl_arr = np.array(trades["pnl"])
    r_arr = np.array(trades["r"])
    dir_arr = np.array(trades["dir"])
    
    n_trades = len(pnl_arr)
    n_long = (dir_arr == 1).sum()
    n_short = (dir_arr == -1).sum()
    
    results[horizon_name] = {
        "n_trades": n_trades,
        "total_r": r_arr.sum(),
        "avg_r": r_arr.mean(),
        "win_rate": (pnl_arr > 0).mean(),
        "long_count": n_long,
        "short_count": n_short,
        "long_pct": 100 * n_long / n_trades,
        "short_pct": 100 * n_short / n_trades,
        "long_win": (pnl_arr[dir_arr == 1] > 0).mean() if n_long > 0 else 0,
        "short_win": (pnl_arr[dir_arr == -1] > 0).mean() if n_short > 0 else 0,
        "total_pnl": pnl_arr.sum() * 100,
        "max_dd": (np.maximum.accumulate(pnl_arr.cumsum()) - pnl_arr.cumsum()).max() * 100,
    }

# =============================================================================
# RESULTS
# =============================================================================
print("\n" + "=" * 70)
print("RESULTS - Alpha-Based Pipeline")
print("=" * 70)

for h, m in results.items():
    if m["n_trades"] == 0:
        print(f"\n{h}: No trades")
        continue
    
    print(f"\n{h} HORIZON:")
    print(f"  Trades: {m['n_trades']:,} (L:{m['long_count']:,}, S:{m['short_count']:,})")
    print(f"  Balance: {m['long_pct']:.1f}% L / {m['short_pct']:.1f}% S")
    print(f"  Win Rate: {m['win_rate']:.1%} (L:{m['long_win']:.1%}, S:{m['short_win']:.1%})")
    print(f"  Avg R: {m['avg_r']:.3f}")
    print(f"  Total R: {m['total_r']:.2f}")
    print(f"  Total P&L: {m['total_pnl']:.2f}%")
    print(f"  Max DD: {m['max_dd']:.2f}%")
    
    status = "✓ PROFITABLE" if m['avg_r'] > 0 else "⚠ LOSING"
    balance = "✓ BALANCED" if 35 < m['long_pct'] < 65 else "⚠ IMBALANCED"
    print(f"  Status: {status}, {balance}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for h, m in results.items():
    if m["n_trades"] == 0:
        print(f"  {h}: No trades")
    else:
        prof = "✓" if m['avg_r'] > 0 else "✗"
        bal = "✓" if 35 < m['long_pct'] < 65 else "✗"
        print(f"  {h}: R={m['total_r']:+.2f}, WR={m['win_rate']:.1%}, Balance {bal} ({m['long_pct']:.0f}%L), Profit {prof}")

elapsed = time.time() - start_time
print(f"\nCompleted in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

