#!/usr/bin/env python3
"""
Test script for the Dr. Chen Environment-Based Signal System.

This script:
1. Loads multi-year data (2020-2025)
2. Builds features with regime detection
3. Creates BINARY environment labels (not directional)
4. Trains environment quality models (AUC-optimized)
5. Runs backtest with TWO-STAGE logic:
   - Stage 1: Model predicts environment quality
   - Stage 2: Microstructure determines direction

Expected outcomes:
- Balanced long/short signals (not 99% one direction)
- Environment accuracy > 50%
- Direction accuracy > 50%
- Positive total R (or at least not catastrophically negative)
"""

import time
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_EDGE_MULTIPLIER = 1.5
MICRO_SLIPPAGE_PCT = 0.0001
MIN_SIGMA_THRESHOLD = 0.00003
ENV_THRESHOLD = 0.5
IMBALANCE_THRESHOLD = 0.3

HORIZON_PARAMS = {
    "5m": {"minutes": 5, "k1": 1.5, "k2": 2.0, "min_edge_mult": 1.5},
    "15m": {"minutes": 15, "k1": 1.5, "k2": 2.5, "min_edge_mult": 1.8},
    "30m": {"minutes": 30, "k1": 1.5, "k2": 3.0, "min_edge_mult": 2.0},
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_minute_bars(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index().drop_duplicates()


def load_quotes(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index().drop_duplicates()


def align_minute_bars_with_quotes(minute_df: pd.DataFrame, quotes_df: pd.DataFrame) -> pd.DataFrame:
    minute_df = minute_df.reset_index()
    quotes_df = quotes_df.reset_index()
    merged = pd.merge_asof(
        minute_df.sort_values("timestamp"),
        quotes_df.sort_values("timestamp")[["timestamp", "bid_price", "ask_price", "bid_size", "ask_size"]],
        on="timestamp",
        direction="backward"
    )
    merged = merged.set_index("timestamp")
    merged["mid"] = (merged["bid_price"] + merged["ask_price"]) / 2
    merged["spread"] = merged["ask_price"] - merged["bid_price"]
    merged["spread_pct"] = merged["spread"] / merged["mid"]
    return merged.dropna(subset=["bid_price", "ask_price"])


def load_multi_year_data(minute_dir: str, quotes_dir: str, years: list) -> pd.DataFrame:
    dfs = []
    for year in years:
        minute_path = f"{minute_dir}/XAUUSD_minute_{year}.parquet"
        quotes_path = f"{quotes_dir}/XAUUSD_quotes_{year}.parquet"
        if not Path(minute_path).exists() or not Path(quotes_path).exists():
            print(f"  Skipping {year}: files not found")
            continue
        print(f"  Loading {year}...", end=" ")
        minute_df = load_minute_bars(minute_path)
        quotes_df = load_quotes(quotes_path)
        combined = align_minute_bars_with_quotes(minute_df, quotes_df)
        print(f"{len(combined):,} rows")
        dfs.append(combined)
    result = pd.concat(dfs, axis=0).sort_index()
    return result[~result.index.duplicated(keep='first')]


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Basic microstructure
    if "mid" not in df.columns:
        df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    if "spread" not in df.columns:
        df["spread"] = df["ask_price"] - df["bid_price"]
    if "spread_pct" not in df.columns:
        df["spread_pct"] = df["spread"] / df["mid"]
    
    total_size = df["bid_size"] + df["ask_size"]
    df["imbalance"] = np.where(total_size > 0, (df["bid_size"] - df["ask_size"]) / total_size, 0.0)
    
    # Log return and volatility
    df["log_ret"] = np.log(df["mid"] / df["mid"].shift(1))
    past_returns = df["log_ret"].shift(1)
    
    for n in [5, 15, 30, 60]:
        df[f"sigma_{n}"] = past_returns.rolling(window=n, min_periods=max(n//2, 3)).std()
    df["sigma"] = df["sigma_60"]
    
    df["sigma_slope"] = df["sigma"].diff(5)
    df["sigma_increasing"] = (df["sigma_slope"] > 0).astype(int)
    
    # Regime features
    past_sigma = df["sigma"].shift(1)
    rolling_median_sigma = past_sigma.rolling(60, min_periods=30).median()
    df["vol_regime_ratio"] = np.where(rolling_median_sigma > 0, past_sigma / rolling_median_sigma, 1.0)
    df["vol_regime_high"] = (df["vol_regime_ratio"] > 1.5).astype(int)
    df["vol_regime_low"] = (df["vol_regime_ratio"] < 0.7).astype(int)
    df["vol_regime_normal"] = ((df["vol_regime_ratio"] >= 0.7) & (df["vol_regime_ratio"] <= 1.5)).astype(int)
    
    # Trend regime
    past_mid = df["mid"].shift(1)
    past_close = df["close"].shift(1) if "close" in df.columns else past_mid
    past_volume = df["volume"].shift(1) if "volume" in df.columns else pd.Series(1, index=df.index)
    pv = past_close * past_volume
    rolling_pv = pv.rolling(60, min_periods=30).sum()
    rolling_vol = past_volume.rolling(60, min_periods=30).sum()
    df["rolling_vwap"] = np.where(rolling_vol > 0, rolling_pv / rolling_vol, past_close)
    df["vwap_deviation"] = np.where(df["rolling_vwap"] > 0, (past_mid - df["rolling_vwap"]) / df["rolling_vwap"], 0.0)
    df["trend_bullish"] = (df["vwap_deviation"] > 0.001).astype(int)
    df["trend_bearish"] = (df["vwap_deviation"] < -0.001).astype(int)
    df["trend_neutral"] = ((df["vwap_deviation"] >= -0.001) & (df["vwap_deviation"] <= 0.001)).astype(int)
    
    # Liquidity regime
    past_spread_pct = df["spread_pct"].shift(1)
    rolling_spread_median = past_spread_pct.rolling(60, min_periods=30).median()
    df["spread_regime_ratio"] = np.where(rolling_spread_median > 0, past_spread_pct / rolling_spread_median, 1.0)
    df["liquidity_tight"] = (df["spread_regime_ratio"] < 0.8).astype(int)
    df["liquidity_wide"] = (df["spread_regime_ratio"] > 1.5).astype(int)
    df["liquidity_normal"] = ((df["spread_regime_ratio"] >= 0.8) & (df["spread_regime_ratio"] <= 1.5)).astype(int)
    
    # Imbalance regime
    past_imbalance = df["imbalance"].shift(1)
    rolling_imb_std = past_imbalance.rolling(60, min_periods=30).std()
    df["imbalance_zscore"] = np.where(rolling_imb_std > 0, past_imbalance / rolling_imb_std, 0.0)
    df["imbalance_buy_pressure"] = (df["imbalance_zscore"] > 1.0).astype(int)
    df["imbalance_sell_pressure"] = (df["imbalance_zscore"] < -1.0).astype(int)
    df["imbalance_neutral"] = ((df["imbalance_zscore"] >= -1.0) & (df["imbalance_zscore"] <= 1.0)).astype(int)
    
    # Direction indicators
    df["imbalance_ma5"] = past_imbalance.rolling(5, min_periods=2).mean()
    df["imbalance_ma15"] = past_imbalance.rolling(15, min_periods=5).mean()
    df["momentum_5"] = past_mid.pct_change(5)
    df["momentum_15"] = past_mid.pct_change(15)
    
    # Session features
    hour = df.index.hour
    df["is_asia"] = ((hour >= 0) & (hour < 8)).astype(int)
    df["is_london"] = ((hour >= 8) & (hour < 16)).astype(int)
    df["is_ny"] = ((hour >= 13) & (hour < 22)).astype(int)
    df["is_overlap"] = ((hour >= 13) & (hour < 16)).astype(int)
    df["session_quality"] = df["is_overlap"] * 2 + df["is_london"] + df["is_ny"]
    
    # Time features
    df["minute_of_day"] = df.index.hour * 60 + df.index.minute
    df["hour_of_day"] = df.index.hour
    df["day_of_week"] = df.index.dayofweek
    
    # Spread dynamics
    past_spread = df["spread_pct"].shift(1)
    rolling_mean = past_spread.rolling(60, min_periods=30).mean()
    rolling_std = past_spread.rolling(60, min_periods=30).std()
    df["spread_zscore"] = np.where(rolling_std > 0, (past_spread - rolling_mean) / rolling_std, 0.0)
    df["spread_expanding"] = (df["spread_zscore"] > 1.5).astype(int)
    
    # Environment quality
    past_high = df["high"].shift(1) if "high" in df.columns else past_mid
    past_low = df["low"].shift(1) if "low" in df.columns else past_mid
    tr = past_high - past_low
    df["atr_20"] = tr.rolling(20, min_periods=10).mean()
    df["atr_spread_ratio"] = np.where(df["spread_pct"].shift(1) > 0, (df["atr_20"] / past_mid) / df["spread_pct"].shift(1), 0.0)
    
    return df


def get_environment_model_features() -> list:
    return [
        "sigma_5", "sigma_15", "sigma_30", "sigma_60", "sigma", "sigma_increasing",
        "vol_regime_ratio", "vol_regime_high", "vol_regime_low", "vol_regime_normal",
        "spread_regime_ratio", "liquidity_tight", "liquidity_wide", "liquidity_normal",
        "atr_spread_ratio",
        "is_asia", "is_london", "is_ny", "is_overlap", "session_quality",
        "minute_of_day", "hour_of_day", "day_of_week",
        "spread_pct", "spread_zscore", "spread_expanding",
    ]


# =============================================================================
# LABELING (Binary Environment)
# =============================================================================

def label_environment_for_horizon(df: pd.DataFrame, horizon_minutes: int, min_edge_mult: float) -> pd.Series:
    n = len(df)
    labels = np.zeros(n, dtype=int)
    spread_pct_arr = df["spread_pct"].values
    sigma_arr = df["sigma"].values if "sigma" in df.columns else np.ones(n) * 0.0001
    mid_arr = df["mid"].values
    
    valid_end = n - horizon_minutes
    
    for i in range(valid_end):
        sigma_t = sigma_arr[i]
        spread_pct_t = spread_pct_arr[i]
        mid_t = mid_arr[i]
        
        if pd.isna(sigma_t) or sigma_t < MIN_SIGMA_THRESHOLD:
            continue
        if pd.isna(mid_t) or mid_t <= 0:
            continue
        
        # Forward returns
        future_mids = mid_arr[i+1:i+1+horizon_minutes]
        if len(future_mids) == 0:
            continue
        
        returns = (future_mids / mid_t) - 1.0
        mu_plus = np.nanmax(returns)
        mu_minus = np.nanmin(returns)
        
        movement_potential = max(mu_plus, abs(mu_minus))
        cost = spread_pct_t + MICRO_SLIPPAGE_PCT if not pd.isna(spread_pct_t) else 0.0001
        
        if cost <= 0:
            cost = 0.0001
        
        if movement_potential / cost >= min_edge_mult:
            labels[i] = 1
    
    return pd.Series(labels, index=df.index)


def generate_labels_for_all_horizons(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    df = df.copy()
    
    for horizon_name, params in HORIZON_PARAMS.items():
        horizon_minutes = params["minutes"]
        min_edge_mult = params["min_edge_mult"]
        
        if verbose:
            print(f"  Labeling {horizon_name}...", end=" ")
        
        labels = label_environment_for_horizon(df, horizon_minutes, min_edge_mult)
        df[f"env_{horizon_minutes}m"] = labels
        df[f"y_{horizon_minutes}m"] = labels  # Legacy
        
        if verbose:
            good = (labels == 1).sum()
            total = len(labels) - horizon_minutes
            print(f"Good env: {good:,} ({100*good/total:.1f}%)")
    
    return df


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_environment_model(df: pd.DataFrame, horizon: str, model_path: str,
                            feature_cols: list, train_end: str, val_end: str,
                            verbose: bool = True) -> dict:
    
    horizon_minutes = HORIZON_PARAMS[horizon]["minutes"]
    label_col = f"env_{horizon_minutes}m"
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training BINARY environment model for {horizon}")
        print(f"{'='*60}")
    
    # Split data
    train_end_dt = pd.Timestamp(train_end, tz='UTC')
    val_end_dt = pd.Timestamp(val_end, tz='UTC')
    
    train_df = df[df.index <= train_end_dt]
    val_df = df[(df.index > train_end_dt + pd.Timedelta(days=1)) & (df.index <= val_end_dt)]
    test_df = df[df.index > val_end_dt + pd.Timedelta(days=1)]
    
    if verbose:
        print(f"Train: {train_df.index.min().date()} to {train_df.index.max().date()}")
        print(f"Val:   {val_df.index.min().date()} to {val_df.index.max().date()}")
        print(f"Test:  {test_df.index.min().date()} to {test_df.index.max().date()}")
    
    # Prepare data
    available = [c for c in feature_cols if c in df.columns]
    
    def prep_data(sub_df):
        work = sub_df[available + [label_col]].dropna()
        X = work[available].values
        y = work[label_col].values.astype(int)
        return X, y
    
    X_train, y_train = prep_data(train_df)
    X_val, y_val = prep_data(val_df)
    X_test, y_test = prep_data(test_df)
    
    if verbose:
        print(f"\nTrain: {len(y_train):,}, Val: {len(y_val):,}, Test: {len(y_test):,}")
    
    # Class balance
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    
    if verbose:
        print(f"Class balance - Good: {n_pos:,} ({100*n_pos/len(y_train):.1f}%), Bad: {n_neg:,}")
        print(f"scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Train model
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        max_depth=4,
        learning_rate=0.05,
        n_estimators=500,
        min_child_weight=20,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=2.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        early_stopping_rounds=50,
    )
    
    if verbose:
        print("\nTraining with AUC optimization...")
    
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    if verbose:
        print(f"Best iteration: {model.best_iteration}")
    
    # Evaluate
    def eval_metrics(X, y, name):
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_proba)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        if verbose:
            print(f"  {name}: AUC={auc:.3f}, Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}")
        return {"auc": auc, "accuracy": acc, "precision": prec, "recall": rec}
    
    if verbose:
        print("\nResults:")
    train_metrics = eval_metrics(X_train, y_train, "Train")
    val_metrics = eval_metrics(X_val, y_val, "Val")
    test_metrics = eval_metrics(X_test, y_test, "Test")
    
    # Save model
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": available,
        "horizon": horizon,
        "scale_pos_weight": scale_pos_weight,
        "model_type": "binary_environment",
    }, model_path)
    
    return {
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


# =============================================================================
# SIGNAL GENERATION & BACKTEST
# =============================================================================

def batch_generate_signals(df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    results = {}
    
    for horizon_name, params in HORIZON_PARAMS.items():
        horizon_minutes = params["minutes"]
        model_path = Path(model_dir) / f"model_{horizon_name}.pkl"
        data = joblib.load(model_path)
        model = data["model"]
        features = data["features"]
        
        available = [c for c in features if c in df.columns]
        X = df[available].fillna(0).values
        
        env_proba = model.predict_proba(X)
        p_good = env_proba[:, 1] if env_proba.shape[1] == 2 else env_proba[:, 0]
        
        results[f"env_score_{horizon_name}"] = p_good
        results[f"env_good_{horizon_name}"] = (p_good >= ENV_THRESHOLD).astype(int)
    
    # Direction from microstructure (vectorized)
    imbalance = df["imbalance"].values if "imbalance" in df.columns else np.zeros(len(df))
    imbalance_ma5 = df["imbalance_ma5"].values if "imbalance_ma5" in df.columns else imbalance
    sigma_increasing = df["sigma_increasing"].values if "sigma_increasing" in df.columns else np.ones(len(df))
    
    direction = np.zeros(len(df), dtype=int)
    direction = np.where((imbalance_ma5 > IMBALANCE_THRESHOLD) & (sigma_increasing == 1), 1, direction)
    direction = np.where((imbalance_ma5 < -IMBALANCE_THRESHOLD) & (sigma_increasing == 1), -1, direction)
    
    results["direction"] = direction
    
    # Filters
    sigma = df["sigma"].values if "sigma" in df.columns else np.ones(len(df)) * 0.0001
    spread_pct = df["spread_pct"].values if "spread_pct" in df.columns else np.zeros(len(df))
    session_quality = df["session_quality"].values if "session_quality" in df.columns else np.ones(len(df)) * 2
    
    filter_mask = (
        (sigma >= MIN_SIGMA_THRESHOLD) &
        (sigma <= 0.003) &
        (spread_pct <= 0.001) &
        (session_quality >= 1) &
        (~np.isnan(sigma)) &
        (~np.isnan(spread_pct))
    )
    results["filter_passed"] = filter_mask.astype(int)
    
    # Final signals
    for horizon_name in HORIZON_PARAMS:
        env_good = results[f"env_good_{horizon_name}"]
        signal = np.where((env_good == 1) & filter_mask & (direction != 0), direction, 0)
        results[f"signal_{horizon_name}"] = signal
    
    return pd.DataFrame(results, index=df.index)


def run_backtest(df: pd.DataFrame, signals_df: pd.DataFrame, horizon_name: str) -> pd.DataFrame:
    params = HORIZON_PARAMS[horizon_name]
    horizon_minutes = params["minutes"]
    k1, k2 = params["k1"], params["k2"]
    
    mid = df["mid"].values
    sigma = df["sigma"].values if "sigma" in df.columns else np.ones(len(df)) * 0.0001
    spread_pct = df["spread_pct"].values
    signal = signals_df[f"signal_{horizon_name}"].values
    env_score = signals_df[f"env_score_{horizon_name}"].values
    
    n = len(df)
    trades = []
    valid_end = n - horizon_minutes - 1
    
    for i in range(valid_end):
        if signal[i] == 0:
            continue
        
        entry_mid = mid[i]
        entry_sigma = sigma[i]
        entry_spread_pct = spread_pct[i]
        
        if np.isnan(entry_mid) or np.isnan(entry_sigma) or entry_sigma <= 0:
            continue
        
        if i + 1 >= n:
            continue
        entry_price = mid[i + 1]
        if np.isnan(entry_price):
            continue
        
        sl_ret = k1 * entry_sigma
        tp_ret = k2 * entry_sigma + entry_spread_pct
        
        trade_dir = signal[i]
        
        if trade_dir == 1:
            sl_price = entry_price * (1 - sl_ret)
            tp_price = entry_price * (1 + tp_ret)
        else:
            sl_price = entry_price * (1 + sl_ret)
            tp_price = entry_price * (1 - tp_ret)
        
        exit_price = None
        hit_tp = hit_sl = timed_out = False
        
        for j in range(i + 2, min(i + 2 + horizon_minutes, n)):
            future_mid = mid[j]
            if np.isnan(future_mid):
                continue
            
            if trade_dir == 1:
                if future_mid <= sl_price:
                    exit_price, hit_sl = sl_price, True
                    break
                elif future_mid >= tp_price:
                    exit_price, hit_tp = tp_price, True
                    break
            else:
                if future_mid >= sl_price:
                    exit_price, hit_sl = sl_price, True
                    break
                elif future_mid <= tp_price:
                    exit_price, hit_tp = tp_price, True
                    break
        
        if exit_price is None:
            exit_idx = min(i + 1 + horizon_minutes, n - 1)
            exit_price = mid[exit_idx]
            timed_out = True
        
        if np.isnan(exit_price):
            continue
        
        if trade_dir == 1:
            actual_return = (exit_price / entry_price) - 1.0
        else:
            actual_return = (entry_price / exit_price) - 1.0
        
        pnl_ret = actual_return - entry_spread_pct
        r_multiple = pnl_ret / sl_ret if sl_ret > 0 else 0
        
        # Environment accuracy check
        forward_returns = (mid[i+1:i+1+horizon_minutes] / entry_mid) - 1.0
        forward_returns = forward_returns[~np.isnan(forward_returns)]
        if len(forward_returns) > 0:
            max_move = max(np.max(forward_returns), abs(np.min(forward_returns)))
            env_was_correct = max_move > (entry_spread_pct * 1.5)
        else:
            env_was_correct = False
        
        trades.append({
            "timestamp": df.index[i],
            "signal": trade_dir,
            "env_score": env_score[i],
            "pnl_ret": pnl_ret,
            "r_multiple": r_multiple,
            "hit_tp": hit_tp,
            "hit_sl": hit_sl,
            "timed_out": timed_out,
            "env_was_correct": env_was_correct,
            "direction_was_correct": pnl_ret > 0,
        })
    
    return pd.DataFrame(trades)


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    if len(trades_df) == 0:
        return {"n_trades": 0, "error": "No trades"}
    
    n = len(trades_df)
    total_r = trades_df["r_multiple"].sum()
    avg_r = trades_df["r_multiple"].mean()
    win_rate = (trades_df["pnl_ret"] > 0).mean()
    
    long_trades = trades_df[trades_df["signal"] == 1]
    short_trades = trades_df[trades_df["signal"] == -1]
    
    long_count = len(long_trades)
    short_count = len(short_trades)
    
    long_win = (long_trades["pnl_ret"] > 0).mean() if long_count > 0 else 0
    short_win = (short_trades["pnl_ret"] > 0).mean() if short_count > 0 else 0
    long_avg_r = long_trades["r_multiple"].mean() if long_count > 0 else 0
    short_avg_r = short_trades["r_multiple"].mean() if short_count > 0 else 0
    
    env_acc = trades_df["env_was_correct"].mean()
    dir_acc = trades_df["direction_was_correct"].mean()
    
    hit_tp = trades_df["hit_tp"].mean()
    hit_sl = trades_df["hit_sl"].mean()
    timeout = trades_df["timed_out"].mean()
    
    cumulative = trades_df["pnl_ret"].cumsum()
    max_dd = (cumulative.cummax() - cumulative).max()
    
    return {
        "n_trades": n,
        "total_r": total_r,
        "avg_r": avg_r,
        "win_rate": win_rate,
        "long_count": long_count,
        "short_count": short_count,
        "long_pct": 100 * long_count / n if n > 0 else 0,
        "short_pct": 100 * short_count / n if n > 0 else 0,
        "long_win_rate": long_win,
        "short_win_rate": short_win,
        "long_avg_r": long_avg_r,
        "short_avg_r": short_avg_r,
        "env_accuracy": env_acc,
        "direction_accuracy": dir_acc,
        "hit_tp_rate": hit_tp,
        "hit_sl_rate": hit_sl,
        "timeout_rate": timeout,
        "max_drawdown": max_dd,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = time.time()
    
    print("=" * 70)
    print("Dr. Chen Environment-Based Signal System - Test")
    print("=" * 70)
    print()
    
    DATA_DIR = Path(__file__).parent.parent / "Data"
    MINUTE_DIR = DATA_DIR / "ohlcv_minute"
    QUOTES_DIR = DATA_DIR / "quotes"
    MODEL_DIR = Path(__file__).parent / "models"
    
    YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
    TRAIN_END = "2023-12-31"
    VAL_END = "2024-06-30"
    
    # Step 1: Load data
    print("STEP 1: Loading data...")
    df = load_multi_year_data(str(MINUTE_DIR), str(QUOTES_DIR), YEARS)
    print(f"\nTotal: {len(df):,} rows")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    
    # Step 2: Build features
    print("\nSTEP 2: Building features...")
    df = build_feature_matrix(df)
    features = get_environment_model_features()
    available = [c for c in features if c in df.columns]
    print(f"Features: {len(available)}")
    
    # Step 3: Generate labels
    print("\nSTEP 3: Generating BINARY environment labels...")
    df = generate_labels_for_all_horizons(df, verbose=True)
    
    # Step 4: Train models
    print("\nSTEP 4: Training BINARY environment models...")
    MODEL_DIR.mkdir(exist_ok=True)
    
    for horizon in ["5m", "15m", "30m"]:
        train_environment_model(
            df=df,
            horizon=horizon,
            model_path=str(MODEL_DIR / f"model_{horizon}.pkl"),
            feature_cols=available,
            train_end=TRAIN_END,
            val_end=VAL_END,
            verbose=True
        )
    
    # Step 5: Backtest
    print("\nSTEP 5: Running TWO-STAGE backtest...")
    
    test_start = pd.Timestamp(VAL_END, tz='UTC') + pd.Timedelta(days=2)
    test_df = df[df.index >= test_start].copy()
    print(f"Test period: {test_df.index.min().date()} to {test_df.index.max().date()}")
    print(f"Test samples: {len(test_df):,}")
    
    signals_df = batch_generate_signals(test_df, str(MODEL_DIR))
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    all_metrics = {}
    for horizon in ["5m", "15m", "30m"]:
        trades = run_backtest(test_df, signals_df, horizon)
        metrics = compute_metrics(trades)
        all_metrics[horizon] = metrics
        
        m = metrics
        print(f"\n{horizon} HORIZON:")
        print(f"  Trades: {m['n_trades']:,}")
        print(f"  Long/Short: {m['long_count']:,} / {m['short_count']:,} ({m['long_pct']:.1f}% / {m['short_pct']:.1f}%)")
        print(f"  Win Rate: {m['win_rate']:.1%}")
        print(f"  Avg R: {m['avg_r']:.3f}")
        print(f"  Total R: {m['total_r']:.2f}")
        print(f"  Long Win: {m['long_win_rate']:.1%}, Short Win: {m['short_win_rate']:.1%}")
        print(f"  Environment Accuracy: {m['env_accuracy']:.1%}")
        print(f"  Direction Accuracy: {m['direction_accuracy']:.1%}")
        print(f"  Exit: TP {m['hit_tp_rate']:.1%}, SL {m['hit_sl_rate']:.1%}, Timeout {m['timeout_rate']:.1%}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for horizon, m in all_metrics.items():
        balanced = 30 < m['long_pct'] < 70
        positive_r = m['avg_r'] > 0
        status = "✓" if balanced and positive_r else "⚠" if balanced else "✗"
        print(f"  {horizon}: {status} R={m['total_r']:+.2f}, WR={m['win_rate']:.1%}, Balance={m['long_pct']:.0f}%L/{m['short_pct']:.0f}%S")
    
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.1f} seconds")
    
    return all_metrics


if __name__ == "__main__":
    main()
