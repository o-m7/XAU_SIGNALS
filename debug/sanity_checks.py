"""
Sanity Checks Module - Debug & Audit Layer

This module contains diagnostic functions to verify:
1. Price data and returns are computed correctly
2. SL/TP levels are correctly positioned
3. Labels have sane distributions
4. Features have correct directionality

DO NOT change strategy logic here - only validate and diagnose.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple


# =============================================================================
# 1. DATA LOADING FOR SPECIFIC PERIODS
# =============================================================================

def load_year_data(
    minute_dir: str,
    quotes_dir: str,
    year: int
) -> pd.DataFrame:
    """
    Load data for a specific year.
    
    Returns combined minute OHLCV + quotes data.
    """
    minute_path = Path(minute_dir) / f"XAUUSD_minute_{year}.parquet"
    quotes_path = Path(quotes_dir) / f"XAUUSD_quotes_{year}.parquet"
    
    if not minute_path.exists():
        raise FileNotFoundError(f"Minute data not found: {minute_path}")
    if not quotes_path.exists():
        raise FileNotFoundError(f"Quotes data not found: {quotes_path}")
    
    m = pd.read_parquet(minute_path)
    q = pd.read_parquet(quotes_path)
    
    # Parse timestamps
    for d in [m, q]:
        if "timestamp" in d.columns:
            d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
            d.set_index("timestamp", inplace=True)
    
    m = m.reset_index()
    q = q.reset_index()
    
    # Merge
    df = pd.merge_asof(
        m.sort_values("timestamp"),
        q.sort_values("timestamp")[["timestamp", "bid_price", "ask_price"]],
        on="timestamp",
        direction="backward"
    ).set_index("timestamp")
    
    df = df.dropna(subset=["bid_price", "ask_price"])
    df.sort_index(inplace=True)
    
    return df


# =============================================================================
# 2. MID PRICE AND TREND DIRECTION CHECKS
# =============================================================================

def check_mid_price_trend(df: pd.DataFrame) -> dict:
    """
    Check that mid price trend matches known market behavior.
    
    For 2024, gold was in a bull market - mid should trend UP.
    """
    df = df.copy()
    df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    
    results = {
        "start_mid": float(df["mid"].iloc[0]),
        "end_mid": float(df["mid"].iloc[-1]),
        "start_date": str(df.index[0].date()),
        "end_date": str(df.index[-1].date()),
        "total_return_pct": float((df["mid"].iloc[-1] / df["mid"].iloc[0] - 1) * 100),
        "min_mid": float(df["mid"].min()),
        "max_mid": float(df["mid"].max()),
        "mean_mid": float(df["mid"].mean()),
    }
    
    print("=" * 60)
    print("MID PRICE TREND CHECK")
    print("=" * 60)
    print(f"Period: {results['start_date']} to {results['end_date']}")
    print(f"Start Mid: ${results['start_mid']:.2f}")
    print(f"End Mid:   ${results['end_mid']:.2f}")
    print(f"Total Return: {results['total_return_pct']:+.2f}%")
    print(f"Min/Max: ${results['min_mid']:.2f} / ${results['max_mid']:.2f}")
    print()
    
    # Validation
    if results["total_return_pct"] > 0:
        print("✓ Price trended UP (bullish)")
    else:
        print("⚠ Price trended DOWN (bearish)")
    
    # For 2024, gold went from ~$2060 to ~$2650 (roughly +28%)
    if "2024" in results["start_date"]:
        if results["total_return_pct"] < 15:
            print("⚠ WARNING: 2024 gold gained ~25-30%. This data shows less.")
        if results["total_return_pct"] < 0:
            print("❌ ERROR: 2024 was a BULL year for gold, but data shows DOWN!")
            print("   This could indicate inverted data or wrong symbol.")
    
    return results


def check_forward_returns(df: pd.DataFrame, horizon: int = 5) -> dict:
    """
    Check forward returns calculation.
    
    Forward return at time t = (price at t+h) / (price at t) - 1
    
    This should NOT use shift(+horizon) on the wrong direction.
    """
    df = df.copy()
    df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    
    # CORRECT: shift(-horizon) means we're looking FORWARD
    # mid.shift(-5) at index i gives us the price at i+5
    df[f"ret_fwd_{horizon}m"] = df["mid"].shift(-horizon) / df["mid"] - 1.0
    
    fwd_ret = df[f"ret_fwd_{horizon}m"].dropna()
    
    results = {
        "horizon": horizon,
        "count": len(fwd_ret),
        "mean": float(fwd_ret.mean()),
        "std": float(fwd_ret.std()),
        "min": float(fwd_ret.min()),
        "max": float(fwd_ret.max()),
        "pct_positive": float((fwd_ret > 0).mean() * 100),
        "pct_negative": float((fwd_ret < 0).mean() * 100),
    }
    
    print("=" * 60)
    print(f"FORWARD RETURNS CHECK ({horizon}m horizon)")
    print("=" * 60)
    print(f"Count: {results['count']:,}")
    print(f"Mean:  {results['mean']*100:.6f}%")
    print(f"Std:   {results['std']*100:.6f}%")
    print(f"Min/Max: {results['min']*100:.4f}% / {results['max']*100:.4f}%")
    print(f"% Positive: {results['pct_positive']:.1f}%")
    print(f"% Negative: {results['pct_negative']:.1f}%")
    print()
    
    # Validation
    if abs(results["pct_positive"] - 50) > 10:
        print(f"⚠ Forward returns are skewed ({results['pct_positive']:.1f}% positive)")
    else:
        print("✓ Forward returns roughly balanced")
    
    # In a bull market, mean forward return should be slightly positive
    if results["mean"] > 0:
        print(f"✓ Mean forward return positive (consistent with uptrend)")
    else:
        print(f"⚠ Mean forward return negative despite uptrend")
    
    return results


# =============================================================================
# 3. BASELINE STRATEGY CHECK (TREND FOLLOW)
# =============================================================================

def baseline_trend_strategy(df: pd.DataFrame, horizon: int = 5) -> dict:
    """
    Simple trend-following baseline to verify return alignment.
    
    Strategy: If past H minutes were up, go long. If down, go short.
    
    This tests that past and future returns are aligned correctly.
    If this is catastrophically wrong, there's a shift bug.
    """
    df = df.copy()
    df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    
    # Past return: how much did price move in PREVIOUS horizon
    df[f"ret_past_{horizon}m"] = df["mid"] / df["mid"].shift(horizon) - 1.0
    
    # Future return: how much will price move in NEXT horizon  
    df[f"ret_fwd_{horizon}m"] = df["mid"].shift(-horizon) / df["mid"] - 1.0
    
    # Signal: follow the trend
    df["signal"] = 0
    df.loc[df[f"ret_past_{horizon}m"] > 0, "signal"] = 1   # Long after up move
    df.loc[df[f"ret_past_{horizon}m"] < 0, "signal"] = -1  # Short after down move
    
    # Strategy return
    df["strat_ret"] = df["signal"] * df[f"ret_fwd_{horizon}m"]
    
    valid = df["strat_ret"].dropna()
    
    results = {
        "horizon": horizon,
        "count": len(valid),
        "mean_strat_ret": float(valid.mean()),
        "total_strat_ret": float(valid.sum()),
        "win_rate": float((valid > 0).mean()),
        "n_long": int((df["signal"] == 1).sum()),
        "n_short": int((df["signal"] == -1).sum()),
    }
    
    print("=" * 60)
    print(f"BASELINE TREND-FOLLOW STRATEGY ({horizon}m)")
    print("=" * 60)
    print(f"Trades: {results['count']:,}")
    print(f"Long/Short: {results['n_long']:,} / {results['n_short']:,}")
    print(f"Mean Strategy Return: {results['mean_strat_ret']*100:.6f}%")
    print(f"Win Rate: {results['win_rate']:.1%}")
    print()
    
    # Validation
    # In efficient markets, trend-following on minute bars shouldn't work great,
    # but it also shouldn't be catastrophically negative
    if results["mean_strat_ret"] < -0.0001:  # Worse than -0.01% per trade
        print("❌ Baseline strategy is very negative - possible shift/sign bug!")
        print("   Check if past and future returns are correctly aligned.")
    elif results["mean_strat_ret"] < 0:
        print("⚠ Baseline slightly negative (normal for minute-bar trend following)")
    else:
        print("✓ Baseline non-negative")
    
    balance = results["n_long"] / (results["n_long"] + results["n_short"]) * 100
    print(f"Balance: {balance:.1f}% Long")
    
    return results


# =============================================================================
# 4. SL/TP ORDERING CHECKS
# =============================================================================

def check_sl_tp_ordering(
    df: pd.DataFrame,
    mid_col: str = "mid",
    sample_size: int = 1000
) -> dict:
    """
    Check that SL/TP levels are correctly positioned.
    
    For LONG: SL < mid < TP
    For SHORT: TP < mid < SL
    
    If these are violated, the SL/TP calculation is wrong.
    """
    # Check if SL/TP columns exist
    sl_tp_cols = ["sl_price_long", "tp_price_long", "sl_price_short", "tp_price_short"]
    missing = [c for c in sl_tp_cols if c not in df.columns]
    
    if missing:
        print("=" * 60)
        print("SL/TP ORDERING CHECK")
        print("=" * 60)
        print(f"⚠ Missing columns: {missing}")
        print("  Cannot perform SL/TP check. Computing from scratch...")
        
        # Compute SL/TP if we have sigma
        if "sigma" in df.columns and mid_col in df.columns:
            df = df.copy()
            sigma = df["sigma"]
            mid = df[mid_col]
            k1, k2 = 1.0, 2.0  # Standard multipliers
            
            df["sl_price_long"] = mid * (1 - k1 * sigma)
            df["tp_price_long"] = mid * (1 + k2 * sigma)
            df["sl_price_short"] = mid * (1 + k1 * sigma)
            df["tp_price_short"] = mid * (1 - k2 * sigma)
        else:
            print("  Cannot compute - no sigma column")
            return {"error": "missing_columns"}
    
    # Filter to valid rows
    required = [mid_col] + sl_tp_cols
    df_valid = df.dropna(subset=required).copy()
    
    if len(df_valid) == 0:
        print("❌ No valid rows with SL/TP data")
        return {"error": "no_valid_rows"}
    
    # Sample
    sample_n = min(sample_size, len(df_valid))
    df_sample = df_valid.sample(sample_n, random_state=42)
    
    mid = df_sample[mid_col]
    
    # Check LONG: SL < mid < TP
    long_sl_ok = df_sample["sl_price_long"] < mid
    long_tp_ok = mid < df_sample["tp_price_long"]
    long_ok = long_sl_ok & long_tp_ok
    
    # Check SHORT: TP < mid < SL
    short_tp_ok = df_sample["tp_price_short"] < mid
    short_sl_ok = mid < df_sample["sl_price_short"]
    short_ok = short_tp_ok & short_sl_ok
    
    results = {
        "sample_size": sample_n,
        "long_sl_correct_pct": float(long_sl_ok.mean() * 100),
        "long_tp_correct_pct": float(long_tp_ok.mean() * 100),
        "long_fully_correct_pct": float(long_ok.mean() * 100),
        "short_tp_correct_pct": float(short_tp_ok.mean() * 100),
        "short_sl_correct_pct": float(short_sl_ok.mean() * 100),
        "short_fully_correct_pct": float(short_ok.mean() * 100),
    }
    
    print("=" * 60)
    print("SL/TP ORDERING CHECK")
    print("=" * 60)
    print(f"Sample Size: {sample_n:,}")
    print()
    print("LONG Positions (should be: SL < mid < TP):")
    print(f"  SL < mid: {results['long_sl_correct_pct']:.1f}%")
    print(f"  mid < TP: {results['long_tp_correct_pct']:.1f}%")
    print(f"  FULLY CORRECT: {results['long_fully_correct_pct']:.1f}%")
    print()
    print("SHORT Positions (should be: TP < mid < SL):")
    print(f"  TP < mid: {results['short_tp_correct_pct']:.1f}%")
    print(f"  mid < SL: {results['short_sl_correct_pct']:.1f}%")
    print(f"  FULLY CORRECT: {results['short_fully_correct_pct']:.1f}%")
    print()
    
    # Validation
    if results["long_fully_correct_pct"] < 95:
        print("❌ LONG SL/TP ordering incorrect in > 5% of cases!")
        print("   Check the SL/TP calculation math.")
    else:
        print("✓ LONG SL/TP ordering correct")
    
    if results["short_fully_correct_pct"] < 95:
        print("❌ SHORT SL/TP ordering incorrect in > 5% of cases!")
        print("   Check the SL/TP calculation math.")
    else:
        print("✓ SHORT SL/TP ordering correct")
    
    return results


# =============================================================================
# 5. LABEL DISTRIBUTION CHECKS
# =============================================================================

def check_label_distribution(df: pd.DataFrame, label_cols: List[str]) -> dict:
    """
    Check label distributions for extreme imbalance.
    
    No class should be > 85% unless intentionally designed.
    """
    print("=" * 60)
    print("LABEL DISTRIBUTION CHECK")
    print("=" * 60)
    
    results = {}
    
    for col in label_cols:
        if col not in df.columns:
            print(f"\n⚠ Column '{col}' not found, skipping")
            continue
        
        print(f"\n{col}:")
        
        dist = df[col].value_counts(dropna=False, normalize=True)
        dist_counts = df[col].value_counts(dropna=False)
        
        print(dist.to_string())
        print(f"Total: {len(df[col].dropna()):,}")
        
        # Store results
        results[col] = {
            "distribution": dist.to_dict(),
            "counts": dist_counts.to_dict(),
            "max_class_pct": float(dist.max() * 100),
        }
        
        # Validation
        max_pct = dist.max() * 100
        if max_pct > 85:
            print(f"❌ EXTREME IMBALANCE: {max_pct:.1f}% in one class!")
            print("   This will cause the model to favor one direction.")
        elif max_pct > 70:
            print(f"⚠ Moderate imbalance: {max_pct:.1f}% in one class")
        else:
            print(f"✓ Balanced (max class: {max_pct:.1f}%)")
    
    return results


# =============================================================================
# 6. FEATURE DIRECTIONALITY CHECKS
# =============================================================================

def check_feature_directionality(
    df: pd.DataFrame,
    horizon: int = 5,
    features_to_check: Optional[List[str]] = None
) -> dict:
    """
    Check correlation between directional features and future returns.
    
    Tests that feature signs match theory:
    - Positive imbalance should have mild positive correlation with future returns
    - Positive momentum should have mild positive correlation
    - etc.
    
    Strong NEGATIVE correlations may indicate sign errors.
    """
    df = df.copy()
    
    # Ensure mid and forward return exist
    if "mid" not in df.columns:
        df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    
    df[f"fwd_ret_{horizon}m"] = df["mid"].shift(-horizon) / df["mid"] - 1.0
    
    if features_to_check is None:
        # Default features to check
        features_to_check = [
            "imbalance", "ofi_proxy", "momentum_5", "momentum_15",
            "vwap_deviation", "micro_dislocation", "log_ret"
        ]
    
    print("=" * 60)
    print(f"FEATURE DIRECTIONALITY CHECK ({horizon}m forward)")
    print("=" * 60)
    
    results = {}
    fwd_ret = df[f"fwd_ret_{horizon}m"]
    
    for feat in features_to_check:
        if feat not in df.columns:
            print(f"\n⚠ '{feat}' not found, skipping")
            continue
        
        # Compute correlation
        corr = df[[feat, f"fwd_ret_{horizon}m"]].corr().iloc[0, 1]
        
        results[feat] = {
            "correlation": float(corr) if not np.isnan(corr) else None,
            "horizon": horizon,
        }
        
        print(f"\n{feat}:")
        print(f"  Corr with {horizon}m forward return: {corr:.4f}")
        
        # Validation based on expected directionality
        if feat in ["imbalance", "ofi_proxy", "momentum_5", "momentum_15"]:
            # These should have mild POSITIVE correlation (buying pressure = up)
            if not np.isnan(corr):
                if corr < -0.05:
                    print(f"  ⚠ Unexpected negative correlation! Check feature sign.")
                elif corr > 0:
                    print(f"  ✓ Positive correlation (expected)")
                else:
                    print(f"  ~ Near zero correlation (normal for efficient markets)")
        
        if feat in ["log_ret"]:
            # Log return should have mild positive autocorrelation in trends
            if not np.isnan(corr):
                if abs(corr) > 0.1:
                    print(f"  ⚠ Unusually high autocorrelation")
                else:
                    print(f"  ✓ Normal autocorrelation range")
    
    return results


# =============================================================================
# 7. COMPREHENSIVE AUDIT RUNNER
# =============================================================================

def run_full_audit(
    minute_dir: str,
    quotes_dir: str,
    year: int = 2024,
    horizons: List[int] = [5, 15, 30]
) -> dict:
    """
    Run all sanity checks on a specific year of data.
    
    Returns comprehensive audit results.
    """
    print("=" * 70)
    print(f"FULL AUDIT - {year}")
    print("=" * 70)
    print()
    
    all_results = {"year": year}
    
    # Load data
    print("Loading data...")
    try:
        df = load_year_data(minute_dir, quotes_dir, year)
        print(f"Loaded {len(df):,} rows")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return {"error": str(e)}
    
    print()
    
    # 1. Mid price trend
    all_results["mid_trend"] = check_mid_price_trend(df)
    print()
    
    # 2. Forward returns for each horizon
    all_results["forward_returns"] = {}
    for h in horizons:
        all_results["forward_returns"][h] = check_forward_returns(df, h)
        print()
    
    # 3. Baseline strategy for each horizon
    all_results["baseline_strategy"] = {}
    for h in horizons:
        all_results["baseline_strategy"][h] = baseline_trend_strategy(df, h)
        print()
    
    # 4. Feature engineering (add basic features)
    print("Building features for additional checks...")
    df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
    df["spread"] = df["ask_price"] - df["bid_price"]
    df["spread_pct"] = df["spread"] / df["mid"]
    df["log_ret"] = np.log(df["mid"] / df["mid"].shift(1))
    
    # Volatility
    df["sigma"] = df["log_ret"].shift(1).rolling(60, min_periods=30).std()
    
    # Momentum
    df["momentum_5"] = df["mid"].shift(1).pct_change(5)
    df["momentum_15"] = df["mid"].shift(1).pct_change(15)
    
    # SL/TP
    k1, k2 = 1.0, 2.0
    df["sl_price_long"] = df["mid"] * (1 - k1 * df["sigma"])
    df["tp_price_long"] = df["mid"] * (1 + k2 * df["sigma"])
    df["sl_price_short"] = df["mid"] * (1 + k1 * df["sigma"])
    df["tp_price_short"] = df["mid"] * (1 - k2 * df["sigma"])
    
    # 5. SL/TP check
    all_results["sl_tp"] = check_sl_tp_ordering(df)
    print()
    
    # 6. Feature directionality
    all_results["feature_directionality"] = {}
    for h in horizons:
        all_results["feature_directionality"][h] = check_feature_directionality(
            df, h, ["log_ret", "momentum_5", "momentum_15"]
        )
        print()
    
    # Summary
    print("=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    
    issues = []
    
    # Check mid trend
    if all_results["mid_trend"]["total_return_pct"] < 0 and year == 2024:
        issues.append("Price trend DOWN in bull year (2024)")
    
    # Check baseline strategy
    for h, res in all_results["baseline_strategy"].items():
        if res["mean_strat_ret"] < -0.0001:
            issues.append(f"Baseline {h}m strategy very negative")
    
    # Check SL/TP
    if "sl_tp" in all_results and "error" not in all_results["sl_tp"]:
        if all_results["sl_tp"].get("long_fully_correct_pct", 100) < 95:
            issues.append("LONG SL/TP ordering incorrect")
        if all_results["sl_tp"].get("short_fully_correct_pct", 100) < 95:
            issues.append("SHORT SL/TP ordering incorrect")
    
    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ No critical issues found")
    
    return all_results


# =============================================================================
# MAIN - Run if executed directly
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Default paths
    BASE_DIR = Path(__file__).parent.parent.parent / "Data"
    MINUTE_DIR = BASE_DIR / "ohlcv_minute"
    QUOTES_DIR = BASE_DIR / "quotes"
    
    print(f"Data directory: {BASE_DIR}")
    print(f"Minute dir exists: {MINUTE_DIR.exists()}")
    print(f"Quotes dir exists: {QUOTES_DIR.exists()}")
    print()
    
    # Run audit on 2024
    results = run_full_audit(
        str(MINUTE_DIR),
        str(QUOTES_DIR),
        year=2024
    )

