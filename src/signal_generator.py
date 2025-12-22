"""
Signal generation module for the XAUUSD Signal Engine.

Dr. Chen Style Implementation:
==============================
This module implements the TWO-STAGE signal generation:

Stage 1: ENVIRONMENT SCORING
    - Model predicts P(good_environment)
    - If P < threshold → NO TRADE (bad environment)
    - If P >= threshold → Proceed to Stage 2

Stage 2: DIRECTION DETERMINATION (Microstructure-Based)
    - Direction comes from MICROSTRUCTURE features, NOT from model
    - Uses: imbalance, volatility trend, VWAP deviation
    - This prevents the model from collapsing to one direction

The model learns WHEN to trade.
The microstructure logic determines WHICH DIRECTION.

This separation eliminates:
- Directional bias (always long/always short)
- Regime confusion
- Class collapse
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from enum import IntEnum
from typing import Dict, List, Optional, Any, Tuple

from xgboost import XGBClassifier

from .feature_engineering import get_environment_model_features, get_direction_features


class Signal(IntEnum):
    """Trading signal enum."""
    SHORT = -1
    FLAT = 0
    LONG = 1
    
    def __str__(self) -> str:
        return self.name


# =============================================================================
# CONFIGURATION
# =============================================================================

# Environment score threshold
ENV_THRESHOLD = 0.5  # P(good_env) must exceed this

# Direction determination thresholds
IMBALANCE_THRESHOLD = 0.3  # |imbalance| must exceed this for direction
SIGMA_SLOPE_THRESHOLD = 0.0  # Volatility must be increasing (> 0)

# Horizon parameters for SL/TP
HORIZON_PARAMS = {
    "5m": {"k1": 1.5, "k2": 2.0},
    "15m": {"k1": 1.5, "k2": 2.5},
    "30m": {"k1": 1.5, "k2": 3.0},
}

# Pre-trade filters (applied before environment scoring)
FILTER_PARAMS = {
    "min_sigma": 0.00003,
    "max_sigma": 0.003,
    "max_spread_pct": 0.001,
    "min_session_quality": 1,
}


def load_environment_models(model_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load trained BINARY environment models.
    
    These models predict P(good_environment), NOT direction.
    """
    model_dir = Path(model_dir)
    models = {}
    
    for horizon in ["5m", "15m", "30m"]:
        model_path = model_dir / f"model_{horizon}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        data = joblib.load(model_path)
        
        models[horizon] = {
            "model": data["model"],
            "features": data["features"],
            "model_type": data.get("model_type", "unknown"),
        }
    
    return models


def determine_direction_from_microstructure(
    row: pd.Series,
    imbalance_threshold: float = IMBALANCE_THRESHOLD,
    sigma_slope_threshold: float = SIGMA_SLOPE_THRESHOLD
) -> Signal:
    """
    Determine trade direction from MICROSTRUCTURE features only.
    
    This is the SECOND STAGE of signal generation.
    Called only AFTER environment is classified as good.
    
    Direction Logic:
    - LONG if: imbalance > threshold AND (sigma_slope > 0 OR momentum > 0)
    - SHORT if: imbalance < -threshold AND (sigma_slope > 0 OR momentum > 0)
    - FLAT otherwise
    
    Args:
        row: Series with microstructure features
        imbalance_threshold: Minimum |imbalance| for direction signal
        sigma_slope_threshold: Minimum sigma slope
        
    Returns:
        Signal.LONG, Signal.SHORT, or Signal.FLAT
    """
    # Get microstructure indicators
    imbalance = row.get("imbalance", 0)
    imbalance_ma5 = row.get("imbalance_ma5", imbalance)
    sigma_slope = row.get("sigma_slope", 0)
    sigma_increasing = row.get("sigma_increasing", 0)
    momentum_5 = row.get("momentum_5", 0)
    vwap_deviation = row.get("vwap_deviation", 0)
    
    # Use smoothed imbalance for stability
    imb = imbalance_ma5 if not pd.isna(imbalance_ma5) else imbalance
    
    # Volatility condition: sigma should be stable or increasing
    vol_ok = (sigma_increasing == 1) or (sigma_slope >= sigma_slope_threshold)
    
    # Momentum confirmation (optional, helps filter)
    mom_up = momentum_5 > 0 if not pd.isna(momentum_5) else True
    mom_down = momentum_5 < 0 if not pd.isna(momentum_5) else True
    
    # Direction determination
    if imb > imbalance_threshold and (vol_ok or mom_up):
        return Signal.LONG
    elif imb < -imbalance_threshold and (vol_ok or mom_down):
        return Signal.SHORT
    else:
        return Signal.FLAT


def check_pre_trade_filters(
    row: pd.Series,
    params: Optional[Dict] = None
) -> Tuple[bool, str]:
    """
    Check pre-trade filters before environment scoring.
    
    Returns:
        Tuple (passed: bool, reason: str)
    """
    if params is None:
        params = FILTER_PARAMS
    
    sigma = row.get("sigma", row.get("sigma_60", 0))
    if pd.isna(sigma) or sigma < params.get("min_sigma", 0):
        return False, "vol_too_low"
    if sigma > params.get("max_sigma", 1):
        return False, "vol_too_high"
    
    spread_pct = row.get("spread_pct", 0)
    if pd.isna(spread_pct) or spread_pct > params.get("max_spread_pct", 1):
        return False, "spread_too_high"
    
    session_quality = row.get("session_quality", 2)
    if session_quality < params.get("min_session_quality", 0):
        return False, "poor_session"
    
    return True, "ok"


def compute_sl_tp_prices(
    mid: float,
    sigma: float,
    spread_pct: float,
    direction: Signal,
    k1: float,
    k2: float
) -> Dict[str, float]:
    """Compute SL/TP prices based on volatility."""
    sl_ret = k1 * sigma
    tp_ret = k2 * sigma + spread_pct  # Cost-adjusted TP
    
    if direction == Signal.LONG:
        sl_price = mid * (1 - sl_ret)
        tp_price = mid * (1 + tp_ret)
    elif direction == Signal.SHORT:
        sl_price = mid * (1 + sl_ret)
        tp_price = mid * (1 - tp_ret)
    else:
        sl_price = None
        tp_price = None
    
    return {
        "sl_price": sl_price,
        "tp_price": tp_price,
        "sl_ret": sl_ret,
        "tp_ret": tp_ret,
        "risk_reward": tp_ret / sl_ret if sl_ret > 0 else 0,
    }


def generate_signal_for_horizon(
    model: XGBClassifier,
    features: np.ndarray,
    row: pd.Series,
    horizon: str,
    env_threshold: float = ENV_THRESHOLD
) -> Dict[str, Any]:
    """
    Generate signal for a single horizon using TWO-STAGE logic.
    
    Stage 1: Environment scoring (from model)
    Stage 2: Direction determination (from microstructure)
    """
    # Stage 1: Environment Score
    env_proba = model.predict_proba(features)[0]
    
    # Binary model: proba is [P(bad), P(good)]
    if len(env_proba) == 2:
        p_good_env = env_proba[1]
    else:
        p_good_env = env_proba[0]  # Fallback
    
    result = {
        "env_score": float(p_good_env),
        "env_good": p_good_env >= env_threshold,
    }
    
    # If environment is bad, no trade
    if p_good_env < env_threshold:
        result["signal"] = Signal.FLAT
        result["direction_reason"] = "bad_environment"
        result["sl_price"] = None
        result["tp_price"] = None
        return result
    
    # Stage 2: Direction from Microstructure
    direction = determine_direction_from_microstructure(row)
    result["signal"] = direction
    
    if direction == Signal.FLAT:
        result["direction_reason"] = "no_clear_direction"
        result["sl_price"] = None
        result["tp_price"] = None
        return result
    
    result["direction_reason"] = "imbalance_based"
    
    # Compute SL/TP
    mid = float(row["mid"])
    sigma = float(row.get("sigma", row.get("sigma_60", 0.0001)))
    spread_pct = float(row.get("spread_pct", 0.0001))
    
    hp = HORIZON_PARAMS.get(horizon, {"k1": 1.5, "k2": 2.0})
    sl_tp = compute_sl_tp_prices(mid, sigma, spread_pct, direction, hp["k1"], hp["k2"])
    
    result.update(sl_tp)
    
    return result


def generate_signals_for_latest_row(
    df: pd.DataFrame,
    model_dir: str,
    env_threshold: float = ENV_THRESHOLD,
    filter_params: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Generate signals for all horizons using TWO-STAGE logic.
    
    Stage 1: Model predicts P(good_environment)
    Stage 2: Microstructure determines direction
    
    Args:
        df: DataFrame with features
        model_dir: Directory with trained environment models
        env_threshold: Threshold for environment quality
        filter_params: Pre-trade filter parameters
        
    Returns:
        Dictionary with:
        - timestamp, mid, sigma, spread_pct
        - filter_passed, filter_reason
        - signals per horizon with env_score and direction
    """
    models = load_environment_models(model_dir)
    
    last_row = df.iloc[-1]
    timestamp = df.index[-1]
    mid = float(last_row["mid"])
    sigma = float(last_row.get("sigma", last_row.get("sigma_60", 0)))
    spread_pct = float(last_row["spread_pct"])
    
    # Pre-trade filter
    filter_passed, filter_reason = check_pre_trade_filters(last_row, filter_params)
    
    result = {
        "timestamp": timestamp,
        "mid": mid,
        "sigma": sigma,
        "spread_pct": spread_pct,
        "filter_passed": filter_passed,
        "filter_reason": filter_reason if not filter_passed else None,
        "signals": {},
    }
    
    for horizon in ["5m", "15m", "30m"]:
        model_data = models[horizon]
        model = model_data["model"]
        model_features = model_data["features"]
        
        # Extract features
        available = [c for c in model_features if c in df.columns]
        features = df[available].iloc[-1:].values
        
        # Handle NaN
        if np.any(np.isnan(features)):
            result["signals"][horizon] = {
                "signal": Signal.FLAT,
                "env_score": 0.0,
                "error": "nan_features"
            }
            continue
        
        # Generate signal (two-stage)
        signal_result = generate_signal_for_horizon(
            model=model,
            features=features,
            row=last_row,
            horizon=horizon,
            env_threshold=env_threshold
        )
        
        # Apply pre-trade filter
        if not filter_passed and signal_result["signal"] != Signal.FLAT:
            signal_result["original_signal"] = signal_result["signal"]
            signal_result["signal"] = Signal.FLAT
            signal_result["filtered"] = True
            signal_result["filter_reason"] = filter_reason
        else:
            signal_result["filtered"] = False
        
        result["signals"][horizon] = signal_result
    
    return result


def format_signals_summary(result: Dict[str, Any]) -> str:
    """Format signal results as human-readable summary."""
    lines = [
        f"Signal Summary @ {result['timestamp']}",
        f"{'='*55}",
        f"Mid Price: ${result['mid']:.2f}",
        f"Volatility (σ): {result['sigma']:.6f}",
        f"Spread: {result['spread_pct']*100:.4f}%",
        f"Filter Passed: {result['filter_passed']}",
    ]
    
    if not result['filter_passed']:
        lines.append(f"Filter Reason: {result['filter_reason']}")
    
    lines.append("")
    
    for horizon in ["5m", "15m", "30m"]:
        sig = result["signals"].get(horizon, {})
        signal_name = str(sig.get("signal", Signal.FLAT))
        env_score = sig.get("env_score", 0)
        
        lines.append(f"{horizon} Horizon:")
        lines.append(f"  Env Score: {env_score:.1%} ({'GOOD' if sig.get('env_good') else 'BAD'})")
        lines.append(f"  Signal: {signal_name}")
        lines.append(f"  Direction Reason: {sig.get('direction_reason', 'N/A')}")
        
        if sig.get("sl_price"):
            lines.append(f"  SL: ${sig['sl_price']:.2f}")
            lines.append(f"  TP: ${sig['tp_price']:.2f}")
            lines.append(f"  R:R: {sig.get('risk_reward', 0):.2f}")
        lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# BATCH SIGNAL GENERATION (for backtesting)
# =============================================================================

def batch_generate_signals(
    df: pd.DataFrame,
    model_dir: str,
    env_threshold: float = ENV_THRESHOLD,
    filter_params: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Generate signals for all rows using vectorized operations.
    
    Two-stage logic:
    1. Batch predict environment scores
    2. Determine direction from microstructure features
    """
    if filter_params is None:
        filter_params = FILTER_PARAMS
    
    models = load_environment_models(model_dir)
    
    results = {}
    
    for horizon in ["5m", "15m", "30m"]:
        model = models[horizon]["model"]
        features = models[horizon]["features"]
        
        # Get available features
        available = [c for c in features if c in df.columns]
        X = df[available].values
        
        # Stage 1: Batch predict environment scores
        env_proba = model.predict_proba(X)
        p_good_env = env_proba[:, 1] if env_proba.shape[1] == 2 else env_proba[:, 0]
        
        results[f"env_score_{horizon}"] = p_good_env
        results[f"env_good_{horizon}"] = (p_good_env >= env_threshold).astype(int)
    
    # Stage 2: Direction from microstructure (vectorized)
    imbalance = df["imbalance"].values if "imbalance" in df.columns else np.zeros(len(df))
    imbalance_ma5 = df["imbalance_ma5"].values if "imbalance_ma5" in df.columns else imbalance
    sigma_increasing = df["sigma_increasing"].values if "sigma_increasing" in df.columns else np.ones(len(df))
    
    # Direction logic (vectorized)
    direction = np.zeros(len(df), dtype=int)
    direction = np.where(
        (imbalance_ma5 > IMBALANCE_THRESHOLD) & (sigma_increasing == 1),
        1,  # LONG
        direction
    )
    direction = np.where(
        (imbalance_ma5 < -IMBALANCE_THRESHOLD) & (sigma_increasing == 1),
        -1,  # SHORT
        direction
    )
    
    results["direction"] = direction
    
    # Apply filters (vectorized)
    sigma = df["sigma"].values if "sigma" in df.columns else np.ones(len(df)) * 0.0001
    spread_pct = df["spread_pct"].values if "spread_pct" in df.columns else np.zeros(len(df))
    session_quality = df["session_quality"].values if "session_quality" in df.columns else np.ones(len(df)) * 2
    
    filter_mask = (
        (sigma >= filter_params.get("min_sigma", 0)) &
        (sigma <= filter_params.get("max_sigma", 1)) &
        (spread_pct <= filter_params.get("max_spread_pct", 1)) &
        (session_quality >= filter_params.get("min_session_quality", 0)) &
        (~np.isnan(sigma)) &
        (~np.isnan(spread_pct))
    )
    
    results["filter_passed"] = filter_mask.astype(int)
    
    # Final signals: env_good AND direction != 0 AND filter_passed
    for horizon in ["5m", "15m", "30m"]:
        env_good = results[f"env_good_{horizon}"]
        signal = np.where(
            (env_good == 1) & (filter_mask) & (direction != 0),
            direction,
            0
        )
        results[f"signal_{horizon}"] = signal
    
    return pd.DataFrame(results, index=df.index)


# Legacy alias
load_models = load_environment_models
