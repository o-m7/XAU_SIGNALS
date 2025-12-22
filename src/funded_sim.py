#!/usr/bin/env python3
"""
Funded Account Simulation for XAUUSD Trading Strategy.

Simulates a prop-firm style funded account with:
- Starting balance: $25,000
- Profit target: +5% ($26,250)
- Max drawdown: -6% ($23,500)

Uses model predictions to generate long/short signals and tracks
equity curve with R-based position sizing.

Usage:
    cd /Users/omar/Desktop/ML
    source xauusd_signals/venv/bin/activate
    python xauusd_signals/src/funded_sim.py --split test --risk_pct 0.0025 --threshold_long 0.60 --threshold_short 0.40
"""

import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2024.parquet"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "y_tb_60_hgb_tuned.joblib"

# Funded account parameters
STARTING_BALANCE = 25_000.0
PROFIT_TARGET_PCT = 0.05  # +5%
MAX_DRAWDOWN_PCT = 0.06   # -6%

PROFIT_TARGET = STARTING_BALANCE * (1 + PROFIT_TARGET_PCT)  # $26,250
MAX_DRAWDOWN_LEVEL = STARTING_BALANCE * (1 - MAX_DRAWDOWN_PCT)  # $23,500

# Split ratios (must match training)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

TARGET = "y_tb_60"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SimulationResult:
    """Container for simulation results."""
    status: str  # PASSED, FAILED, INCOMPLETE
    starting_equity: float
    ending_equity: float
    profit_dollars: float
    profit_pct: float
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    max_drawdown_dollars: float
    max_drawdown_pct: float
    avg_r_per_trade: float
    expectancy: float  # avg $ per trade
    equity_curve: pd.DataFrame


@dataclass 
class ModelArtifact:
    """Container for loaded model artifacts."""
    path: str
    model: object
    features: List[str]
    params: Dict


# =============================================================================
# DATA LOADING
# =============================================================================

def load_features_data() -> pd.DataFrame:
    """Load the features parquet file."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_PATH}")
    
    df = pd.read_parquet(FEATURES_PATH)
    
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df = df.set_index("timestamp")
    
    df = df.sort_index()
    return df


def load_model_artifacts(model_paths: List[str]) -> List[ModelArtifact]:
    """Load multiple model artifacts."""
    artifacts = []
    
    for path_str in model_paths:
        path = Path(path_str)
        if not path.exists():
            # Try relative to project root
            path = PROJECT_ROOT / path_str
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path_str}")
        
        artifact = joblib.load(path)
        
        artifacts.append(ModelArtifact(
            path=str(path),
            model=artifact["model"],
            features=artifact["features"],
            params=artifact.get("best_params", {})
        ))
    
    return artifacts


# =============================================================================
# SPLIT LOGIC
# =============================================================================

def get_split_indices(n: int, split: str) -> np.ndarray:
    """
    Get indices for a specific split.
    
    Args:
        n: Total number of samples
        split: One of 'train', 'val', 'test', 'full'
        
    Returns:
        Array of indices
    """
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    if split == "train":
        return np.arange(0, train_end)
    elif split == "val":
        return np.arange(train_end, val_end)
    elif split == "test":
        return np.arange(val_end, n)
    elif split == "full":
        return np.arange(0, n)
    else:
        raise ValueError(f"Unknown split: {split}")


# =============================================================================
# SIGNAL GENERATION
# =============================================================================

def generate_signals_single_model(
    model,
    X: np.ndarray,
    threshold_long: float,
    threshold_short: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate signals from a single model.
    
    Returns:
        (signals, probabilities)
        signals: -1 (short), 0 (flat), +1 (long)
        probabilities: P(up) for each bar
    """
    proba = model.predict_proba(X)[:, 1]
    
    signals = np.zeros(len(proba), dtype=int)
    signals[proba >= threshold_long] = 1   # Long
    signals[proba <= threshold_short] = -1  # Short
    
    return signals, proba


def combine_signals_vote(
    all_signals: List[np.ndarray],
    weights: Optional[List[float]] = None
) -> np.ndarray:
    """Combine signals using majority vote."""
    if weights is None:
        weights = [1.0] * len(all_signals)
    
    # Stack signals and compute weighted sum
    stacked = np.stack(all_signals, axis=0)  # (n_models, n_samples)
    weights_arr = np.array(weights)[:, np.newaxis]
    
    weighted_sum = (stacked * weights_arr).sum(axis=0)
    
    # Determine final signal based on weighted sum
    combined = np.zeros(len(all_signals[0]), dtype=int)
    combined[weighted_sum > 0] = 1
    combined[weighted_sum < 0] = -1
    
    return combined


def combine_signals_avg(
    all_probas: List[np.ndarray],
    threshold_long: float,
    threshold_short: float,
    weights: Optional[List[float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Combine signals using averaged probability."""
    if weights is None:
        weights = [1.0] * len(all_probas)
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    # Weighted average probability
    stacked = np.stack(all_probas, axis=0)
    avg_proba = (stacked * weights[:, np.newaxis]).sum(axis=0)
    
    # Generate signals from averaged probability
    signals = np.zeros(len(avg_proba), dtype=int)
    signals[avg_proba >= threshold_long] = 1
    signals[avg_proba <= threshold_short] = -1
    
    return signals, avg_proba


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def run_funded_simulation(
    timestamps: np.ndarray,
    signals: np.ndarray,
    probas: np.ndarray,
    y_true: np.ndarray,
    risk_pct: float = 0.0025,
    max_trades: Optional[int] = None,
    starting_balance: float = STARTING_BALANCE,
    profit_target: float = PROFIT_TARGET,
    max_dd_level: float = MAX_DRAWDOWN_LEVEL
) -> SimulationResult:
    """
    Run the funded account simulation.
    
    Args:
        timestamps: Bar timestamps
        signals: Signal array (-1, 0, +1)
        probas: Probability array
        y_true: True labels (-1, +1)
        risk_pct: Risk per trade as fraction of equity
        max_trades: Optional cap on number of trades
        starting_balance: Initial equity
        profit_target: Target equity to pass
        max_dd_level: Minimum equity before failure
        
    Returns:
        SimulationResult with all metrics
    """
    equity = starting_balance
    peak_equity = starting_balance
    
    # Track metrics
    trade_results = []  # List of R multiples
    equity_history = []
    
    n_trades = 0
    status = "INCOMPLETE"
    
    for i in range(len(signals)):
        signal = signals[i]
        
        # Skip flat signals
        if signal == 0:
            continue
        
        # Skip if y_true is 0 (shouldn't happen, but safety)
        if y_true[i] == 0:
            continue
        
        # Check max trades cap
        if max_trades is not None and n_trades >= max_trades:
            break
        
        # Calculate trade outcome
        # R_multiple = signal * y_true
        # If signal matches y_true direction: +1R
        # If opposite: -1R
        r_multiple = signal * y_true[i]
        
        # Calculate PnL in dollars
        risk_dollars = equity * risk_pct
        pnl = risk_dollars * r_multiple
        
        # Update equity
        equity += pnl
        n_trades += 1
        
        # Update peak and track
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / starting_balance
        
        trade_results.append(r_multiple)
        equity_history.append({
            "timestamp": timestamps[i],
            "equity": equity,
            "drawdown_pct": drawdown * 100,
            "signal": signal,
            "proba_up": probas[i],
            "y_true": y_true[i],
            "r_multiple": r_multiple,
            "pnl": pnl,
        })
        
        # Check stop conditions
        if equity >= profit_target:
            status = "PASSED"
            break
        
        if equity <= max_dd_level:
            status = "FAILED"
            break
    
    # Build equity curve DataFrame
    if equity_history:
        equity_df = pd.DataFrame(equity_history)
    else:
        equity_df = pd.DataFrame(columns=[
            "timestamp", "equity", "drawdown_pct", "signal", 
            "proba_up", "y_true", "r_multiple", "pnl"
        ])
    
    # Calculate final metrics
    n_wins = sum(1 for r in trade_results if r > 0)
    n_losses = sum(1 for r in trade_results if r < 0)
    win_rate = n_wins / n_trades if n_trades > 0 else 0.0
    
    profit_dollars = equity - starting_balance
    profit_pct = profit_dollars / starting_balance
    
    max_dd_dollars = (peak_equity - min(equity_history, key=lambda x: x["equity"])["equity"]) if equity_history else 0
    max_dd_pct = max_dd_dollars / starting_balance
    
    avg_r = np.mean(trade_results) if trade_results else 0.0
    expectancy = profit_dollars / n_trades if n_trades > 0 else 0.0
    
    return SimulationResult(
        status=status,
        starting_equity=starting_balance,
        ending_equity=equity,
        profit_dollars=profit_dollars,
        profit_pct=profit_pct,
        n_trades=n_trades,
        n_wins=n_wins,
        n_losses=n_losses,
        win_rate=win_rate,
        max_drawdown_dollars=max_dd_dollars,
        max_drawdown_pct=max_dd_pct,
        avg_r_per_trade=avg_r,
        expectancy=expectancy,
        equity_curve=equity_df,
    )


# =============================================================================
# REPORTING
# =============================================================================

def print_report(result: SimulationResult, args) -> None:
    """Print simulation report to terminal."""
    print("\n" + "=" * 70)
    print("  FUNDED ACCOUNT SIMULATION REPORT")
    print("=" * 70)
    
    # Status with color-like emphasis
    status_str = f"  ★★★ {result.status} ★★★" if result.status == "PASSED" else \
                 f"  ✗✗✗ {result.status} ✗✗✗" if result.status == "FAILED" else \
                 f"  --- {result.status} ---"
    print(f"\n{status_str}")
    
    print(f"\n  Account Parameters:")
    print(f"    Starting Balance:  ${result.starting_equity:,.2f}")
    print(f"    Profit Target:     ${PROFIT_TARGET:,.2f} (+{PROFIT_TARGET_PCT*100:.0f}%)")
    print(f"    Max Drawdown:      ${MAX_DRAWDOWN_LEVEL:,.2f} (-{MAX_DRAWDOWN_PCT*100:.0f}%)")
    print(f"    Risk per Trade:    {args.risk_pct*100:.2f}%")
    
    print(f"\n  Strategy Parameters:")
    print(f"    Split:             {args.split}")
    print(f"    Threshold Long:    {args.threshold_long}")
    print(f"    Threshold Short:   {args.threshold_short}")
    print(f"    Combine Method:    {args.combine}")
    
    print(f"\n  Performance:")
    print(f"    Ending Equity:     ${result.ending_equity:,.2f}")
    profit_color = "+" if result.profit_pct >= 0 else ""
    print(f"    Profit:            ${result.profit_dollars:,.2f} ({profit_color}{result.profit_pct*100:.2f}%)")
    print(f"    Max Drawdown:      ${result.max_drawdown_dollars:,.2f} ({result.max_drawdown_pct*100:.2f}%)")
    
    print(f"\n  Trade Statistics:")
    print(f"    Total Trades:      {result.n_trades:,}")
    print(f"    Wins:              {result.n_wins:,}")
    print(f"    Losses:            {result.n_losses:,}")
    print(f"    Win Rate:          {result.win_rate*100:.1f}%")
    print(f"    Average R:         {result.avg_r_per_trade:+.4f}")
    print(f"    Expectancy:        ${result.expectancy:+.2f} per trade")
    
    print("\n" + "=" * 70)


def save_equity_curve(result: SimulationResult, output_path: Path) -> None:
    """Save equity curve to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Select columns for output
    df = result.equity_curve[["timestamp", "equity", "drawdown_pct", "signal", "proba_up", "y_true"]].copy()
    df.to_csv(output_path, index=False)
    print(f"\n  Equity curve saved to: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run funded account simulation for XAUUSD trading strategy"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test", "full"],
        help="Data split to evaluate on (default: test)"
    )
    
    parser.add_argument(
        "--risk_pct",
        type=float,
        default=0.0025,
        help="Risk per trade as fraction of equity (default: 0.0025 = 0.25%%)"
    )
    
    parser.add_argument(
        "--threshold_long",
        type=float,
        default=0.60,
        help="Probability threshold to go long (default: 0.60)"
    )
    
    parser.add_argument(
        "--threshold_short",
        type=float,
        default=0.40,
        help="Probability threshold to go short (default: 0.40)"
    )
    
    parser.add_argument(
        "--model_paths",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Comma-separated list of model paths (default: models/y_tb_60_hgb_tuned.joblib)"
    )
    
    parser.add_argument(
        "--combine",
        type=str,
        default="avg",
        choices=["vote", "avg"],
        help="Method to combine multiple models (default: avg)"
    )
    
    parser.add_argument(
        "--max_trades",
        type=int,
        default=None,
        help="Optional cap on number of trades"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPORTS_DIR / "funded_equity_curve.csv"),
        help="Output path for equity curve CSV"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("  FUNDED ACCOUNT SIMULATION")
    print("=" * 70)
    
    # Parse model paths
    model_paths = [p.strip() for p in args.model_paths.split(",")]
    
    print(f"\n  Loading {len(model_paths)} model(s)...")
    artifacts = load_model_artifacts(model_paths)
    for art in artifacts:
        print(f"    - {art.path}: {len(art.features)} features")
    
    # Load data
    print(f"\n  Loading features data...")
    df = load_features_data()
    print(f"    Loaded: {len(df):,} rows")
    
    # Prepare data
    # Filter out NaN and y_tb_60 == 0
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found")
    
    # Get common features across all models
    if len(artifacts) > 1:
        common_features = set(artifacts[0].features)
        for art in artifacts[1:]:
            common_features &= set(art.features)
        common_features = list(common_features)
        print(f"\n  Using {len(common_features)} common features across models")
    else:
        common_features = artifacts[0].features
    
    # Check all features exist
    missing = [f for f in common_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features: {missing[:5]}...")
    
    # Filter data
    df_clean = df.dropna(subset=common_features + [TARGET])
    df_clean = df_clean[df_clean[TARGET] != 0]
    
    print(f"    After filtering: {len(df_clean):,} rows")
    
    # Get split indices
    split_idx = get_split_indices(len(df_clean), args.split)
    df_split = df_clean.iloc[split_idx]
    
    print(f"\n  Split: {args.split}")
    print(f"    Samples: {len(df_split):,}")
    print(f"    Date range: {df_split.index.min().date()} to {df_split.index.max().date()}")
    
    # Extract arrays
    timestamps = df_split.index.values
    X = df_split[common_features].values
    y_true = df_split[TARGET].values
    
    print(f"    Labels: +1={int((y_true == 1).sum()):,}, -1={int((y_true == -1).sum()):,}")
    
    # Generate signals
    print(f"\n  Generating signals...")
    all_signals = []
    all_probas = []
    
    for art in artifacts:
        # Use the model's own features
        X_model = df_split[art.features].values
        signals, probas = generate_signals_single_model(
            art.model, X_model, args.threshold_long, args.threshold_short
        )
        all_signals.append(signals)
        all_probas.append(probas)
    
    # Combine signals if multiple models
    if len(artifacts) == 1:
        final_signals = all_signals[0]
        final_probas = all_probas[0]
    else:
        if args.combine == "vote":
            final_signals = combine_signals_vote(all_signals)
            final_probas = np.mean(all_probas, axis=0)
        else:  # avg
            final_signals, final_probas = combine_signals_avg(
                all_probas, args.threshold_long, args.threshold_short
            )
    
    n_long = (final_signals == 1).sum()
    n_short = (final_signals == -1).sum()
    n_flat = (final_signals == 0).sum()
    print(f"    Signals: LONG={n_long:,}, SHORT={n_short:,}, FLAT={n_flat:,}")
    
    # Run simulation
    print(f"\n  Running simulation...")
    result = run_funded_simulation(
        timestamps=timestamps,
        signals=final_signals,
        probas=final_probas,
        y_true=y_true,
        risk_pct=args.risk_pct,
        max_trades=args.max_trades,
    )
    
    # Print report
    print_report(result, args)
    
    # Save equity curve
    output_path = Path(args.output)
    save_equity_curve(result, output_path)
    
    print("\n" + "=" * 70)
    print("  SIMULATION COMPLETE")
    print("=" * 70 + "\n")
    
    return result


if __name__ == "__main__":
    main()

