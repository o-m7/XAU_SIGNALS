"""
Metrics computation module for the XAUUSD Signal Engine.

Provides functions for calculating trading performance metrics
and model evaluation statistics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: List[int] = [-1, 0, 1]
) -> Dict[str, float]:
    """
    Compute classification metrics for multiclass prediction.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_labels: List of class labels in order
        
    Returns:
        Dictionary containing:
            - accuracy: Overall accuracy
            - precision_macro: Macro-averaged precision
            - recall_macro: Macro-averaged recall
            - f1_macro: Macro-averaged F1 score
            - precision_per_class: Dict of class -> precision
            - recall_per_class: Dict of class -> recall
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=class_labels, 
                                            average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=class_labels,
                                      average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=class_labels,
                             average="macro", zero_division=0),
    }
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, labels=class_labels,
                                           average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, labels=class_labels,
                                     average=None, zero_division=0)
    
    metrics["precision_per_class"] = {
        label: precision_per_class[i] for i, label in enumerate(class_labels)
    }
    metrics["recall_per_class"] = {
        label: recall_per_class[i] for i, label in enumerate(class_labels)
    }
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: List[int] = [-1, 0, 1]
) -> np.ndarray:
    """
    Compute confusion matrix for multiclass prediction.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_labels: List of class labels in order
        
    Returns:
        Confusion matrix as numpy array
        Rows: true labels, Columns: predicted labels
    """
    return confusion_matrix(y_true, y_pred, labels=class_labels)


def compute_trading_metrics(trades_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute trading performance metrics from a trades DataFrame.
    
    Args:
        trades_df: DataFrame with columns:
            - pnl_ret: Net return per trade (after costs)
            - direction: 1 for long, -1 for short
            
    Returns:
        Dictionary containing:
            - total_trades: Total number of trades
            - win_rate: Percentage of profitable trades
            - avg_return: Average return per trade
            - total_return: Cumulative return
            - expected_value: E[R] = win_rate * avg_win - (1-win_rate) * avg_loss
            - sharpe_ratio: Simplified Sharpe (mean/std of returns)
            - max_drawdown: Maximum peak-to-trough drawdown
            - profit_factor: Gross profit / Gross loss
    """
    if len(trades_df) == 0:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_return": 0.0,
            "total_return": 0.0,
            "expected_value": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "profit_factor": 0.0,
        }
    
    pnl = trades_df["pnl_ret"].values
    
    # Basic stats
    total_trades = len(pnl)
    winners = pnl > 0
    losers = pnl < 0
    
    win_rate = winners.mean() if total_trades > 0 else 0.0
    avg_return = pnl.mean()
    total_return = pnl.sum()
    
    # Expected value
    avg_win = pnl[winners].mean() if winners.any() else 0.0
    avg_loss = abs(pnl[losers].mean()) if losers.any() else 0.0
    expected_value = win_rate * avg_win - (1 - win_rate) * avg_loss
    
    # Sharpe ratio (simplified, no risk-free rate)
    sharpe_ratio = pnl.mean() / pnl.std() if pnl.std() > 0 else 0.0
    
    # Max drawdown
    cumulative = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0
    
    # Profit factor
    gross_profit = pnl[winners].sum() if winners.any() else 0.0
    gross_loss = abs(pnl[losers].sum()) if losers.any() else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_return": avg_return,
        "total_return": total_return,
        "expected_value": expected_value,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "profit_factor": profit_factor,
    }


def compute_per_direction_metrics(
    trades_df: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Compute trading metrics separated by direction (long/short).
    
    Args:
        trades_df: DataFrame with 'direction' and 'pnl_ret' columns
        
    Returns:
        Dict with 'long' and 'short' sub-dicts of metrics
    """
    result = {}
    
    for direction, name in [(1, "long"), (-1, "short")]:
        subset = trades_df[trades_df["direction"] == direction]
        result[name] = compute_trading_metrics(subset)
    
    return result


def compute_rolling_metrics(
    trades_df: pd.DataFrame,
    window: int = 50
) -> pd.DataFrame:
    """
    Compute rolling window metrics over trades.
    
    Args:
        trades_df: DataFrame with 'pnl_ret' and timestamp index
        window: Rolling window size
        
    Returns:
        DataFrame with rolling metrics:
            - rolling_win_rate
            - rolling_avg_return
            - rolling_sharpe
    """
    pnl = trades_df["pnl_ret"]
    
    rolling_win_rate = (pnl > 0).rolling(window).mean()
    rolling_avg_return = pnl.rolling(window).mean()
    rolling_sharpe = pnl.rolling(window).apply(
        lambda x: x.mean() / x.std() if x.std() > 0 else 0.0,
        raw=True
    )
    
    return pd.DataFrame({
        "rolling_win_rate": rolling_win_rate,
        "rolling_avg_return": rolling_avg_return,
        "rolling_sharpe": rolling_sharpe,
    }, index=trades_df.index)


def compute_class_balance(labels: pd.Series) -> Dict[int, float]:
    """
    Compute class distribution for labels.
    
    Args:
        labels: Series of labels (-1, 0, 1)
        
    Returns:
        Dict mapping class -> percentage
    """
    counts = labels.value_counts(normalize=True)
    return {int(k): float(v) for k, v in counts.items()}


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: List[int] = [-1, 0, 1],
    target_names: List[str] = ["Short", "Flat", "Long"]
) -> str:
    """
    Generate a formatted classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_labels: List of class labels
        target_names: Human-readable names for classes
        
    Returns:
        Formatted classification report string
    """
    return classification_report(
        y_true, y_pred,
        labels=class_labels,
        target_names=target_names,
        zero_division=0
    )

