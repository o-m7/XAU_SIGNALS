"""
Evaluation module for the XAUUSD Signal Engine.

Dr. Chen Style Implementation:
==============================
Evaluates the TWO-STAGE system separately:
1. Environment model quality (AUC, precision, recall)
2. Direction determination quality (accuracy given good env)
3. Combined system performance (P&L, risk metrics)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


def evaluate_environment_model(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]
) -> Dict:
    """
    Evaluate the binary environment classification model.
    
    Key metrics:
    - AUC-ROC: Discriminative ability
    - Precision at threshold: When model says "good env", how often correct?
    - Recall at threshold: Of all good envs, how many found?
    """
    from sklearn.metrics import roc_auc_score, precision_score, recall_score
    
    results = {
        "auc_roc": roc_auc_score(y_true, y_pred_proba),
        "threshold_analysis": {},
    }
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        n_predicted = y_pred.sum()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        
        results["threshold_analysis"][thresh] = {
            "n_predicted": int(n_predicted),
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
        }
    
    return results


def evaluate_direction_accuracy(trades_df: pd.DataFrame) -> Dict:
    """
    Evaluate direction determination quality.
    
    Given that the model said "good environment",
    how well did microstructure determine direction?
    """
    if len(trades_df) == 0:
        return {"error": "No trades"}
    
    # Overall direction accuracy
    correct = (trades_df["pnl_ret"] > 0).sum()
    total = len(trades_df)
    
    # By direction
    long_trades = trades_df[trades_df["signal"] == 1]
    short_trades = trades_df[trades_df["signal"] == -1]
    
    long_correct = (long_trades["pnl_ret"] > 0).sum() if len(long_trades) > 0 else 0
    short_correct = (short_trades["pnl_ret"] > 0).sum() if len(short_trades) > 0 else 0
    
    return {
        "overall_accuracy": correct / total,
        "long_accuracy": long_correct / len(long_trades) if len(long_trades) > 0 else 0,
        "short_accuracy": short_correct / len(short_trades) if len(short_trades) > 0 else 0,
        "long_count": len(long_trades),
        "short_count": len(short_trades),
        "balance_ratio": len(long_trades) / len(short_trades) if len(short_trades) > 0 else float('inf'),
    }


def compute_r_statistics(trades_df: pd.DataFrame) -> Dict:
    """
    Compute R-multiple statistics.
    
    R = return / risk (SL distance)
    This normalizes returns by the risk taken.
    """
    if len(trades_df) == 0 or "r_multiple" not in trades_df.columns:
        return {"error": "No R data"}
    
    r = trades_df["r_multiple"]
    
    return {
        "total_r": r.sum(),
        "mean_r": r.mean(),
        "median_r": r.median(),
        "std_r": r.std(),
        "max_r": r.max(),
        "min_r": r.min(),
        "positive_r_count": (r > 0).sum(),
        "negative_r_count": (r < 0).sum(),
        "win_rate": (r > 0).mean(),
        "profit_factor": r[r > 0].sum() / abs(r[r < 0].sum()) if (r < 0).any() else float('inf'),
    }


def compute_risk_metrics(trades_df: pd.DataFrame) -> Dict:
    """
    Compute risk-related metrics.
    """
    if len(trades_df) == 0:
        return {"error": "No trades"}
    
    returns = trades_df["pnl_ret"]
    
    # Cumulative returns
    cumulative = returns.cumsum()
    
    # Max drawdown
    running_max = cumulative.cummax()
    drawdowns = running_max - cumulative
    max_drawdown = drawdowns.max()
    
    # Sharpe (annualized)
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(252 * 24 * 12) if std_ret > 0 else 0  # 5-min bars
    
    # Sortino (downside deviation)
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 0 else 0.0001
    sortino = (mean_ret / downside_std) * np.sqrt(252 * 24 * 12) if downside_std > 0 else 0
    
    # Calmar
    calmar = (returns.sum() / max_drawdown) if max_drawdown > 0 else float('inf')
    
    return {
        "total_return": returns.sum(),
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "volatility": std_ret,
        "skew": returns.skew(),
        "kurtosis": returns.kurtosis(),
    }


def format_evaluation_summary(
    env_eval: Dict,
    direction_eval: Dict,
    r_stats: Dict,
    risk_metrics: Dict,
    horizon: str
) -> str:
    """
    Format comprehensive evaluation summary.
    """
    lines = [
        f"EVALUATION SUMMARY: {horizon}",
        "=" * 60,
        "",
        "ENVIRONMENT MODEL:",
        f"  AUC-ROC: {env_eval.get('auc_roc', 0):.3f}",
    ]
    
    if "threshold_analysis" in env_eval:
        for thresh, stats in env_eval["threshold_analysis"].items():
            lines.append(
                f"  @ {thresh}: Precision={stats['precision']:.1%}, "
                f"Recall={stats['recall']:.1%}, "
                f"N={stats['n_predicted']:,}"
            )
    
    lines.extend([
        "",
        "DIRECTION DETERMINATION:",
        f"  Overall Accuracy: {direction_eval.get('overall_accuracy', 0):.1%}",
        f"  Long Accuracy:    {direction_eval.get('long_accuracy', 0):.1%}",
        f"  Short Accuracy:   {direction_eval.get('short_accuracy', 0):.1%}",
        f"  Balance Ratio:    {direction_eval.get('balance_ratio', 0):.2f} (L/S)",
        "",
        "R-MULTIPLES:",
        f"  Total R:       {r_stats.get('total_r', 0):.2f}",
        f"  Mean R:        {r_stats.get('mean_r', 0):.3f}",
        f"  Win Rate:      {r_stats.get('win_rate', 0):.1%}",
        f"  Profit Factor: {r_stats.get('profit_factor', 0):.2f}",
        "",
        "RISK METRICS:",
        f"  Max Drawdown:  {risk_metrics.get('max_drawdown', 0)*100:.2f}%",
        f"  Sharpe Ratio:  {risk_metrics.get('sharpe', 0):.2f}",
        f"  Sortino Ratio: {risk_metrics.get('sortino', 0):.2f}",
        "",
    ])
    
    return "\n".join(lines)
