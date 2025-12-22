"""
Plotting utility functions for the XAUUSD Signal Engine.

Provides visualization helpers for features, labels, signals,
and model performance analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def set_plotting_style():
    """Set consistent plotting style for all visualizations."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 100,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })


def plot_feature_distributions(
    df: pd.DataFrame,
    feature_cols: List[str],
    ncols: int = 3,
    figsize: Tuple[int, int] = (15, 12),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot histograms of feature distributions.
    
    Args:
        df: DataFrame containing features
        feature_cols: List of feature column names
        ncols: Number of columns in subplot grid
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    n_features = len(feature_cols)
    nrows = (n_features + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for i, col in enumerate(feature_cols):
        if col in df.columns:
            data = df[col].dropna()
            axes[i].hist(data, bins=50, edgecolor="black", alpha=0.7)
            axes[i].set_title(col)
            axes[i].set_xlabel("Value")
            axes[i].set_ylabel("Frequency")
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_label_distribution(
    df: pd.DataFrame,
    label_cols: List[str] = ["y_5m", "y_15m", "y_30m"],
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot bar charts of label distributions for each horizon.
    
    Args:
        df: DataFrame containing labels
        label_cols: List of label column names
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, len(label_cols), figsize=figsize)
    
    class_names = {-1: "Short", 0: "Flat", 1: "Long"}
    colors = {-1: "#e74c3c", 0: "#95a5a6", 1: "#27ae60"}
    
    for i, col in enumerate(label_cols):
        if col in df.columns:
            counts = df[col].value_counts().sort_index()
            bars = axes[i].bar(
                [class_names.get(k, str(k)) for k in counts.index],
                counts.values,
                color=[colors.get(k, "#3498db") for k in counts.index],
                edgecolor="black"
            )
            axes[i].set_title(f"Label Distribution: {col}")
            axes[i].set_xlabel("Class")
            axes[i].set_ylabel("Count")
            
            # Add percentage labels
            total = counts.sum()
            for bar, val in zip(bars, counts.values):
                pct = 100 * val / total
                axes[i].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(counts.values),
                    f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize=10
                )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = ["Short", "Flat", "Long"],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    
    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_feature_importance(
    importance_dict: Dict[str, float],
    top_n: int = 20,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot horizontal bar chart of feature importances.
    
    Args:
        importance_dict: Dictionary mapping feature name to importance
        top_n: Number of top features to show
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    # Sort and take top N
    sorted_importance = sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    features = [x[0] for x in sorted_importance]
    importances = [x[1] for x in sorted_importance]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(features))
    ax.barh(y_pos, importances, color="#3498db", edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_equity_curve(
    trades_df: pd.DataFrame,
    title: str = "Equity Curve",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot cumulative returns (equity curve) from trades.
    
    Args:
        trades_df: DataFrame with 'pnl_ret' column and datetime index
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    cumulative = trades_df["pnl_ret"].cumsum()
    ax.plot(cumulative.index, cumulative.values, color="#2c3e50", linewidth=1.5)
    ax.fill_between(
        cumulative.index, 0, cumulative.values,
        where=cumulative.values >= 0, color="#27ae60", alpha=0.3
    )
    ax.fill_between(
        cumulative.index, 0, cumulative.values,
        where=cumulative.values < 0, color="#e74c3c", alpha=0.3
    )
    
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Cumulative Return")
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_price_with_signals(
    df: pd.DataFrame,
    signals_df: pd.DataFrame,
    price_col: str = "mid",
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (14, 7),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot price with signal markers overlaid.
    
    Args:
        df: DataFrame with price data
        signals_df: DataFrame with 'signal' column (-1, 0, 1)
        price_col: Column name for price
        start_idx: Start index for plotting
        end_idx: End index for plotting
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    if end_idx is None:
        end_idx = len(df)
    
    plot_df = df.iloc[start_idx:end_idx].copy()
    plot_signals = signals_df.iloc[start_idx:end_idx].copy()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot price
    ax.plot(plot_df.index, plot_df[price_col], color="#2c3e50", 
            linewidth=1, label="Mid Price")
    
    # Plot signals
    if "signal" in plot_signals.columns:
        long_signals = plot_signals[plot_signals["signal"] == 1]
        short_signals = plot_signals[plot_signals["signal"] == -1]
        
        ax.scatter(
            long_signals.index,
            plot_df.loc[long_signals.index, price_col],
            marker="^", color="#27ae60", s=50, label="Long", zorder=5
        )
        ax.scatter(
            short_signals.index,
            plot_df.loc[short_signals.index, price_col],
            marker="v", color="#e74c3c", s=50, label="Short", zorder=5
        )
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.set_title("Price with Trading Signals")
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig


def plot_rolling_metrics(
    rolling_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot rolling performance metrics.
    
    Args:
        rolling_df: DataFrame with rolling metrics columns
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    if "rolling_win_rate" in rolling_df.columns:
        axes[0].plot(rolling_df.index, rolling_df["rolling_win_rate"], 
                     color="#3498db", linewidth=1)
        axes[0].axhline(y=0.5, color="red", linestyle="--", linewidth=0.5)
        axes[0].set_ylabel("Win Rate")
        axes[0].set_title("Rolling Win Rate")
    
    if "rolling_avg_return" in rolling_df.columns:
        axes[1].plot(rolling_df.index, rolling_df["rolling_avg_return"],
                     color="#27ae60", linewidth=1)
        axes[1].axhline(y=0, color="red", linestyle="--", linewidth=0.5)
        axes[1].set_ylabel("Avg Return")
        axes[1].set_title("Rolling Average Return")
    
    if "rolling_sharpe" in rolling_df.columns:
        axes[2].plot(rolling_df.index, rolling_df["rolling_sharpe"],
                     color="#9b59b6", linewidth=1)
        axes[2].axhline(y=0, color="red", linestyle="--", linewidth=0.5)
        axes[2].set_ylabel("Sharpe")
        axes[2].set_title("Rolling Sharpe Ratio")
        axes[2].set_xlabel("Time")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig

