#!/usr/bin/env python3
"""
Feature Engineering for Intraday XAUUSD Strategy

Based on validated edges from research:
1. Mean Reversion Edge (lag-1 ACF = -0.0385, p < 0.0001)
2. Session Effects (ANOVA p = 0.001)
3. Volatility Clustering (ARCH p < 0.0001)

Design Principles:
- All features computed on bar close (no lookahead bias)
- Capture validated statistical edges
- Multi-timeframe where beneficial
- Normalized for regime independence

Author: Quant Research Team
Date: 2026-01-06
"""

import pandas as pd
import numpy as np
from pathlib import Path


def add_price_features(df, lookbacks=[5, 10, 20, 50]):
    """
    Price-based features capturing mean reversion dynamics.

    Features:
    - ROC (Rate of Change) at multiple lookbacks
    - Distance from moving averages (z-score)
    - Price percentile rank
    - Consecutive up/down bars
    """
    df = df.copy()

    # 1. Rate of Change (captures recent moves for reversion)
    for lb in lookbacks:
        df[f'roc_{lb}'] = (df['close'] / df['close'].shift(lb) - 1) * 100

    # 2. Distance from MA (mean reversion signal)
    for lb in lookbacks:
        df[f'ma_{lb}'] = df['close'].rolling(lb).mean()
        df[f'dist_ma_{lb}'] = (df['close'] - df[f'ma_{lb}']) / df[f'ma_{lb}'] * 100

        # Z-score (standardized distance)
        rolling_std = df['close'].rolling(lb).std()
        df[f'zscore_ma_{lb}'] = (df['close'] - df[f'ma_{lb}']) / rolling_std

    # 3. Price percentile rank (where is price relative to recent range?)
    for lb in [20, 50, 100]:
        df[f'pct_rank_{lb}'] = df['close'].rolling(lb).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )

    # 4. Consecutive up/down bars (streak detection)
    df['bar_direction'] = np.sign(df['close'] - df['open'])
    df['consec_up'] = (df['bar_direction'] == 1).astype(int)
    df['consec_down'] = (df['bar_direction'] == -1).astype(int)

    # Count consecutive
    for col in ['consec_up', 'consec_down']:
        groups = (df[col] != df[col].shift()).cumsum()
        df[col] = df.groupby(groups)[col].cumsum()

    return df


def add_momentum_features(df):
    """
    Momentum and reversal features.

    Even though we found mean reversion, we need to know the strength
    of the move we're reverting from.
    """
    df = df.copy()

    # Recent returns
    for lag in [1, 2, 3, 5, 10]:
        df[f'ret_lag{lag}'] = df['returns'].shift(lag)

    # Cumulative returns (strength of recent move)
    for window in [3, 5, 10]:
        df[f'cum_ret_{window}'] = df['returns'].rolling(window).sum()

    # Reversal signal (negative of recent return - for mean reversion)
    df['reversion_signal'] = -df['returns'].shift(1)  # Opposite of last move

    # Strength of recent move (absolute)
    df['move_strength'] = df['returns'].shift(1).abs()

    return df


def add_volatility_features(df):
    """
    Volatility features (for position sizing and regime awareness).

    Based on ARCH effects (volatility clustering).
    """
    df = df.copy()

    # Realized volatility (rolling std)
    for window in [5, 10, 20, 50]:
        df[f'rvol_{window}'] = df['returns'].rolling(window).std()

    # ATR (Average True Range)
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )

    for window in [5, 10, 20]:
        df[f'atr_{window}'] = df['tr'].rolling(window).mean()
        df[f'atr_pct_{window}'] = df[f'atr_{window}'] / df['close'] * 100

    # Volatility rank (percentile of current vol)
    for window in [20, 50]:
        df[f'vol_rank_{window}'] = df[f'rvol_{window}'].rolling(window).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )

    # Parkinson volatility (high-low estimator)
    df['parkinson_vol'] = np.sqrt(1 / (4 * np.log(2)) * np.log(df['high'] / df['low'])**2)
    df['parkinson_vol_ma'] = df['parkinson_vol'].rolling(20).mean()

    return df


def add_volume_features(df):
    """Volume and trade-based features."""
    df = df.copy()

    # Volume moving averages
    for window in [5, 10, 20]:
        df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
        df[f'volume_ratio_{window}'] = df['volume'] / df[f'volume_ma_{window}']

    # Volume percentile rank
    df['volume_pct_rank'] = df['volume'].rolling(50).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
    )

    # Trade intensity (trades per minute)
    if 'trades' in df.columns:
        for window in [5, 10, 20]:
            df[f'trades_ma_{window}'] = df['trades'].rolling(window).mean()

    return df


def add_microstructure_features(df):
    """
    Microstructure features (if quotes data available).

    Note: This requires bid/ask data from quotes
    """
    df = df.copy()

    # OHLC-based microstructure proxies

    # 1. Close position in bar (where did we close relative to high-low?)
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    df['close_position'] = df['close_position'].fillna(0.5)  # Handle zero range

    # 2. Upper/lower shadows
    df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'])
    df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['high'] - df['low'])
    df['upper_shadow'] = df['upper_shadow'].fillna(0)
    df['lower_shadow'] = df['lower_shadow'].fillna(0)

    # 3. Body size
    df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    df['body_size'] = df['body_size'].fillna(0)

    # 4. Buying/selling pressure proxy
    df['buying_pressure'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['buying_pressure'] = df['buying_pressure'].fillna(0)

    return df


def add_session_features(df):
    """
    Session-based features.

    Based on validated session effects (ANOVA p = 0.001).
    """
    df = df.copy()

    # Extract time features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # Define sessions
    def get_session(hour):
        if 0 <= hour < 7:
            return 0  # Asia/Pacific
        elif 7 <= hour < 16:
            return 1  # London
        elif 16 <= hour < 22:
            return 2  # NY
        else:
            return 3  # After-hours

    df['session'] = df['hour'].apply(get_session)

    # One-hot encode sessions
    df['session_asia'] = (df['session'] == 0).astype(int)
    df['session_london'] = (df['session'] == 1).astype(int)
    df['session_ny'] = (df['session'] == 2).astype(int)
    df['session_afterhours'] = (df['session'] == 3).astype(int)

    # Time since session start
    df['minutes_since_midnight'] = df.index.hour * 60 + df.index.minute

    return df


def add_multi_timeframe_features(df, timeframes=['30T', '1H']):
    """
    Multi-timeframe features (optional - test if beneficial).

    Higher timeframe context for 15-minute decisions.
    """
    df = df.copy()
    df_orig_freq = df.copy()

    for tf in timeframes:
        # Resample to higher timeframe
        df_htf = df_orig_freq[['open', 'high', 'low', 'close', 'volume']].resample(tf).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Calculate HTF features
        df_htf[f'htf_{tf}_roc'] = df_htf['close'].pct_change()
        df_htf[f'htf_{tf}_ma20'] = df_htf['close'].rolling(20).mean()
        df_htf[f'htf_{tf}_dist_ma'] = (df_htf['close'] - df_htf[f'htf_{tf}_ma20']) / df_htf[f'htf_{tf}_ma20']

        # Merge back to original timeframe (forward-fill)
        for col in [f'htf_{tf}_roc', f'htf_{tf}_dist_ma']:
            if col in df_htf.columns:
                df = df.merge(
                    df_htf[[col]],
                    left_index=True,
                    right_index=True,
                    how='left'
                )
                df[col] = df[col].fillna(method='ffill')

    return df


def engineer_all_features(df, include_multi_tf=False):
    """
    Apply all feature engineering.

    Args:
        df: DataFrame with OHLCV + returns
        include_multi_tf: Whether to include multi-timeframe features

    Returns:
        DataFrame with all features
    """
    print("Feature Engineering...")
    print(f"  Input shape: {df.shape}")

    # Add features
    df = add_price_features(df)
    print(f"  After price features: {df.shape}")

    df = add_momentum_features(df)
    print(f"  After momentum features: {df.shape}")

    df = add_volatility_features(df)
    print(f"  After volatility features: {df.shape}")

    df = add_volume_features(df)
    print(f"  After volume features: {df.shape}")

    df = add_microstructure_features(df)
    print(f"  After microstructure features: {df.shape}")

    df = add_session_features(df)
    print(f"  After session features: {df.shape}")

    if include_multi_tf:
        df = add_multi_timeframe_features(df)
        print(f"  After multi-timeframe features: {df.shape}")

    # Drop NaN rows from feature calculation
    initial_rows = len(df)
    df = df.dropna()
    print(f"  Dropped {initial_rows - len(df)} NaN rows")
    print(f"  Final shape: {df.shape}")

    return df


def get_feature_columns(df):
    """
    Extract feature column names (exclude OHLCV, timestamps, labels).

    Returns:
        List of feature column names
    """
    exclude_cols = [
        'open', 'high', 'low', 'close', 'volume', 'vwap', 'trades',
        'returns', 'log_returns', 'returns_bps',
        'timestamp', 'date', 'time',
        'label', 'target', 'forward_return',
        # Intermediate columns
        'ma_5', 'ma_10', 'ma_20', 'ma_50',  # MA are intermediate, not features
        'bar_direction', 'tr'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('Unnamed')]

    return feature_cols


if __name__ == "__main__":
    # Test feature engineering
    from pathlib import Path

    RESULTS_DIR = Path("/Users/omar/Desktop/ML/xauusd_signals/research_results")

    # Load data
    df = pd.read_parquet(RESULTS_DIR / "data_15min_2020_2024.parquet")

    print("=" * 80)
    print("FEATURE ENGINEERING TEST")
    print("=" * 80)
    print()

    # Engineer features
    df_features = engineer_all_features(df, include_multi_tf=False)

    # Get feature names
    feature_cols = get_feature_columns(df_features)

    print()
    print("=" * 80)
    print(f"FEATURE ENGINEERING COMPLETE")
    print("=" * 80)
    print(f"Total features: {len(feature_cols)}")
    print()
    print("Feature categories:")

    # Categorize features
    categories = {
        'Price/Mean Reversion': [c for c in feature_cols if any(x in c for x in ['roc', 'dist_ma', 'zscore', 'pct_rank', 'consec'])],
        'Momentum': [c for c in feature_cols if 'ret_lag' in c or 'cum_ret' in c or 'reversion' in c or 'move_strength' in c],
        'Volatility': [c for c in feature_cols if any(x in c for x in ['vol', 'atr', 'parkinson'])],
        'Volume': [c for c in feature_cols if 'volume' in c or 'trades' in c],
        'Microstructure': [c for c in feature_cols if any(x in c for x in ['close_position', 'shadow', 'body', 'pressure'])],
        'Session/Time': [c for c in feature_cols if any(x in c for x in ['hour', 'day_of_week', 'month', 'session', 'minutes'])]
    }

    for cat, cols in categories.items():
        print(f"  {cat}: {len(cols)}")

    # Save
    output_path = RESULTS_DIR / "data_15min_2020_2024_features.parquet"
    df_features.to_parquet(output_path)
    print()
    print(f"Saved to: {output_path}")
    print()
