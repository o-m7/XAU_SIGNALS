"""
features.py - Statistically Grounded Feature Engineering

Every feature has:
1. Mathematical formula
2. Statistical interpretation
3. Testable hypothesis
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import List, Dict, Optional, Tuple
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback decorator
    def njit(func=None, *args, **kwargs):
        if func is None:
            def decorator(f):
                return f
            return decorator
        return func

from .config import Model5Config


# =============================================================================
# NUMBA-OPTIMIZED CORE FUNCTIONS
# =============================================================================

@njit
def realized_variance(returns: np.ndarray) -> float:
    """
    Realized variance: RV = Σ r_i²
    Consistent estimator of integrated variance.
    """
    return np.sum(returns ** 2)


@njit
def bipower_variation(returns: np.ndarray) -> float:
    """
    Bipower variation: BV = (π/2) * Σ |r_{i-1}| * |r_i|
    Robust to jumps - estimates continuous variation only.
    """
    n = len(returns)
    if n < 2:
        return 0.0
    abs_r = np.abs(returns)
    return (np.pi / 2) * np.sum(abs_r[:-1] * abs_r[1:])


@njit
def variance_ratio(returns: np.ndarray, q: int) -> float:
    """
    Variance ratio: VR(q) = Var(r^(q)) / (q * Var(r^(1)))
    
    Under random walk: VR = 1
    Mean reversion: VR < 1
    Momentum: VR > 1
    
    Reference: Lo & MacKinlay (1988)
    """
    n = len(returns)
    if n < q + 5:
        return 1.0
    
    # 1-period variance
    var_1 = np.var(returns)
    if var_1 == 0:
        return 1.0
    
    # q-period returns
    returns_q = np.zeros(n - q + 1)
    for i in range(n - q + 1):
        returns_q[i] = np.sum(returns[i:i+q])
    
    var_q = np.var(returns_q)
    return var_q / (q * var_1)


@njit
def parkinson_volatility(high: np.ndarray, low: np.ndarray) -> float:
    """
    Parkinson volatility: σ_P = sqrt(1/(4*ln(2)*n) * Σ ln(H/L)²)
    More efficient than close-to-close volatility.
    Reference: Parkinson (1980)
    """
    n = len(high)
    if n == 0:
        return 0.0
    
    sum_sq = 0.0
    for i in range(n):
        if low[i] > 0:
            log_hl = np.log(high[i] / low[i])
            sum_sq += log_hl ** 2
    
    return np.sqrt(sum_sq / (4 * np.log(2) * n))


@njit
def count_sign_changes(returns: np.ndarray) -> float:
    """
    Count sign changes in returns (reversal frequency).
    Returns proportion of reversals [0, 1].
    """
    n = len(returns)
    if n < 2:
        return 0.0
    
    changes = 0
    for i in range(1, n):
        if returns[i] * returns[i-1] < 0:
            changes += 1
    
    return changes / (n - 1)


@njit
def linear_trend_slope(values: np.ndarray) -> float:
    """
    OLS slope of values over time, normalized by first value.
    """
    n = len(values)
    if n < 2 or values[0] == 0:
        return 0.0
    
    x = np.arange(n, dtype=np.float64)
    x_mean = np.mean(x)
    y_mean = np.mean(values)
    
    num = 0.0
    den = 0.0
    for i in range(n):
        num += (x[i] - x_mean) * (values[i] - y_mean)
        den += (x[i] - x_mean) ** 2
    
    if den == 0:
        return 0.0
    
    slope = num / den
    return slope / values[0]  # Normalize


# =============================================================================
# PANDAS-BASED ROLLING FUNCTIONS
# =============================================================================

def compute_zscore(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling z-score: z_t = (x_t - μ_t) / σ_t
    """
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def compute_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling percentile rank [0, 100] (optimized).
    """
    # Use pandas built-in rank which is much faster
    rolling_rank = series.rolling(window).rank(pct=True) * 100
    return rolling_rank.fillna(50.0)


def rolling_autocorr(series: pd.Series, lag: int, window: int) -> pd.Series:
    """
    Rolling autocorrelation at specified lag (optimized).
    """
    # Use vectorized approach for lag-1 autocorrelation
    if lag == 1:
        # For lag-1, we can use a faster vectorized method
        shifted = series.shift(lag)
        rolling_mean = series.rolling(window).mean()
        rolling_std = series.rolling(window).std()
        
        # Center the series
        centered = series - rolling_mean
        centered_lag = shifted - rolling_mean
        
        # Autocorrelation
        autocorr = (centered * centered_lag).rolling(window).sum() / (rolling_std ** 2 * window)
        return autocorr.fillna(0)
    else:
        # For other lags, use apply (slower but necessary)
        # NOTE: With raw=False, x is already a Series, so no need to wrap
        def autocorr_func(x):
            if len(x) < lag + 2:
                return 0.0
            autocorr_val = x.autocorr(lag=lag)
            return autocorr_val if not pd.isna(autocorr_val) else 0.0

        return series.rolling(window).apply(autocorr_func, raw=False)


def rolling_variance_ratio(returns: pd.Series, q: int, window: int) -> pd.Series:
    """
    Rolling variance ratio VR(q) (optimized).
    """
    # Use vectorized calculation where possible
    if q == 2:
        # Optimized for q=2
        var_1 = returns.rolling(window).var()
        returns_2 = returns + returns.shift(1)
        var_2 = returns_2.rolling(window).var()
        vr = var_2 / (2 * var_1.replace(0, np.nan))
        return vr.fillna(1.0)
    else:
        # For other q, use apply
        def vr_func(x):
            return variance_ratio(x, q)
        
        return returns.rolling(window).apply(vr_func, raw=True)


def rolling_parkinson(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """
    Rolling Parkinson volatility (vectorized for speed).
    """
    # Vectorized calculation
    log_hl = np.log(high / low.replace(0, np.nan))
    log_hl_sq = log_hl ** 2
    result = np.sqrt(log_hl_sq.rolling(window).sum() / (4 * np.log(2) * window))
    return result


# =============================================================================
# MAIN FEATURE BUILDERS
# =============================================================================

def build_15m_features(df: pd.DataFrame, config=None, show_progress: bool = False) -> pd.DataFrame:
    """
    Build features from 15-minute OHLCV data.
    
    Args:
        df: DataFrame with [open, high, low, close, volume] and DatetimeIndex
        config: Model5Config
        show_progress: Show progress bar
    
    Returns:
        DataFrame with all 15M-derived features
    """
    if config is None:
        config = Model5Config()
    
    df = df.copy()
    
    if show_progress:
        from tqdm import tqdm
        tqdm.pandas(desc="Building features")
    
    # ==================== RETURNS ====================
    if show_progress:
        print("   Computing returns...")
    df['returns_1'] = df['close'].pct_change(1)
    df['returns_2'] = df['close'].pct_change(2)
    df['returns_4'] = df['close'].pct_change(4)
    
    # ==================== Z-SCORE & PERCENTILE ====================
    if show_progress:
        print("   Computing z-scores and percentiles...")
    df['zscore_20'] = compute_zscore(df['close'], window=20)
    df['zscore_50'] = compute_zscore(df['close'], window=50)
    df['percentile_20'] = compute_percentile_rank(df['close'], window=20)
    df['percentile_50'] = compute_percentile_rank(df['close'], window=50)
    
    # ==================== VOLATILITY ====================
    if show_progress:
        print("   Computing volatility measures...")
    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift(1)).abs(),
        (df['low'] - df['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr_14'] = tr.ewm(span=14, adjust=False).mean()
    
    # Parkinson volatility
    df['parkinson_vol_20'] = rolling_parkinson(df['high'], df['low'], window=20)
    
    # Returns std
    df['returns_std_20'] = df['returns_1'].rolling(20).std()
    
    # ATR percentile (for filtering)
    df['atr_percentile'] = compute_percentile_rank(df['atr_14'], window=100)
    
    # ==================== VARIANCE RATIOS ====================
    if show_progress:
        print("   Computing variance ratios (this may take a while)...")
    df['variance_ratio_2'] = rolling_variance_ratio(df['returns_1'].fillna(0), q=2, window=50)
    df['variance_ratio_4'] = rolling_variance_ratio(df['returns_1'].fillna(0), q=4, window=50)
    
    # ==================== AUTOCORRELATION ====================
    if show_progress:
        print("   Computing autocorrelations...")
    df['autocorr_1'] = rolling_autocorr(df['returns_1'], lag=1, window=50)
    df['autocorr_2'] = rolling_autocorr(df['returns_1'], lag=2, window=50)
    
    # ==================== CANDLE STRUCTURE ====================
    df['range'] = df['high'] - df['low']
    df['body'] = (df['close'] - df['open']).abs()
    df['body_ratio'] = df['body'] / df['range'].replace(0, np.nan)
    
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_wick_ratio'] = df['upper_wick'] / df['range'].replace(0, np.nan)
    df['lower_wick_ratio'] = df['lower_wick'] / df['range'].replace(0, np.nan)
    df['close_location'] = (df['close'] - df['low']) / df['range'].replace(0, np.nan)
    
    # ==================== DISTRIBUTION ====================
    df['returns_skew_20'] = df['returns_1'].rolling(20).skew()
    df['returns_kurtosis_20'] = df['returns_1'].rolling(20).kurt()
    
    # ==================== TIME ====================
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['is_london'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
    df['is_ny'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
    df['is_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
    df['is_active'] = df['hour'].isin(config.active_hours).astype(int)
    
    # ==================== CLEANUP ====================
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df


def build_quote_features(
    df_15m: pd.DataFrame,
    df_quotes: pd.DataFrame,
    spread_lookback: int = 100
) -> pd.DataFrame:
    """
    Build features from top-of-book quote data.
    
    Args:
        df_15m: 15-minute bars with DatetimeIndex
        df_quotes: Quote data with columns:
            - participant_timestamp (nanoseconds) OR timestamp (datetime)
            - bid_price
            - ask_price
    
    Returns:
        df_15m with quote features added
    """
    df = df_15m.copy()
    
    # Prepare quotes index
    if 'participant_timestamp' in df_quotes.columns:
        df_quotes = df_quotes.copy()
        df_quotes['timestamp'] = pd.to_datetime(df_quotes['participant_timestamp'], unit='ns')
        df_quotes = df_quotes.set_index('timestamp')
    elif not isinstance(df_quotes.index, pd.DatetimeIndex):
        if 'timestamp' in df_quotes.columns:
            df_quotes = df_quotes.set_index('timestamp')
    
    # Initialize columns
    quote_cols = [
        'spread_mean_bps', 'spread_std_bps', 'spread_max_bps',
        'quote_count', 'quote_intensity',
        'bid_updates', 'ask_updates', 'quote_imbalance',
        'inter_quote_mean_ms', 'inter_quote_std_ms'
    ]
    for col in quote_cols:
        df[col] = np.nan
    
    # Track spread history for z-score
    spread_history = []
    
    for bar_time in df.index:
        bar_start = bar_time - pd.Timedelta(minutes=15)
        bar_end = bar_time
        
        mask = (df_quotes.index > bar_start) & (df_quotes.index <= bar_end)
        bar_quotes = df_quotes.loc[mask]
        
        if len(bar_quotes) < 2:
            spread_history.append(np.nan)
            continue
        
        bar_quotes = bar_quotes.copy()
        
        # Spread calculations
        bar_quotes['spread'] = bar_quotes['ask_price'] - bar_quotes['bid_price']
        bar_quotes['mid'] = (bar_quotes['ask_price'] + bar_quotes['bid_price']) / 2
        bar_quotes['spread_bps'] = bar_quotes['spread'] / bar_quotes['mid'] * 10000
        
        spread_mean = bar_quotes['spread_bps'].mean()
        spread_std = bar_quotes['spread_bps'].std()
        spread_max = bar_quotes['spread_bps'].max()
        
        # Update counts
        bid_updates = (bar_quotes['bid_price'].diff().abs() > 1e-10).sum()
        ask_updates = (bar_quotes['ask_price'].diff().abs() > 1e-10).sum()
        total_updates = bid_updates + ask_updates
        
        # Inter-quote times
        inter_times = bar_quotes.index.to_series().diff().dt.total_seconds() * 1000
        inter_times = inter_times.dropna()
        
        # Quote intensity
        duration = (bar_end - bar_start).total_seconds()
        intensity = len(bar_quotes) / duration if duration > 0 else 0
        
        # Store
        spread_history.append(spread_mean)
        
        df.loc[bar_time, 'spread_mean_bps'] = spread_mean
        df.loc[bar_time, 'spread_std_bps'] = spread_std
        df.loc[bar_time, 'spread_max_bps'] = spread_max
        df.loc[bar_time, 'quote_count'] = len(bar_quotes)
        df.loc[bar_time, 'quote_intensity'] = intensity
        df.loc[bar_time, 'bid_updates'] = bid_updates
        df.loc[bar_time, 'ask_updates'] = ask_updates
        df.loc[bar_time, 'quote_imbalance'] = (ask_updates - bid_updates) / total_updates if total_updates > 0 else 0
        df.loc[bar_time, 'inter_quote_mean_ms'] = inter_times.mean() if len(inter_times) > 0 else np.nan
        df.loc[bar_time, 'inter_quote_std_ms'] = inter_times.std() if len(inter_times) > 1 else np.nan
    
    # Spread z-score (vs recent history)
    spread_series = pd.Series(spread_history, index=df.index)
    df['spread_zscore'] = compute_zscore(spread_series, window=spread_lookback)
    
    # Spread percentile
    df['spread_percentile'] = compute_percentile_rank(spread_series, window=spread_lookback)
    
    return df


def build_intrabar_features(
    df_15m: pd.DataFrame,
    df_seconds: pd.DataFrame
) -> pd.DataFrame:
    """
    Build features from second-level data (intrabar analysis).
    
    Args:
        df_15m: 15-minute bars
        df_seconds: Second-level OHLCV with DatetimeIndex
    
    Returns:
        df_15m with intrabar features added
    """
    df = df_15m.copy()
    
    # Ensure index
    if not isinstance(df_seconds.index, pd.DatetimeIndex):
        if 'timestamp' in df_seconds.columns:
            df_seconds = df_seconds.set_index('timestamp')
    
    # Initialize columns
    intrabar_cols = [
        'intrabar_rv', 'intrabar_bv', 'intrabar_jump',
        'intrabar_trend', 'intrabar_reversals'
    ]
    for col in intrabar_cols:
        df[col] = np.nan
    
    for bar_time in df.index:
        bar_start = bar_time - pd.Timedelta(minutes=15)
        bar_end = bar_time
        
        mask = (df_seconds.index > bar_start) & (df_seconds.index <= bar_end)
        bar_data = df_seconds.loc[mask]
        
        if len(bar_data) < 10:
            continue
        
        # Compute returns
        returns = bar_data['close'].pct_change().dropna().values
        
        if len(returns) < 5:
            continue
        
        # Features
        rv = realized_variance(returns)
        bv = bipower_variation(returns)
        jump = max(rv - bv, 0)
        trend = linear_trend_slope(bar_data['close'].values)
        reversals = count_sign_changes(returns)
        
        df.loc[bar_time, 'intrabar_rv'] = rv
        df.loc[bar_time, 'intrabar_bv'] = bv
        df.loc[bar_time, 'intrabar_jump'] = jump
        df.loc[bar_time, 'intrabar_trend'] = trend
        df.loc[bar_time, 'intrabar_reversals'] = reversals
    
    return df


def build_all_features(
    df_15m: pd.DataFrame,
    df_quotes: Optional[pd.DataFrame] = None,
    df_seconds: Optional[pd.DataFrame] = None,
    config=None,
    show_progress: bool = False
) -> pd.DataFrame:
    """
    Build all features from available data sources.
    
    Args:
        df_15m: 15-minute OHLCV (required)
        df_quotes: Top-of-book quotes (optional)
        df_seconds: Second aggregates (optional)
        config: Model5Config
    
    Returns:
        DataFrame with all features
    """
    if config is None:
        config = Model5Config()
    
    # Start with 15M features
    df = build_15m_features(df_15m, config, show_progress=show_progress)
    
    # Add quote features if available
    if df_quotes is not None and len(df_quotes) > 0:
        df = build_quote_features(df, df_quotes, config.spread_lookback)
    
    # Add intrabar features if available
    if df_seconds is not None and len(df_seconds) > 0:
        df = build_intrabar_features(df, df_seconds)
    
    return df


def get_feature_columns(include_quotes: bool = True, include_intrabar: bool = True) -> List[str]:
    """
    Get list of feature columns for modeling.
    
    Args:
        include_quotes: Include quote-derived features
        include_intrabar: Include second-aggregate features
    
    Returns:
        List of feature column names
    """
    features = [
        # Core price statistics
        'returns_1',
        'returns_2',
        'zscore_20',
        'percentile_20',
        
        # Volatility
        'atr_14',
        'parkinson_vol_20',
        'returns_std_20',
        
        # Mean reversion indicators
        'variance_ratio_2',
        'variance_ratio_4',
        'autocorr_1',
        
        # Candle
        'body_ratio',
        'close_location',
        
        # Distribution
        'returns_skew_20',
        'returns_kurtosis_20',
        
        # Time
        'hour_sin',
        'hour_cos',
        'is_overlap',
    ]
    
    if include_quotes:
        features.extend([
            'spread_mean_bps',
            'spread_zscore',
            'quote_imbalance',
            'quote_intensity',
        ])
    
    if include_intrabar:
        features.extend([
            'intrabar_rv',
            'intrabar_jump',
            'intrabar_trend',
            'intrabar_reversals',
        ])
    
    return features

