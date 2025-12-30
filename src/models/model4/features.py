"""
Model 4 Feature Engineering
Only features legitimately derivable from OHLCV + quotes
"""
import numpy as np
import pandas as pd
from typing import Optional

# Try to import pandas_ta, fall back to manual calculation if not available
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Compute Average True Range manually."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
    """Compute ADX manually."""
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0),
        index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0),
        index=high.index
    )

    # Smoothed indicators
    atr = tr.ewm(span=length, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=length, adjust=False).mean() / (atr + 1e-8)
    minus_di = 100 * minus_dm.ewm(span=length, adjust=False).mean() / (atr + 1e-8)

    # ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    adx = dx.ewm(span=length, adjust=False).mean()

    return pd.DataFrame({
        f'ADX_{length}': adx,
        f'DMP_{length}': plus_di,
        f'DMN_{length}': minus_di
    }, index=high.index)


def build_model4_features(
    df_1t: pd.DataFrame,
    df_quotes: Optional[pd.DataFrame] = None,
    timeframe: str = "5T"
) -> pd.DataFrame:
    """
    Build features for Model 4 using only available data.

    Parameters:
    -----------
    df_1t : pd.DataFrame
        1-minute OHLCV data with DatetimeIndex
    df_quotes : pd.DataFrame, optional
        Quote data with bid_price, ask_price, participant_timestamp
    timeframe : str
        Target timeframe for resampling

    Returns:
    --------
    pd.DataFrame with features
    """

    # Resample to target timeframe
    df = df_1t.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # ========== VOLATILITY FEATURES ==========

    # ATR at multiple lookbacks
    if HAS_PANDAS_TA:
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_5'] = ta.atr(df['high'], df['low'], df['close'], length=5)
    else:
        df['atr_14'] = compute_atr(df['high'], df['low'], df['close'], length=14)
        df['atr_5'] = compute_atr(df['high'], df['low'], df['close'], length=5)

    df['atr_ratio'] = df['atr_5'] / df['atr_14'].replace(0, np.nan)

    # Range compression (NR4/NR7 detection)
    df['range'] = df['high'] - df['low']
    df['range_ma_20'] = df['range'].rolling(20).mean()
    df['range_compression'] = df['range'] / df['range_ma_20'].replace(0, np.nan)

    # Realized volatility from higher-frequency data
    # Compute on 1T data before resampling for accuracy
    rv_1t = df_1t['close'].pct_change().rolling(60).std() * np.sqrt(252 * 1440)
    df['realized_vol'] = rv_1t.resample(timeframe).last()
    df['realized_vol_ma'] = df['realized_vol'].rolling(20).mean()
    df['realized_vol_zscore'] = (
        (df['realized_vol'] - df['realized_vol_ma']) /
        df['realized_vol'].rolling(50).std().replace(0, np.nan)
    )

    # ========== PRICE POSITION FEATURES ==========

    # Session high/low (rolling ~6.5 hours = 78 bars on 5T)
    session_bars = 78 if timeframe == "5T" else 39 if timeframe == "10T" else 26
    df['session_high'] = df['high'].rolling(session_bars).max()
    df['session_low'] = df['low'].rolling(session_bars).min()
    df['session_range'] = df['session_high'] - df['session_low']

    df['price_in_session_range'] = (
        (df['close'] - df['session_low']) /
        df['session_range'].replace(0, np.nan)
    )

    df['dist_from_session_high_atr'] = (
        (df['session_high'] - df['close']) / df['atr_14'].replace(0, np.nan)
    )
    df['dist_from_session_low_atr'] = (
        (df['close'] - df['session_low']) / df['atr_14'].replace(0, np.nan)
    )

    # ========== MOMENTUM FEATURES ==========

    df['returns_1bar'] = df['close'].pct_change(1)
    df['returns_5bar'] = df['close'].pct_change(5)
    df['returns_12bar'] = df['close'].pct_change(12)

    # Z-scored momentum
    df['returns_5bar_zscore'] = (
        (df['returns_5bar'] - df['returns_5bar'].rolling(50).mean()) /
        df['returns_5bar'].rolling(50).std().replace(0, np.nan)
    )
    df['returns_12bar_zscore'] = (
        (df['returns_12bar'] - df['returns_12bar'].rolling(50).mean()) /
        df['returns_12bar'].rolling(50).std().replace(0, np.nan)
    )

    # ========== SPREAD FEATURES (from quotes) ==========

    if df_quotes is not None and len(df_quotes) > 0:
        quotes_resampled = df_quotes.resample(timeframe).agg({
            'ask_price': 'mean',
            'bid_price': 'mean',
        })
        quotes_resampled['mid'] = (quotes_resampled['ask_price'] + quotes_resampled['bid_price']) / 2
        quotes_resampled['spread'] = quotes_resampled['ask_price'] - quotes_resampled['bid_price']
        quotes_resampled['spread_pct'] = quotes_resampled['spread'] / quotes_resampled['mid'].replace(0, np.nan)
        quotes_resampled['spread_zscore'] = (
            (quotes_resampled['spread_pct'] - quotes_resampled['spread_pct'].rolling(60).mean()) /
            quotes_resampled['spread_pct'].rolling(60).std().replace(0, np.nan)
        )

        # Quote arrival rate
        quote_counts = df_quotes.resample(timeframe).size()
        quotes_resampled['quote_rate'] = quote_counts
        quotes_resampled['quote_rate_zscore'] = (
            (quotes_resampled['quote_rate'] - quotes_resampled['quote_rate'].rolling(60).mean()) /
            quotes_resampled['quote_rate'].rolling(60).std().replace(0, np.nan)
        )

        df = df.join(quotes_resampled[['spread_pct', 'spread_zscore', 'quote_rate_zscore']], how='left')
    else:
        # Placeholder if no quote data
        df['spread_pct'] = 0.0001  # Assume 1 bp spread
        df['spread_zscore'] = 0.0
        df['quote_rate_zscore'] = 0.0

    # ========== TREND FEATURES (for filtering, not prediction) ==========

    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['trend'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)

    # ADX for trend strength
    if HAS_PANDAS_TA:
        adx_df = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_df is not None:
            df['adx'] = adx_df['ADX_14']
            df['plus_di'] = adx_df['DMP_14']
            df['minus_di'] = adx_df['DMN_14']
        else:
            adx_df = compute_adx(df['high'], df['low'], df['close'], length=14)
            df['adx'] = adx_df['ADX_14']
            df['plus_di'] = adx_df['DMP_14']
            df['minus_di'] = adx_df['DMN_14']
    else:
        adx_df = compute_adx(df['high'], df['low'], df['close'], length=14)
        df['adx'] = adx_df['ADX_14']
        df['plus_di'] = adx_df['DMP_14']
        df['minus_di'] = adx_df['DMN_14']

    # ========== TIME FEATURES ==========

    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek

    # Cyclical encoding for hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Session flags
    df['is_london'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
    df['is_ny'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
    df['is_overlap_session'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
    df['is_asia'] = ((df['hour'] >= 0) & (df['hour'] < 7)).astype(int)

    # ========== CLEANUP ==========

    # Forward fill then drop remaining NaNs
    df = df.ffill().dropna()

    return df


def get_model4_feature_columns() -> list:
    """Return the feature columns used by Model 4."""
    return [
        'atr_ratio',
        'range_compression',
        'realized_vol_zscore',
        'price_in_session_range',
        'dist_from_session_high_atr',
        'dist_from_session_low_atr',
        'returns_5bar_zscore',
        'returns_12bar_zscore',
        'spread_pct',
        'spread_zscore',
        'quote_rate_zscore',
        'adx',
        'hour_sin',
        'hour_cos',
        'is_overlap_session',
    ]
