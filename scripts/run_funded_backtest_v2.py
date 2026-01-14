#!/usr/bin/env python3
"""
Funded Account Backtest v2 - All 7 Models

Runs a comprehensive backtest using the new backtest_engine_v2.
Tests against prop challenge criteria:
- Win Rate > 60%
- Max Drawdown < 4%
- Profit Target > 6%
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_engine_v2 import run_backtest, print_detailed_results, BacktestResult


# =============================================================================
# FEATURE BUILDERS (copied from training scripts)
# =============================================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def build_cmf_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build CMF and MACD features for Model 3."""
    df = df.copy()
    
    # CMF calculation
    high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
    hl_range = high - low
    hl_range = hl_range.replace(0, np.nan)
    mf_multiplier = ((close - low) - (high - close)) / hl_range
    mf_volume = mf_multiplier * volume
    
    for period in [10, 20, 40]:
        df[f'cmf_{period}'] = mf_volume.rolling(period).sum() / volume.rolling(period).sum()
    
    df['cmf_trend'] = df['cmf_20'] - df['cmf_20'].shift(5)
    df['cmf_bullish'] = (df['cmf_20'] > 0.05).astype(int)
    df['cmf_bearish'] = (df['cmf_20'] < -0.05).astype(int)
    df['cmf_strong_bullish'] = (df['cmf_20'] > 0.10).astype(int)
    df['cmf_strong_bearish'] = (df['cmf_20'] < -0.10).astype(int)
    df['cmf_support'] = df['cmf_20'].rolling(50).min()
    df['cmf_resistance'] = df['cmf_20'].rolling(50).max()
    df['cmf_position'] = (df['cmf_20'] - df['cmf_support']) / (df['cmf_resistance'] - df['cmf_support'] + 1e-10)
    
    # MACD
    ema_fast = ema(close, 12)
    ema_slow = ema(close, 26)
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = ema(df['macd'], 9)
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['macd_normalized'] = df['macd'] / close * 100
    df['macd_histogram_normalized'] = df['macd_histogram'] / close * 100
    df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
    df['macd_momentum'] = df['macd_histogram'] - df['macd_histogram'].shift(3)
    df['macd_positive'] = (df['macd'] > 0).astype(int)
    df['macd_histogram_positive'] = (df['macd_histogram'] > 0).astype(int)
    df['cmf_macd_bullish'] = ((df['cmf_bullish'] == 1) & (df['macd_histogram_positive'] == 1)).astype(int)
    df['cmf_macd_bearish'] = ((df['cmf_bearish'] == 1) & (df['macd_histogram_positive'] == 0)).astype(int)
    
    # Volume
    df['volume_ma'] = volume.rolling(20).mean()
    df['volume_ratio'] = volume / (df['volume_ma'] + 1e-10)
    df['volume_above_avg'] = (df['volume_ratio'] > 1.0).astype(int)
    df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)
    
    # Price trend
    df['return_5'] = close.pct_change(5)
    df['return_15'] = close.pct_change(15)
    df['return_30'] = close.pct_change(30)
    df['volatility_15'] = close.pct_change().rolling(15).std()
    
    return df


def build_mean_reversion_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build mean reversion features for Model 5."""
    df = df.copy()
    close = df['close']
    
    # Z-scores
    for window in [20, 50, 100]:
        rolling_mean = close.rolling(window).mean()
        rolling_std = close.rolling(window).std()
        df[f'zscore_{window}'] = (close - rolling_mean) / (rolling_std + 1e-10)
    
    df['zscore_extreme_low'] = (df['zscore_50'] < -2.0).astype(int)
    df['zscore_extreme_high'] = (df['zscore_50'] > 2.0).astype(int)
    df['zscore_very_extreme_low'] = (df['zscore_50'] < -2.5).astype(int)
    df['zscore_very_extreme_high'] = (df['zscore_50'] > 2.5).astype(int)
    
    # OU parameters (simplified)
    dx = close.diff()
    x = close.shift(1)
    xy = (x * dx).rolling(100).sum()
    x2 = (x ** 2).rolling(100).sum()
    df['ou_theta'] = (-xy / (x2 + 1e-10)).clip(0, 10)
    df['ou_halflife'] = (np.log(2) / (df['ou_theta'] + 1e-10)).clip(1, 200)
    df['ou_deviation_pct'] = (close - close.rolling(100).mean()) / close.rolling(100).mean() * 100
    
    # ADX
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - close.shift(1))
    tr3 = abs(df['low'] - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(span=14).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=14).mean() / (atr + 1e-10)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx_14'] = dx.ewm(span=14, adjust=False).mean()
    df['is_ranging'] = (df['adx_14'] < 20).astype(int)
    df['is_trending'] = (df['adx_14'] > 25).astype(int)
    
    # Bollinger
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df['bb_position'] = (close - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)
    df['bb_outside_upper'] = (close > bb_mid + 2*bb_std).astype(int)
    df['bb_outside_lower'] = (close < bb_mid - 2*bb_std).astype(int)
    
    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
    
    df['mr_long_signal'] = ((df['zscore_extreme_low'] == 1) & (df['is_ranging'] == 1)).astype(int)
    df['mr_short_signal'] = ((df['zscore_extreme_high'] == 1) & (df['is_ranging'] == 1)).astype(int)
    df['price_velocity'] = close.diff(5).abs() / close * 100
    df['slow_market'] = (df['price_velocity'] < df['price_velocity'].rolling(50).median()).astype(int)
    df['volatility_15'] = close.pct_change().rolling(15).std() * 100
    df['volatility_ratio'] = df['volatility_15'] / close.pct_change().rolling(100).std()
    df['volume_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
    
    return df


def build_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build microstructure features for Model 6."""
    df = df.copy()
    close, volume = df['close'], df['volume']
    
    # Kyle's Lambda
    price_change = close.diff()
    sign = np.sign(close - df['open'])
    signed_volume = sign * volume
    for window in [20, 50, 100]:
        cov = price_change.rolling(window).cov(signed_volume)
        var = signed_volume.rolling(window).var()
        df[f'kyle_lambda_{window}'] = cov / (var + 1e-10)
    
    df['kyle_lambda_zscore'] = (df['kyle_lambda_50'] - df['kyle_lambda_50'].rolling(200).mean()) / (df['kyle_lambda_50'].rolling(200).std() + 1e-10)
    df['low_liquidity'] = (df['kyle_lambda_zscore'] > 1.0).astype(int)
    df['high_liquidity'] = (df['kyle_lambda_zscore'] < -1.0).astype(int)
    
    # OFI
    hl_range = df['high'] - df['low']
    position = (close - df['low']) / (hl_range + 1e-10)
    pressure = (position - 0.5) * 2
    flow = pressure * volume
    for window in [10, 20, 50]:
        df[f'ofi_{window}'] = flow.rolling(window).sum()
    
    df['ofi_zscore'] = (df['ofi_20'] - df['ofi_20'].rolling(200).mean()) / (df['ofi_20'].rolling(200).std() + 1e-10)
    df['ofi_strong_buy'] = (df['ofi_zscore'] > 2.0).astype(int)
    df['ofi_strong_sell'] = (df['ofi_zscore'] < -2.0).astype(int)
    df['ofi_moderate_buy'] = (df['ofi_zscore'] > 1.0).astype(int)
    df['ofi_moderate_sell'] = (df['ofi_zscore'] < -1.0).astype(int)
    
    # VPIN
    buy_vol = volume.where(close > df['open'], 0)
    sell_vol = volume.where(close < df['open'], 0)
    for bucket in [50, 100]:
        total = volume.rolling(bucket).sum()
        buy = buy_vol.rolling(bucket).sum()
        sell = sell_vol.rolling(bucket).sum()
        df[f'vpin_{bucket}'] = (buy - sell).abs() / (total + 1e-10)
    
    df['vpin_high'] = (df['vpin_50'] > 0.3).astype(int)
    df['vpin_very_high'] = (df['vpin_50'] > 0.5).astype(int)
    df['spread_proxy'] = (df['high'] - df['low']) / close
    df['spread_zscore'] = (df['spread_proxy'] - df['spread_proxy'].rolling(100).mean()) / (df['spread_proxy'].rolling(100).std() + 1e-10)
    df['wide_spread'] = (df['spread_zscore'] > 1.5).astype(int)
    df['rv_5'] = close.pct_change().rolling(5).std() * np.sqrt(5) * 100
    df['rv_15'] = close.pct_change().rolling(15).std() * np.sqrt(15) * 100
    df['volume_ma'] = volume.rolling(50).mean()
    df['volume_ratio'] = volume / (df['volume_ma'] + 1e-10)
    df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)
    df['volume_dry'] = (df['volume_ratio'] < 0.5).astype(int)
    df['return_5'] = close.pct_change(5)
    df['return_15'] = close.pct_change(15)
    df['ms_long_signal'] = ((df['ofi_moderate_buy'] == 1) & (df['high_liquidity'] == 1)).astype(int)
    df['ms_short_signal'] = ((df['ofi_moderate_sell'] == 1) & (df['high_liquidity'] == 1)).astype(int)
    df['adverse_selection'] = ((df['vpin_high'] == 1) & (df['low_liquidity'] == 1)).astype(int)
    
    return df


def build_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build raw price features for Model 7."""
    df = df.copy()
    close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
    
    # Returns
    for period in [1, 2, 3, 5, 10, 15, 30]:
        df[f'log_return_{period}'] = np.log(close / close.shift(period))
    df['cum_return_5'] = df['log_return_1'].rolling(5).sum()
    df['cum_return_15'] = df['log_return_1'].rolling(15).sum()
    
    # Candle anatomy
    range_size = high - low
    body = abs(close - df['open'])
    df['body_ratio'] = body / (range_size + 1e-10)
    df['body_direction'] = np.sign(close - df['open'])
    upper_wick = high - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - low
    df['upper_wick_ratio'] = upper_wick / (range_size + 1e-10)
    df['lower_wick_ratio'] = lower_wick / (range_size + 1e-10)
    df['wick_asymmetry'] = df['upper_wick_ratio'] - df['lower_wick_ratio']
    df['range_pct'] = range_size / close * 100
    df['close_position'] = (close - low) / (range_size + 1e-10)
    
    # Price position
    for window in [5, 15]:
        rh = high.rolling(window).max()
        rl = low.rolling(window).min()
        df[f'price_position_{window}'] = (close - rl) / (rh - rl + 1e-10)
    
    # Gap
    df['gap'] = (df['open'] - close.shift(1)) / close.shift(1) * 100
    df['gap_filled'] = (((df['gap'] > 0) & (low <= close.shift(1))) | ((df['gap'] < 0) & (high >= close.shift(1)))).astype(int)
    
    # Higher/lower
    df['higher_high'] = (high > high.shift(1)).astype(int)
    df['lower_low'] = (low < low.shift(1)).astype(int)
    df['higher_close'] = (close > close.shift(1)).astype(int)
    df['consecutive_up'] = df['higher_close'].rolling(3).sum()
    df['consecutive_down'] = (1 - df['higher_close']).rolling(3).sum()
    
    # Volatility
    df['raw_volatility_5'] = df['log_return_1'].rolling(5).std()
    df['raw_volatility_15'] = df['log_return_1'].rolling(15).std()
    df['raw_volatility_30'] = df['log_return_1'].rolling(30).std()
    df['vol_change'] = df['raw_volatility_15'] / (df['raw_volatility_30'] + 1e-10)
    
    # Volume
    df['volume_change'] = volume / (volume.shift(1) + 1e-10)
    for window in [5, 15, 30]:
        df[f'volume_vs_{window}'] = volume / (volume.rolling(window).mean() + 1e-10)
    df['vol_price_sign'] = np.sign(df['log_return_1']) * np.sign(df['volume_change'] - 1)
    df['vol_price_corr_5'] = df['log_return_1'].rolling(5).corr(volume.pct_change())
    
    # Patterns
    prev_body = abs(close.shift(1) - df['open'].shift(1))
    engulfing = (body > prev_body * 1.5) & (df['body_direction'] != df['body_direction'].shift(1))
    df['engulfing'] = engulfing.astype(int)
    df['bullish_engulfing'] = (engulfing & (df['body_direction'] == 1)).astype(int)
    df['bearish_engulfing'] = (engulfing & (df['body_direction'] == -1)).astype(int)
    df['is_doji'] = (df['body_ratio'] < 0.1).astype(int)
    df['hammer_like'] = ((df['lower_wick_ratio'] > 0.6) & (df['upper_wick_ratio'] < 0.2)).astype(int)
    df['shooting_star_like'] = ((df['upper_wick_ratio'] > 0.6) & (df['lower_wick_ratio'] < 0.2)).astype(int)
    
    # Momentum
    df['momentum_5'] = df['log_return_1'].rolling(5).sum()
    df['momentum_15'] = df['log_return_1'].rolling(15).sum()
    df['momentum_delta'] = df['momentum_5'] - df['momentum_5'].shift(5)
    
    # Time
    if df.index.dtype == 'datetime64[ns]':
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    
    return df


def build_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build momentum features for Model 8."""
    df = df.copy()
    close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
    
    # Momentum z-scores
    for period in [10, 20, 30, 60]:
        price_change = close - close.shift(period)
        rolling_std = close.pct_change().rolling(period).std() * close
        df[f'momentum_z_{period}'] = price_change / (rolling_std * np.sqrt(period) + 1e-10)
    
    df['momentum_significant_10'] = (df['momentum_z_10'] > 1.96).astype(int)
    df['momentum_significant_30'] = (df['momentum_z_30'] > 1.96).astype(int)
    df['momentum_strong_30'] = (df['momentum_z_30'] > 2.58).astype(int)
    
    # VWAP
    typical = (high + low + close) / 3
    df['vwap'] = (typical * volume).rolling(30).sum() / volume.rolling(30).sum()
    df['vwap_deviation'] = (close - df['vwap']) / df['vwap'] * 100
    df['above_vwap'] = (close > df['vwap']).astype(int)
    df['far_above_vwap'] = (df['vwap_deviation'] > 0.1).astype(int)
    
    # Volume
    df['volume_ma_20'] = volume.rolling(20).mean()
    df['volume_ratio'] = volume / (df['volume_ma_20'] + 1e-10)
    df['volume_above_avg'] = (df['volume_ratio'] > 1.0).astype(int)
    df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)
    df['obv'] = (np.sign(close.diff()) * volume).cumsum()
    df['obv_trend'] = (df['obv'] > df['obv'].rolling(20).mean()).astype(int)
    
    # Trend
    df['higher_high'] = (high > high.shift(1)).astype(int)
    df['higher_low'] = (low > low.shift(1)).astype(int)
    df['uptrend_bars_5'] = (df['higher_high'] + df['higher_low']).rolling(5).sum()
    df['sma_10'] = close.rolling(10).mean()
    df['sma_30'] = close.rolling(30).mean()
    df['above_sma_10'] = (close > df['sma_10']).astype(int)
    df['above_sma_30'] = (close > df['sma_30']).astype(int)
    df['sma_10_above_30'] = (df['sma_10'] > df['sma_30']).astype(int)
    
    # Acceleration
    df['momentum_accel'] = df['momentum_z_30'] - df['momentum_z_30'].shift(5)
    df['accelerating'] = (df['momentum_accel'] > 0).astype(int)
    
    # Risk
    df['volatility_30'] = close.pct_change().rolling(30).std() * 100
    df['volatility_percentile'] = df['volatility_30'].rolling(200).rank(pct=True)
    df['normal_volatility'] = ((df['volatility_percentile'] > 0.2) & (df['volatility_percentile'] < 0.8)).astype(int)
    
    # Signals
    df['entry_signal_primary'] = ((df['momentum_significant_30'] == 1) & (df['above_vwap'] == 1) & (df['volume_above_avg'] == 1)).astype(int)
    df['entry_signal_strong'] = ((df['momentum_strong_30'] == 1) & (df['far_above_vwap'] == 1) & (df['volume_spike'] == 1) & (df['accelerating'] == 1)).astype(int)
    df['entry_signal_conservative'] = ((df['momentum_significant_30'] == 1) & (df['above_vwap'] == 1) & (df['volume_above_avg'] == 1) & (df['above_sma_30'] == 1) & (df['normal_volatility'] == 1)).astype(int)
    
    for period in [1, 5, 15, 30]:
        df[f'return_{period}'] = close.pct_change(period)
    
    return df


def build_rejection_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build rejection features for Model 9."""
    df = df.copy()
    close, high, low, volume = df['close'], df['high'], df['low'], df['volume']
    
    # Resistance
    for lookback in [20, 50, 100]:
        df[f'resistance_{lookback}'] = high.rolling(lookback).max()
        df[f'support_{lookback}'] = low.rolling(lookback).min()
    
    df['dist_to_resistance'] = (df['resistance_50'] - high) / close * 100
    df['at_resistance'] = (abs(df['dist_to_resistance']) < 0.1).astype(int)
    df['touched_resistance_5'] = (df['at_resistance'].rolling(5).sum() > 0).astype(int)
    df['new_high'] = (high > high.shift(1).rolling(20).max()).astype(int)
    df['failed_breakout'] = ((df['new_high'] == 1) & (close < df['open'])).astype(int)
    
    # Rejection score
    range_size = high - low
    upper_wick = high - df[['open', 'close']].max(axis=1)
    body_position = (close - low) / (range_size + 1e-10)
    upper_wick_ratio = upper_wick / (range_size + 1e-10)
    rejection = upper_wick_ratio * 0.5 + (1 - body_position) * 0.5
    bearish = (close < df['open']).astype(float) * 0.2
    df['rejection_score'] = (rejection + bearish).clip(0, 1)
    df['strong_rejection'] = (df['rejection_score'] > 0.7).astype(int)
    df['very_strong_rejection'] = (df['rejection_score'] > 0.8).astype(int)
    df['rejection_streak'] = df['strong_rejection'].rolling(3).sum()
    
    # Volume
    df['volume_ma'] = volume.rolling(20).mean()
    df['volume_ratio'] = volume / (df['volume_ma'] + 1e-10)
    df['volume_above_avg'] = (df['volume_ratio'] > 1.0).astype(int)
    df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)
    df['volume_climax'] = (df['volume_ratio'] > 2.0).astype(int)
    
    # Momentum
    df['momentum_10'] = close.pct_change(10)
    df['momentum_5'] = close.pct_change(5)
    df['momentum_slowing'] = ((df['momentum_10'] > 0) & (df['momentum_5'] < df['momentum_10'] / 2)).astype(int)
    df['mean_50'] = close.rolling(50).mean()
    df['std_50'] = close.rolling(50).std()
    df['zscore_50'] = (close - df['mean_50']) / (df['std_50'] + 1e-10)
    df['overbought_zscore'] = (df['zscore_50'] > 2.0).astype(int)
    df['extremely_overbought'] = (df['zscore_50'] > 2.5).astype(int)
    
    # Patterns
    body = abs(close - df['open'])
    prev_body = abs(close.shift(1) - df['open'].shift(1))
    df['shooting_star'] = ((upper_wick > 2 * body) & (close < df['open'])).astype(int)
    df['bearish_engulfing'] = ((body > prev_body * 1.5) & (close < df['open']) & (close.shift(1) > df['open'].shift(1))).astype(int)
    df['evening_star'] = ((close.shift(2) > df['open'].shift(2)) & (abs(close.shift(1) - df['open'].shift(1)) < body.shift(1) * 0.3) & (close < df['open'])).astype(int)
    
    # Reversal
    df['lower_high'] = (high < high.shift(1)).astype(int)
    df['lower_close'] = (close < close.shift(1)).astype(int)
    df['downtrend_bars_3'] = df['lower_close'].rolling(3).sum()
    typical = (high + low + close) / 3
    df['vwap'] = (typical * volume).rolling(30).sum() / volume.rolling(30).sum()
    df['below_vwap'] = (close < df['vwap']).astype(int)
    df['volatility_30'] = close.pct_change().rolling(30).std() * 100
    
    # Signals
    df['short_signal_primary'] = ((df['touched_resistance_5'] == 1) & (df['strong_rejection'] == 1) & (df['volume_above_avg'] == 1)).astype(int)
    df['short_signal_strong'] = ((df['touched_resistance_5'] == 1) & (df['very_strong_rejection'] == 1) & (df['volume_spike'] == 1) & (df['overbought_zscore'] == 1)).astype(int)
    df['short_signal_conservative'] = ((df['touched_resistance_5'] == 1) & (df['strong_rejection'] == 1) & (df['volume_spike'] == 1) & (df['below_vwap'] == 1) & (df['momentum_slowing'] == 1)).astype(int)
    
    for period in [1, 5, 15, 30]:
        df[f'return_{period}'] = close.pct_change(period)
    
    return df

# Paths
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
MODELS_DIR = PROJECT_ROOT / "models"

# Test period
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

# Model configurations with feature builders
FEATURE_BUILDERS = {
    "cmf_macd": build_cmf_macd_features,
    "mean_reversion": build_mean_reversion_features,
    "microstructure": build_microstructure_features,
    "raw": build_raw_features,
    "momentum": build_momentum_features,
    "rejection": build_rejection_features,
}

MODELS = {
    "Model 3 (CMF/MACD)": {
        "path": "model3_v3_cmf_macd.joblib",
        "feature_builder": "cmf_macd",
        "direction": "both"
    },
    "Model 5 (Mean Rev)": {
        "path": "model5_v3_mean_reversion.joblib",
        "feature_builder": "mean_reversion",
        "direction": "both"
    },
    "Model 6 (Microstructure)": {
        "path": "model6_v3_microstructure.joblib",
        "feature_builder": "microstructure",
        "direction": "both"
    },
    "Model 7 (Raw Price)": {
        "path": "model7_raw_price.joblib",
        "feature_builder": "raw",
        "direction": "both"
    },
    "Model 8 (Momentum LONG)": {
        "path": "model8_momentum_long.joblib",
        "feature_builder": "momentum",
        "direction": "long_only"
    },
    "Model 9 (Rejection SHORT)": {
        "path": "model9_rejection_short.joblib",
        "feature_builder": "rejection",
        "direction": "short_only"
    }
}


def generate_signals_from_model(
    model_artifact: dict,
    df: pd.DataFrame,
    direction: str = "both"
) -> pd.Series:
    """Generate signals from a model using its stored features."""
    model = model_artifact['model']
    features = model_artifact['features']
    threshold = model_artifact.get('threshold', 0.55)
    model_type = model_artifact.get('model_type', 'Unknown')
    scaler = model_artifact.get('scaler', None)
    
    # Get available features
    available = [f for f in features if f in df.columns]
    if len(available) < len(features) * 0.5:
        print(f"    Warning: Only {len(available)}/{len(features)} features available")
    
    # Prepare data
    X = df[available].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Apply scaler for neural network models
    if model_type == 'NeuralNetwork' and scaler is not None:
        X = scaler.transform(X)
    
    # Get predictions
    proba = model.predict_proba(X)[:, 1]
    
    # Generate signals based on direction
    signals = pd.Series(0, index=df.index, dtype=int)
    
    if direction == "long_only":
        # Model 8: Only long signals (HIGH proba = up prediction)
        signals[proba >= threshold] = 1
    elif direction == "short_only":
        # Model 9: Only short signals (HIGH proba = down prediction)
        signals[proba >= threshold] = -1
    else:
        # Bidirectional models
        signals[proba >= threshold] = 1  # High "up" prob = LONG
        signals[proba <= (1 - threshold)] = -1  # Low "up" prob = SHORT
    
    return signals


def load_and_backtest_model(
    model_name: str,
    config: dict,
    df: pd.DataFrame
) -> BacktestResult:
    """Load a model and run backtest."""
    model_path = MODELS_DIR / config["path"]
    
    if not model_path.exists():
        print(f"  Model not found: {model_path}")
        return None
    
    # Load model
    artifact = joblib.load(model_path)
    print(f"  Loaded {model_name}")
    print(f"    Model Type: {artifact.get('model_type', 'unknown')}")
    print(f"    Strategy: {artifact.get('strategy', 'unknown')}")
    print(f"    Threshold: {artifact.get('threshold', 0.55)}")
    print(f"    Val AUC: {artifact.get('val_auc', 'N/A')}")
    
    # Build features for this model
    feature_builder_name = config.get("feature_builder")
    if feature_builder_name and feature_builder_name in FEATURE_BUILDERS:
        print(f"    Building features...")
        df_with_features = FEATURE_BUILDERS[feature_builder_name](df.copy())
    else:
        df_with_features = df.copy()
    
    # Generate signals
    signals = generate_signals_from_model(artifact, df_with_features, config["direction"])
    
    signal_long = (signals == 1).sum()
    signal_short = (signals == -1).sum()
    print(f"    LONG signals: {signal_long:,}")
    print(f"    SHORT signals: {signal_short:,}")
    
    if signal_long + signal_short < 10:
        print(f"    WARNING: Too few signals to backtest")
        return None
    
    # Run backtest (use original df for OHLCV, signals aligned by index)
    # Use TIME-BASED EXIT for consistent results
    result = run_backtest(
        df=df,  # Original OHLCV data
        signals=signals,  # Signals aligned to df index
        model_name=model_name,
        initial_balance=50000.0,
        position_size_lots=0.15,  # 0.15 lots per trade
        stop_atr_mult=1.0,
        target_atr_mult=1.0,
        max_bars_in_trade=30,
        use_time_exit=True,  # Time-based exit only
        verbose=False
    )
    
    return result


def main():
    print("=" * 80)
    print("FUNDED ACCOUNT BACKTEST v2 - ALL MODELS")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now()}")
    print(f"Test Period: {TEST_START} to {TEST_END}")
    
    # Load test data
    print("\n[1] Loading data...")
    df = pd.read_parquet(FEATURES_PATH)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    # Filter to test period
    df_test = df[TEST_START:TEST_END].copy()
    print(f"  Test samples: {len(df_test):,}")
    
    # Normalize ATR column
    if 'ATR_14' in df_test.columns and 'atr_14' not in df_test.columns:
        df_test['atr_14'] = df_test['ATR_14']
    
    # Run backtests
    print("\n[2] Running backtests...")
    results = {}
    
    for model_name, config in MODELS.items():
        print(f"\n  Testing {model_name}...")
        result = load_and_backtest_model(model_name, config, df_test)
        if result is not None:
            results[model_name] = result
    
    # Summary
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 80)
    
    print("\n" + "-" * 100)
    print(f"{'Model':<25} {'Trades':>8} {'Win Rate':>10} {'PF':>8} {'Sharpe':>8} {'MaxDD':>8} {'Return':>10} {'Status':>10}")
    print("-" * 100)
    
    passed_models = []
    
    for model_name, result in results.items():
        status = "PASS" if (
            result.win_rate > 0.55 and 
            result.max_drawdown_pct < 5.0 and 
            result.total_return_pct > 0
        ) else "FAIL"
        
        if status == "PASS":
            passed_models.append(model_name)
        
        print(f"{model_name:<25} {len(result.trades):>8} {result.win_rate:>9.1%} "
              f"{result.profit_factor:>8.2f} {result.sharpe_ratio:>8.2f} "
              f"{result.max_drawdown_pct:>7.2f}% {result.total_return_pct:>+9.2f}% {status:>10}")
    
    print("-" * 100)
    
    # Prop Challenge Check
    print("\n" + "=" * 80)
    print("PROP CHALLENGE CRITERIA")
    print("=" * 80)
    print("\nMinimum Requirements:")
    print("  - Win Rate > 55%")
    print("  - Max Drawdown < 5%")
    print("  - Profitable (Return > 0%)")
    
    print(f"\nModels that PASSED: {len(passed_models)}/{len(results)}")
    for m in passed_models:
        print(f"  + {m}")
    
    failed_models = [m for m in results if m not in passed_models]
    if failed_models:
        print(f"\nModels that FAILED: {len(failed_models)}")
        for m in failed_models:
            print(f"  - {m}")
    
    # Print detailed results for top models
    if passed_models:
        print("\n" + "=" * 80)
        print("DETAILED RESULTS - PASSED MODELS")
        print("=" * 80)
        
        for model_name in passed_models[:3]:  # Top 3
            print_detailed_results(results[model_name])
    
    # Best model recommendation
    if results:
        best_model = max(results.items(), key=lambda x: x[1].sharpe_ratio if x[1].win_rate > 0.5 else -999)
        print("\n" + "=" * 80)
        print(f"RECOMMENDATION: {best_model[0]}")
        print("=" * 80)
        print(f"  Sharpe Ratio: {best_model[1].sharpe_ratio:.2f}")
        print(f"  Win Rate: {best_model[1].win_rate:.1%}")
        print(f"  Max Drawdown: {best_model[1].max_drawdown_pct:.2f}%")
        print(f"  Total Return: {best_model[1].total_return_pct:+.2f}%")


if __name__ == "__main__":
    main()

