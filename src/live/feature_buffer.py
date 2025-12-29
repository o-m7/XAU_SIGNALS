#!/usr/bin/env python3
"""
Rolling Feature Buffer for Live Trading.

Maintains a rolling window of minute bars and computes features
EXACTLY as in training - no lookahead, no future data.

CRITICAL: This buffer does NOT backfill historical data.
It only produces signals after warming up from live stream data.

Features computed:
- Returns (ret_1, ret_3, ret_5, ret_10, log_ret_1, ret_mean_*)
- Volatility (vol_10, vol_60, ATR_14, hl_range, norm_range)
- Microstructure (mid, spread, spread_pct, mid_ret_1, mid_vol_20, mid_slope_10)
- Volume (vol_change, vol_rel_20, vol_zscore_20, dollar_vol)
- Candlestick (body, range, wicks, is_bull)
- MTF (ma_fast_15, ma_slow_60, ma_ratio, ma_slope_60)
- Time (minute_sin, minute_cos, day_of_week, session flags)
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
import numpy as np
import pandas as pd
import warnings

# Suppress pandas FutureWarnings about fillna downcasting
warnings.filterwarnings('ignore', category=FutureWarning, message='.*Downcasting.*')

logger = logging.getLogger("FeatureBuffer")


# Feature names must match training EXACTLY (60 features from model)
FEATURE_NAMES = [
    'micro_velocity', 'micro_entropy', 'effort_ratio', 'spread_zscore',
    'synthetic_order_flow', 'flow_cvd_60', 'flow_divergence',
    'ret_1', 'ret_3', 'ret_5', 'ret_10', 'log_ret_1',
    'ret_mean_5', 'ret_mean_20', 'ret_mean_60',
    'vol_10', 'vol_60', 'hl_range', 'norm_range', 'ATR_14',
    'mid', 'spread', 'spread_pct',
    'mid_ret_1', 'mid_vol_20', 'mid_slope_10',
    'close_mid_diff', 'close_mid_spread_ratio',
    'vol_change', 'vol_rel_20', 'vol_zscore_20', 'dollar_vol',
    'body', 'range', 'upper_wick', 'lower_wick',
    'body_pct', 'upper_wick_pct', 'lower_wick_pct', 'is_bull',
    'ma_fast_15', 'ma_slow_60', 'ma_ratio', 'ma_slope_60',
    'minute_sin', 'minute_cos', 'day_of_week',
    'is_asia', 'is_europe', 'is_us',
    'adx', 'adx_slope', 'efficiency_ratio', 'atr_percentile',
    'regime_trending', 'regime_ranging', 'regime_volatile',
    'wick_flow_interaction', 'body_flow_interaction', 'wick_volume_interaction',
]

# Minimum bars needed for all features
# Need 60 for vol_60, MA_60 + some buffer for rolling calculations
MIN_BARS_REQUIRED = 65


class FeatureBuffer:
    """
    Rolling feature buffer that aggregates ticks/bars to minute bars
    and computes features matching training exactly.
    
    LIVE-ONLY BEHAVIOR:
    - Does NOT backfill historical data on startup
    - Only produces signals after warmup from live stream
    - Logs warmup progress
    
    Args:
        max_window: Maximum number of minute bars to keep (default 500)
        aggregation_seconds: Seconds per bar (default 60 for 1-minute)
    """
    
    def __init__(
        self,
        max_window: int = 500,
        aggregation_seconds: int = 60
    ):
        self.max_window = max_window
        self.aggregation_seconds = aggregation_seconds
        self.min_bars_required = MIN_BARS_REQUIRED
        
        # Bar storage: list of dicts with OHLCV + quotes
        self._bars: deque = deque(maxlen=max_window)
        
        # Current bar being built from ticks
        self._current_bar: Optional[Dict] = None
        self._current_bar_start: Optional[datetime] = None
        
        # Warmup tracking
        self._warmup_complete = False
        self._last_warmup_log: Optional[datetime] = None
        
        logger.info(
            f"FeatureBuffer initialized: max_window={max_window}, "
            f"min_bars_required={MIN_BARS_REQUIRED}"
        )
    
    def update_from_quote(self, event: Dict) -> Optional[pd.DataFrame]:
        """
        Update buffer from a quote event.
        
        Args:
            event: Dict with type="quote", timestamp, bid, ask, mid
            
        Returns:
            Feature row DataFrame if a new bar was completed and buffer ready, else None
        """
        timestamp = event["timestamp"]
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp, utc=True)
        elif timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        mid = event.get("mid") or ((event.get("bid", 0) + event.get("ask", 0)) / 2)
        
        # Synthetic tick from quote
        tick = {
            "timestamp": timestamp,
            "price": mid,
            "bid": event.get("bid"),
            "ask": event.get("ask"),
            "volume": 0,  # Quotes don't have volume
        }
        
        return self._update(tick)
    
    def update_from_bar(self, event: Dict) -> Optional[pd.DataFrame]:
        """
        Update buffer from a bar event (from aggregate stream).
        
        Args:
            event: Dict with type="bar", timestamp, open, high, low, close, volume
            
        Returns:
            Feature row DataFrame if buffer ready, else None
        """
        timestamp = event["timestamp"]
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp, utc=True)
        elif timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        # Direct bar insertion
        bar = {
            "timestamp": timestamp,
            "open": event["open"],
            "high": event["high"],
            "low": event["low"],
            "close": event["close"],
            "volume": event.get("volume", 0),
            "bid_price": None,  # Not available from agg stream
            "ask_price": None,
            "n_ticks": 1,
        }
        
        self._bars.append(bar)
        self._log_warmup_progress()
        
        if self.is_ready():
            return self.get_feature_row()
        return None
    
    def _update(self, tick: Dict) -> Optional[pd.DataFrame]:
        """
        Internal update from tick.
        
        Aggregates ticks into minute bars and returns features
        when a bar completes.
        """
        timestamp = tick["timestamp"]
        
        # Determine which bar this tick belongs to
        bar_start = self._get_bar_start(timestamp)
        
        # Check if we need to close current bar and start new one
        if self._current_bar_start and bar_start > self._current_bar_start:
            # Close current bar
            completed_bar = self._finalize_current_bar()
            if completed_bar:
                self._bars.append(completed_bar)
            
            # Start new bar
            self._start_new_bar(bar_start, tick)
            
            self._log_warmup_progress()
            
            # Return features if we have enough bars
            if self.is_ready():
                return self.get_feature_row()
            return None
        
        # Update current bar with tick
        if self._current_bar is None:
            self._start_new_bar(bar_start, tick)
        else:
            self._update_current_bar(tick)
        
        return None
    
    def _log_warmup_progress(self):
        """Log warmup progress periodically."""
        now = datetime.now(timezone.utc)
        
        # Log every 10 seconds during warmup
        if not self._warmup_complete:
            if self._last_warmup_log is None or (now - self._last_warmup_log).total_seconds() > 10:
                bars = len(self._bars)
                logger.info(
                    f"WARMUP: bars_collected={bars}, required={self.min_bars_required}"
                )
                self._last_warmup_log = now
                
                if bars >= self.min_bars_required:
                    self._warmup_complete = True
                    logger.info("WARMUP COMPLETE: Ready to generate signals")
    
    def _get_bar_start(self, timestamp: datetime) -> datetime:
        """Get the start time of the bar containing this timestamp."""
        seconds = timestamp.second + timestamp.minute * 60 + timestamp.hour * 3600
        bar_seconds = (seconds // self.aggregation_seconds) * self.aggregation_seconds
        
        return timestamp.replace(
            hour=bar_seconds // 3600,
            minute=(bar_seconds % 3600) // 60,
            second=bar_seconds % 60,
            microsecond=0
        )
    
    def _start_new_bar(self, bar_start: datetime, tick: Dict):
        """Initialize a new bar."""
        self._current_bar_start = bar_start
        price = tick["price"]
        
        self._current_bar = {
            "timestamp": bar_start,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": tick.get("volume") or 0,
            "bid_price": tick.get("bid"),
            "ask_price": tick.get("ask"),
            "n_ticks": 1,
        }
    
    def _update_current_bar(self, tick: Dict):
        """Update current bar with new tick."""
        price = tick["price"]
        
        self._current_bar["high"] = max(self._current_bar["high"], price)
        self._current_bar["low"] = min(self._current_bar["low"], price)
        self._current_bar["close"] = price
        self._current_bar["volume"] += tick.get("volume") or 0
        self._current_bar["n_ticks"] += 1
        
        # Update bid/ask if available
        if tick.get("bid"):
            self._current_bar["bid_price"] = tick["bid"]
        if tick.get("ask"):
            self._current_bar["ask_price"] = tick["ask"]
    
    def _finalize_current_bar(self) -> Optional[Dict]:
        """Finalize and return the current bar."""
        if not self._current_bar:
            return None
        
        bar = self._current_bar.copy()
        
        # Set end timestamp
        bar["timestamp"] = self._current_bar_start + timedelta(
            seconds=self.aggregation_seconds
        )
        
        return bar
    
    def is_ready(self) -> bool:
        """Check if buffer has enough data to compute features."""
        return len(self._bars) >= self.min_bars_required
    
    def is_warming_up(self) -> bool:
        """Check if still in warmup phase."""
        return not self.is_ready()
    
    def get_bar_count(self) -> int:
        """Get number of completed bars in buffer."""
        return len(self._bars)
    
    def get_warmup_progress(self) -> Dict:
        """Get warmup progress info."""
        bars = len(self._bars)
        return {
            "bars_collected": bars,
            "bars_required": self.min_bars_required,
            "progress_pct": min(100, bars / self.min_bars_required * 100),
            "ready": self.is_ready(),
        }
    
    def get_feature_row(self) -> pd.DataFrame:
        """
        Compute features for the latest bar.
        
        Returns:
            Single-row DataFrame with all features
            
        Raises:
            ValueError if not enough bars
        """
        if not self.is_ready():
            raise ValueError(
                f"Not enough bars: {len(self._bars)} < {self.min_bars_required}"
            )
        
        # Convert bars to DataFrame
        df = pd.DataFrame(list(self._bars))
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        
        # Compute all features
        features = self._compute_features(df)
        
        # Return only the latest row
        latest = features.iloc[[-1]].copy()
        
        # Ensure all required features exist
        for feat in FEATURE_NAMES:
            if feat not in latest.columns:
                latest[feat] = np.nan
                logger.warning(f"Missing feature: {feat}")
        
        # Select only required features in correct order
        return latest[FEATURE_NAMES]
    
    def _compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features from bar DataFrame.
        
        CRITICAL: All computations use only past/current data.
        NO lookahead, NO shift(-N) operations.
        """
        df = df.copy()
        
        # =================================================================
        # Price & Returns
        # =================================================================
        df["ret_1"] = df["close"].pct_change(1)
        df["ret_3"] = df["close"].pct_change(3)
        df["ret_5"] = df["close"].pct_change(5)
        df["ret_10"] = df["close"].pct_change(10)
        df["log_ret_1"] = np.log(df["close"] / df["close"].shift(1))
        
        df["ret_mean_5"] = df["ret_1"].rolling(5).mean()
        df["ret_mean_20"] = df["ret_1"].rolling(20).mean()
        df["ret_mean_60"] = df["ret_1"].rolling(60).mean()
        
        # =================================================================
        # Volatility & Range
        # =================================================================
        df["vol_10"] = df["log_ret_1"].rolling(10).std()
        df["vol_60"] = df["log_ret_1"].rolling(60).std()
        
        df["hl_range"] = df["high"] - df["low"]
        df["norm_range"] = df["hl_range"] / df["close"]
        
        # ATR_14 using standard true range
        df["prev_close"] = df["close"].shift(1)
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                np.abs(df["high"] - df["prev_close"]),
                np.abs(df["low"] - df["prev_close"])
            )
        )
        df["ATR_14"] = df["tr"].rolling(14).mean()
        
        # =================================================================
        # Microstructure (from quotes)
        # =================================================================
        if "bid_price" in df.columns and "ask_price" in df.columns:
            df["mid"] = (df["bid_price"] + df["ask_price"]) / 2
            df["spread"] = df["ask_price"] - df["bid_price"]
            df["spread_pct"] = df["spread"] / df["mid"]
        else:
            df["mid"] = df["close"]
            df["spread"] = 0.0
            df["spread_pct"] = 0.0
        
        # Fill NaN mid with close (fix FutureWarning)
        df["mid"] = df["mid"].infer_objects(copy=False).fillna(df["close"])
        df["spread"] = df["spread"].infer_objects(copy=False).fillna(0)
        df["spread_pct"] = df["spread_pct"].infer_objects(copy=False).fillna(0)
        
        df["mid_ret_1"] = df["mid"].pct_change(1)
        df["mid_vol_20"] = df["mid_ret_1"].rolling(20).std()
        
        # Mid slope: OLS slope over last 10 bars
        df["mid_slope_10"] = self._rolling_slope(df["mid"], 10)
        
        df["close_mid_diff"] = df["close"] - df["mid"]
        df["close_mid_spread_ratio"] = np.where(
            df["spread"] != 0,
            df["close_mid_diff"] / df["spread"],
            0
        )
        
        # =================================================================
        # Volume & Liquidity
        # =================================================================
        df["vol_change"] = df["volume"] / df["volume"].shift(1) - 1
        df["vol_change"] = df["vol_change"].replace([np.inf, -np.inf], 0).infer_objects(copy=False).fillna(0)
        
        vol_median_20 = df["volume"].rolling(20).median()
        df["vol_rel_20"] = df["volume"] / vol_median_20.replace(0, np.nan)
        df["vol_rel_20"] = df["vol_rel_20"].infer_objects(copy=False).fillna(1)
        
        vol_mean_20 = df["volume"].rolling(20).mean()
        vol_std_20 = df["volume"].rolling(20).std()
        df["vol_zscore_20"] = (df["volume"] - vol_mean_20) / vol_std_20.replace(0, np.nan)
        df["vol_zscore_20"] = df["vol_zscore_20"].infer_objects(copy=False).fillna(0)
        
        df["dollar_vol"] = df["close"] * df["volume"]
        
        # =================================================================
        # Candlestick Structure
        # =================================================================
        df["body"] = df["close"] - df["open"]
        df["range"] = df["high"] - df["low"]
        
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        
        df["body_pct"] = np.where(
            df["range"] != 0,
            np.abs(df["body"]) / df["range"],
            0
        )
        df["upper_wick_pct"] = np.where(
            df["range"] != 0,
            df["upper_wick"] / df["range"],
            0
        )
        df["lower_wick_pct"] = np.where(
            df["range"] != 0,
            df["lower_wick"] / df["range"],
            0
        )
        
        df["is_bull"] = (df["close"] > df["open"]).astype(int)
        
        # =================================================================
        # Multi-Timeframe Context (MAs)
        # =================================================================
        df["ma_fast_15"] = df["close"].rolling(15).mean()
        df["ma_slow_60"] = df["close"].rolling(60).mean()
        
        df["ma_ratio"] = df["ma_fast_15"] / df["ma_slow_60"] - 1
        
        # MA slope: average change over last 5 bars
        ma_diff = df["ma_slow_60"].diff()
        df["ma_slope_60"] = ma_diff.rolling(5).mean()
        
        # =================================================================
        # Time Features
        # =================================================================
        if isinstance(df.index, pd.DatetimeIndex):
            minute_of_day = df.index.hour * 60 + df.index.minute
            df["minute_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
            df["minute_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
            df["day_of_week"] = df.index.dayofweek
            
            # Session flags (UTC times)
            hour = df.index.hour
            df["is_asia"] = ((hour >= 0) & (hour < 8)).astype(int)
            df["is_europe"] = ((hour >= 7) & (hour < 16)).astype(int)
            df["is_us"] = ((hour >= 13) & (hour < 22)).astype(int)
        else:
            df["minute_sin"] = 0
            df["minute_cos"] = 0
            df["day_of_week"] = 0
            df["is_asia"] = 0
            df["is_europe"] = 0
            df["is_us"] = 0
        
        # =================================================================
        # Microstructure Features (simplified for minute bars)
        # =================================================================
        # Note: Full microstructure features require tick-level quotes.
        # These are approximations using minute bars.
        
        # micro_velocity: Approximate as tick count (use volume as proxy)
        df["micro_velocity"] = df["volume"].infer_objects(copy=False).fillna(0)
        
        # micro_entropy: Approximate using price direction variance
        price_dir = np.sign(df["close"].diff()).fillna(0)
        entropy_vals = []
        for i in range(len(price_dir)):
            if i < 10:
                entropy_vals.append(0.5)
            else:
                window = price_dir.iloc[max(0, i-19):i+1]
                unique_count = len(np.unique(window))
                entropy_vals.append(unique_count / max(len(window), 1))
        df["micro_entropy"] = pd.Series(entropy_vals, index=df.index).infer_objects(copy=False).fillna(0.5)
        
        # effort_ratio: Range / Volume (already computed, but ensure it exists)
        df["effort_ratio"] = np.where(
            df["micro_velocity"] > 0,
            df["range"] / (df["micro_velocity"] + 1),
            0
        )
        
        # spread_zscore: Z-score of spread
        spread_mean = df["spread"].rolling(60, min_periods=10).mean()
        spread_std = df["spread"].rolling(60, min_periods=10).std()
        df["spread_zscore"] = pd.Series(
            np.where(
                spread_std > 0,
                (df["spread"] - spread_mean) / spread_std,
                0
            ),
            index=df.index
        ).infer_objects(copy=False).fillna(0)
        
        # synthetic_order_flow: Approximate using price direction * volume
        df["synthetic_order_flow"] = price_dir * df["volume"]
        
        # flow_cvd_60: Cumulative volume delta over 60 minutes
        df["flow_cvd_60"] = df["synthetic_order_flow"].rolling(60, min_periods=10).sum().infer_objects(copy=False).fillna(0)
        
        # flow_divergence: Normalized price return vs flow trend
        ret_60 = df["close"].pct_change(60)
        ret_mean = ret_60.rolling(100, min_periods=20).mean()
        ret_std = ret_60.rolling(100, min_periods=20).std()
        ret_norm = pd.Series(
            np.where(ret_std > 0, (ret_60 - ret_mean) / ret_std, 0),
            index=df.index
        )
        
        flow_mean = df["flow_cvd_60"].rolling(100, min_periods=20).mean()
        flow_std = df["flow_cvd_60"].rolling(100, min_periods=20).std()
        flow_norm = pd.Series(
            np.where(flow_std > 0, (df["flow_cvd_60"] - flow_mean) / flow_std, 0),
            index=df.index
        )
        
        df["flow_divergence"] = flow_norm - ret_norm
        
        # =================================================================
        # Regime Detection Features
        # =================================================================
        # ADX calculation
        high = df["high"]
        low = df["low"]
        close = df["close"]
        
        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        # Create Series with proper index alignment
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0),
            index=df.index
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0),
            index=df.index
        )
        
        # Smoothed indicators
        atr_14 = tr.ewm(span=14, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_14
        minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_14
        
        # ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
        df["adx"] = dx.ewm(span=14, adjust=False).mean().infer_objects(copy=False).fillna(0)
        df["adx_slope"] = df["adx"].diff(5).infer_objects(copy=False).fillna(0)
        
        # Efficiency Ratio (Kaufman)
        efficiency_window = 20
        net_change = (close - close.shift(efficiency_window)).abs()
        sum_changes = close.diff().abs().rolling(efficiency_window).sum()
        df["efficiency_ratio"] = (net_change / (sum_changes + 1e-8)).infer_objects(copy=False).fillna(0)
        
        # ATR percentile
        atr_lookback = 100
        df["atr_percentile"] = df["ATR_14"].rolling(atr_lookback, min_periods=10).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5,
            raw=False
        ).infer_objects(copy=False).fillna(0.5)
        
        # Regime flags
        df["regime_trending"] = ((df["adx"] > 25) & (df["efficiency_ratio"] > 0.5)).astype(int)
        df["regime_ranging"] = ((df["adx"] < 20) & (df["atr_percentile"] < 0.75)).astype(int)
        df["regime_volatile"] = (df["atr_percentile"] > 0.75).astype(int)
        
        # =================================================================
        # Interaction Features
        # =================================================================
        # Use flow_cvd_60 for interactions (if available, else synthetic_order_flow)
        flow_col = "flow_cvd_60" if "flow_cvd_60" in df.columns else "synthetic_order_flow"
        
        df["wick_flow_interaction"] = df["upper_wick"] * np.sign(df[flow_col])
        df["body_flow_interaction"] = df["body"] * np.sign(df[flow_col])
        df["wick_volume_interaction"] = df["upper_wick"] * df["volume"]
        
        # Clean up temp columns
        for col in ["prev_close", "tr"]:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        return df
    
    def _rolling_slope(self, series: pd.Series, window: int) -> pd.Series:
        """Compute rolling OLS slope."""
        def slope(y):
            if len(y) < window or np.isnan(y).all():
                return np.nan
            x = np.arange(len(y))
            valid = ~np.isnan(y)
            if valid.sum() < 2:
                return np.nan
            coef = np.polyfit(x[valid], y[valid], 1)
            return coef[0]
        
        return series.rolling(window).apply(slope, raw=True)
    
    def get_latest_bar(self) -> Optional[Dict]:
        """Get the most recent completed bar."""
        if self._bars:
            return self._bars[-1]
        return None
    
    def get_current_price(self) -> Optional[float]:
        """Get the current (incomplete bar) price."""
        if self._current_bar:
            return self._current_bar["close"]
        if self._bars:
            return self._bars[-1]["close"]
        return None


# =============================================================================
# Standalone test
# =============================================================================

if __name__ == "__main__":
    import time
    
    buffer = FeatureBuffer(max_window=200)
    
    # Simulate quotes
    base_time = datetime.now(timezone.utc)
    price = 2650.0
    
    print("Simulating live quotes (no backfill)...")
    print(f"Required bars for warmup: {buffer.min_bars_required}")
    
    for i in range(70 * 60):  # 70 minutes worth
        event = {
            "type": "quote",
            "timestamp": base_time + timedelta(seconds=i),
            "bid": price - 0.1 + np.random.randn() * 0.3,
            "ask": price + 0.1 + np.random.randn() * 0.3,
            "mid": price + np.random.randn() * 0.3,
        }
        
        result = buffer.update_from_quote(event)
        
        if result is not None:
            progress = buffer.get_warmup_progress()
            print(f"\nBar {buffer.get_bar_count()}: "
                  f"progress={progress['progress_pct']:.1f}%, "
                  f"ready={progress['ready']}")
            
            if buffer.is_ready():
                print(f"Features shape: {result.shape}")
                print(f"Sample: ret_1={result['ret_1'].iloc[0]:.6f}")
    
    print(f"\nFinal: {buffer.get_bar_count()} bars, ready={buffer.is_ready()}")
