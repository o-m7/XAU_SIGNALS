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

# Import unified feature engineering (SAME AS TRAINING!)
from src.features import build_features, get_all_feature_names

# Suppress pandas FutureWarnings about fillna downcasting
warnings.filterwarnings('ignore', category=FutureWarning, message='.*Downcasting.*')

logger = logging.getLogger("FeatureBuffer")


# Feature names from unified registry (SAME AS TRAINING!)
FEATURE_NAMES = get_all_feature_names()

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
        
        # Store latest quote bid/ask for merging into aggregate bars
        self._last_quote_bid: Optional[float] = None
        self._last_quote_ask: Optional[float] = None
        self._last_quote_time: Optional[datetime] = None
        
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
        
        Quotes provide bid/ask for microstructure features.
        We store the latest bid/ask to merge into bars from aggregates.
        
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
        
        # Store latest bid/ask for merging into aggregate bars
        bid = event.get("bid")
        ask = event.get("ask")
        if bid is not None and ask is not None:
            self._last_quote_bid = bid
            self._last_quote_ask = ask
            self._last_quote_time = timestamp
        
        mid = event.get("mid") or ((bid or 0) + (ask or 0)) / 2 if (bid and ask) else None
        
        if mid is None:
            return None  # Skip invalid quotes
        
        # Synthetic tick from quote (for bar aggregation)
        tick = {
            "timestamp": timestamp,
            "price": mid,
            "bid": bid,
            "ask": ask,
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
        
        # Round timestamp to minute for bar matching
        timestamp_minute = timestamp.replace(second=0, microsecond=0)
        
        # Check if we have a recent quote with bid/ask for this minute
        # Use the most recent bid/ask from quotes (within the same minute)
        bid_price = None
        ask_price = None
        if (self._last_quote_bid is not None and 
            self._last_quote_ask is not None and 
            self._last_quote_time is not None):
            # Use last quote bid/ask if within same minute
            quote_time_minute = self._last_quote_time.replace(second=0, microsecond=0)
            if quote_time_minute == timestamp_minute:
                bid_price = self._last_quote_bid
                ask_price = self._last_quote_ask
        
        # Direct bar insertion with merged bid/ask from quotes
        bar = {
            "timestamp": timestamp,
            "open": event["open"],
            "high": event["high"],
            "low": event["low"],
            "close": event["close"],
            "volume": event.get("volume", 0),
            "bid_price": bid_price,  # Merged from quotes if available
            "ask_price": ask_price,  # Merged from quotes if available
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
        Compute features for the latest bar using UNIFIED MODULE.

        CRITICAL: Uses SAME feature engineering as training (src.features.build_features)
        to ensure perfect parity between training and live inference.

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

        # Build quotes DataFrame if bid/ask available
        quotes = None
        if "bid_price" in df.columns and "ask_price" in df.columns:
            quotes = df[["bid_price", "ask_price"]].copy()

        # USE UNIFIED FEATURES (SAME AS TRAINING!)
        # This replaces 400 lines of inline computation with single source of truth
        try:
            features = build_features(df, quotes=quotes, feature_set="all")
        except Exception as e:
            logger.error(f"Error computing features: {e}")
            raise

        # Return only the latest row
        latest = features.iloc[[-1]].copy()

        # Ensure all required features exist
        missing_features = []
        for feat in FEATURE_NAMES:
            if feat not in latest.columns:
                latest[feat] = np.nan
                missing_features.append(feat)

        if missing_features:
            logger.warning(f"⚠️ Missing {len(missing_features)} features: {missing_features[:10]}")

        # Check for NaN values
        nan_count = latest[FEATURE_NAMES].isna().sum().sum()
        if nan_count > 0:
            nan_features = latest[FEATURE_NAMES].isna().sum()
            nan_features = nan_features[nan_features > 0].index.tolist()
            logger.warning(f"⚠️ {nan_count} NaN values in features: {nan_features[:10]}")
            # Fill NaN with 0 (better than leaving as NaN)
            latest[FEATURE_NAMES] = latest[FEATURE_NAMES].fillna(0)

        # Check for constant/zero features (data flow issue)
        feature_std = latest[FEATURE_NAMES].std()
        constant_features = feature_std[feature_std < 1e-6].index.tolist()
        if constant_features and len(self._bars) > 10:
            logger.debug(f"Constant features (possible data issue): {constant_features[:5]}")

        # Select only required features in correct order
        return latest[FEATURE_NAMES]


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
