#!/usr/bin/env python3
"""
Trend Filter - Prevents counter-trend signals.

Uses multiple timeframe trend detection to filter out signals that go against
the dominant market direction. This prevents SHORT signals during strong uptrends
and LONG signals during strong downtrends.

Trend Detection Methods:
1. SMA-based: Price vs 50-bar SMA
2. EMA-based: 20-bar EMA vs 50-bar EMA (golden/death cross)
3. Returns-based: Rolling 30-bar return direction

Usage:
    filter = TrendFilter(method='sma', lookback=50)
    is_allowed = filter.should_allow_signal(signal='LONG', feature_row=df_row)
"""

import logging
from typing import Optional
import numpy as np
import pandas as pd
from enum import Enum

logger = logging.getLogger("TrendFilter")


class TrendDirection(Enum):
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"


class TrendFilter:
    """
    Filters signals based on trend direction.

    Prevents counter-trend trades:
    - Blocks SHORT signals in uptrends
    - Blocks LONG signals in downtrends
    - Allows all signals in neutral/sideways markets

    Args:
        method: Trend detection method ('sma', 'ema', 'returns', 'combined')
        lookback: Bars to use for trend calculation
        buffer_pct: Percentage buffer around SMA for neutral zone (default: 0.1%)
        enabled: Whether the filter is active (default: True)
    """

    def __init__(
        self,
        method: str = 'sma',
        lookback: int = 50,
        buffer_pct: float = 0.001,  # 0.1% buffer for neutral zone
        enabled: bool = True
    ):
        self.method = method
        self.lookback = lookback
        self.buffer_pct = buffer_pct
        self.enabled = enabled

        # Trend history for debugging/logging
        self.last_trend = TrendDirection.NEUTRAL
        self.signals_blocked = 0
        self.signals_allowed = 0

        logger.info(
            f"TrendFilter initialized: method={method}, lookback={lookback}, "
            f"buffer={buffer_pct:.2%}, enabled={enabled}"
        )

    def get_trend_direction(
        self,
        close: float,
        sma_50: Optional[float] = None,
        ema_20: Optional[float] = None,
        ema_50: Optional[float] = None,
        ret_30: Optional[float] = None
    ) -> TrendDirection:
        """
        Determine current trend direction.

        Args:
            close: Current close price
            sma_50: 50-bar Simple Moving Average (if available)
            ema_20: 20-bar EMA (for EMA method)
            ema_50: 50-bar EMA (for EMA method)
            ret_30: 30-bar return (for returns method)

        Returns:
            TrendDirection enum (UP, DOWN, or NEUTRAL)
        """
        if self.method == 'sma':
            return self._sma_trend(close, sma_50)
        elif self.method == 'ema':
            return self._ema_trend(ema_20, ema_50)
        elif self.method == 'returns':
            return self._returns_trend(ret_30)
        elif self.method == 'combined':
            return self._combined_trend(close, sma_50, ema_20, ema_50, ret_30)
        else:
            logger.warning(f"Unknown trend method '{self.method}', defaulting to NEUTRAL")
            return TrendDirection.NEUTRAL

    def _sma_trend(self, close: float, sma_50: Optional[float]) -> TrendDirection:
        """
        SMA-based trend detection.

        UP: Price > SMA50 * (1 + buffer)
        DOWN: Price < SMA50 * (1 - buffer)
        NEUTRAL: Price within buffer zone
        """
        if sma_50 is None or np.isnan(sma_50) or sma_50 <= 0:
            return TrendDirection.NEUTRAL

        upper_bound = sma_50 * (1 + self.buffer_pct)
        lower_bound = sma_50 * (1 - self.buffer_pct)

        if close > upper_bound:
            return TrendDirection.UP
        elif close < lower_bound:
            return TrendDirection.DOWN
        else:
            return TrendDirection.NEUTRAL

    def _ema_trend(
        self,
        ema_20: Optional[float],
        ema_50: Optional[float]
    ) -> TrendDirection:
        """
        EMA crossover-based trend detection.

        UP: EMA20 > EMA50 (golden cross territory)
        DOWN: EMA20 < EMA50 (death cross territory)
        """
        if ema_20 is None or ema_50 is None:
            return TrendDirection.NEUTRAL
        if np.isnan(ema_20) or np.isnan(ema_50):
            return TrendDirection.NEUTRAL

        if ema_20 > ema_50:
            return TrendDirection.UP
        elif ema_20 < ema_50:
            return TrendDirection.DOWN
        else:
            return TrendDirection.NEUTRAL

    def _returns_trend(self, ret_30: Optional[float]) -> TrendDirection:
        """
        Returns-based trend detection.

        UP: 30-bar return > 0.5% (significant upward movement)
        DOWN: 30-bar return < -0.5% (significant downward movement)
        """
        if ret_30 is None or np.isnan(ret_30):
            return TrendDirection.NEUTRAL

        threshold = 0.005  # 0.5% threshold

        if ret_30 > threshold:
            return TrendDirection.UP
        elif ret_30 < -threshold:
            return TrendDirection.DOWN
        else:
            return TrendDirection.NEUTRAL

    def _combined_trend(
        self,
        close: float,
        sma_50: Optional[float],
        ema_20: Optional[float],
        ema_50: Optional[float],
        ret_30: Optional[float]
    ) -> TrendDirection:
        """
        Combined trend detection using multiple signals.

        Requires at least 2 out of 3 methods to agree for a trend direction.
        """
        votes = {
            TrendDirection.UP: 0,
            TrendDirection.DOWN: 0,
            TrendDirection.NEUTRAL: 0
        }

        # SMA vote
        votes[self._sma_trend(close, sma_50)] += 1

        # EMA vote
        votes[self._ema_trend(ema_20, ema_50)] += 1

        # Returns vote
        votes[self._returns_trend(ret_30)] += 1

        # Need majority (2 out of 3) for directional trend
        if votes[TrendDirection.UP] >= 2:
            return TrendDirection.UP
        elif votes[TrendDirection.DOWN] >= 2:
            return TrendDirection.DOWN
        else:
            return TrendDirection.NEUTRAL

    def should_allow_signal(
        self,
        signal: str,
        feature_row: pd.DataFrame,
        close_col: str = 'close'
    ) -> bool:
        """
        Check if a signal should be allowed based on trend direction.

        Args:
            signal: Signal type ('LONG', 'SHORT', 'FLAT')
            feature_row: Single-row DataFrame with features
            close_col: Name of close price column

        Returns:
            True if signal is allowed, False if blocked
        """
        # If filter disabled, allow all signals
        if not self.enabled:
            self.signals_allowed += 1
            return True

        # FLAT signals are always allowed
        if signal == 'FLAT':
            self.signals_allowed += 1
            return True

        # Extract features for trend detection
        close = feature_row[close_col].iloc[0] if isinstance(feature_row, pd.DataFrame) else feature_row[close_col]

        # Get available trend features
        sma_50 = self._get_feature(feature_row, ['sma_50', 'price_vs_sma50', 'SMA_50'])
        ema_20 = self._get_feature(feature_row, ['ema_20', 'EMA_20'])
        ema_50 = self._get_feature(feature_row, ['ema_50', 'EMA_50'])
        ret_30 = self._get_feature(feature_row, ['ret_30', 'return_30', 'momentum_30'])

        # If using SMA method and sma_50 not available, calculate from price_vs_sma20
        if sma_50 is None and 'price_vs_sma20' in feature_row.columns:
            price_vs_sma20 = self._get_feature(feature_row, ['price_vs_sma20'])
            if price_vs_sma20 is not None:
                # price_vs_sma20 = (close - sma20) / sma20, so sma20 = close / (1 + price_vs_sma20)
                # Use this as a proxy for trend direction
                if price_vs_sma20 > 0.002:  # 0.2% above SMA20
                    trend = TrendDirection.UP
                elif price_vs_sma20 < -0.002:  # 0.2% below SMA20
                    trend = TrendDirection.DOWN
                else:
                    trend = TrendDirection.NEUTRAL

                self.last_trend = trend
                return self._filter_by_trend(signal, trend)

        # Determine trend
        trend = self.get_trend_direction(close, sma_50, ema_20, ema_50, ret_30)
        self.last_trend = trend

        return self._filter_by_trend(signal, trend)

    def _get_feature(self, feature_row, possible_names):
        """Get feature value from DataFrame, trying multiple possible names."""
        for name in possible_names:
            if name in feature_row.columns:
                val = feature_row[name].iloc[0] if isinstance(feature_row, pd.DataFrame) else feature_row[name]
                if not np.isnan(val):
                    return val
        return None

    def _filter_by_trend(self, signal: str, trend: TrendDirection) -> bool:
        """Apply trend filter logic."""
        # LONG signals blocked in downtrends
        if signal == 'LONG' and trend == TrendDirection.DOWN:
            self.signals_blocked += 1
            logger.debug(f"Blocked LONG signal in downtrend")
            return False

        # SHORT signals blocked in uptrends
        if signal == 'SHORT' and trend == TrendDirection.UP:
            self.signals_blocked += 1
            logger.debug(f"Blocked SHORT signal in uptrend")
            return False

        # Signal allowed
        self.signals_allowed += 1
        return True

    def get_stats(self) -> dict:
        """Get filter statistics."""
        total = self.signals_allowed + self.signals_blocked
        return {
            'enabled': self.enabled,
            'method': self.method,
            'signals_allowed': self.signals_allowed,
            'signals_blocked': self.signals_blocked,
            'block_rate': self.signals_blocked / total if total > 0 else 0,
            'last_trend': self.last_trend.value
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.signals_blocked = 0
        self.signals_allowed = 0


# Convenience function for quick filtering
def filter_counter_trend_signals(
    signals: list,
    feature_df: pd.DataFrame,
    method: str = 'sma'
) -> list:
    """
    Filter out counter-trend signals from a list.

    Args:
        signals: List of (timestamp, signal_type, confidence) tuples
        feature_df: DataFrame with features indexed by timestamp
        method: Trend detection method

    Returns:
        Filtered list of signals
    """
    trend_filter = TrendFilter(method=method)
    filtered = []

    for timestamp, signal_type, confidence in signals:
        if timestamp in feature_df.index:
            row = feature_df.loc[[timestamp]]
            if trend_filter.should_allow_signal(signal_type, row):
                filtered.append((timestamp, signal_type, confidence))

    return filtered
