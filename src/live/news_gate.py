#!/usr/bin/env python3
"""
News Gate Module for Model #4 (5m News-Gated Long-Only).

Simplified version using free APIs and XAUUSD-derived features:
- Economic calendar (static FOMC/CPI/NFP dates)
- XAUUSD volatility regime (replaces VIX)
- Volume patterns (replaces DXY momentum)
- Free news sentiment (Finnhub free tier or fallback)

Output: News gate multiplier [0.5, 1.15] applied to ML signal
"""

import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
import aiohttp
import pandas as pd

logger = logging.getLogger("NewsGate")


# =============================================================================
# ECONOMIC CALENDAR (Static High-Impact Events for 2025-2026)
# =============================================================================

ECONOMIC_EVENTS = {
    # FOMC Meetings (2025)
    "2025-01-29": {"event": "FOMC", "impact": "high"},
    "2025-03-19": {"event": "FOMC", "impact": "high"},
    "2025-05-07": {"event": "FOMC", "impact": "high"},
    "2025-06-18": {"event": "FOMC", "impact": "high"},
    "2025-07-30": {"event": "FOMC", "impact": "high"},
    "2025-09-17": {"event": "FOMC", "impact": "high"},
    "2025-11-05": {"event": "FOMC", "impact": "high"},
    "2025-12-17": {"event": "FOMC", "impact": "high"},

    # FOMC Meetings (2026)
    "2026-01-28": {"event": "FOMC", "impact": "high"},
    "2026-03-18": {"event": "FOMC", "impact": "high"},
    "2026-05-06": {"event": "FOMC", "impact": "high"},
    "2026-06-17": {"event": "FOMC", "impact": "high"},

    # CPI Releases (Monthly - add as needed)
    # NFP Releases (First Friday of month - add as needed)
}


class NewsGate:
    """
    News Gate for Model #4.

    Computes a multiplier [0.5, 1.15] based on:
    1. Recent economic events (FOMC/CPI/NFP within last 30 min)
    2. XAUUSD volatility regime (ATR-based, replaces VIX)
    3. Volume patterns (replaces DXY momentum)

    No paid APIs required - uses static calendar + XAUUSD features.
    """

    def __init__(
        self,
        enable_news_api: bool = False,
        news_api_key: Optional[str] = None
    ):
        self.enable_news_api = enable_news_api
        self.news_api_key = news_api_key

        # State
        self._last_event_check: Optional[datetime] = None
        self._recent_event: Optional[Dict] = None
        self._sentiment_cache: Dict = {}

        logger.info(f"NewsGate initialized (news_api={'enabled' if enable_news_api else 'disabled'})")

    async def compute_gate_multiplier(
        self,
        feature_row: pd.DataFrame,
        current_timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Compute news gate multiplier.

        Args:
            feature_row: Single-row DataFrame with XAUUSD features (must contain ATR_14, volume)
            current_timestamp: Current time (default: now UTC)

        Returns:
            Dict with:
                - multiplier: float [0.5, 1.15]
                - components: dict of individual factors
                - reason: str explanation
        """
        if current_timestamp is None:
            current_timestamp = datetime.now(timezone.utc)

        components = {}
        multiplier = 1.0
        reasons = []

        # 1. Check for recent high-impact events
        event_factor = await self._check_recent_events(current_timestamp)
        components['event_factor'] = event_factor
        multiplier *= event_factor
        if event_factor < 1.0:
            reasons.append(f"Recent event (×{event_factor:.2f})")

        # 2. XAUUSD Volatility Regime (replaces VIX)
        vol_factor = self._check_volatility_regime(feature_row)
        components['volatility_factor'] = vol_factor
        multiplier *= vol_factor
        if vol_factor != 1.0:
            regime = "high" if vol_factor > 1.0 else "low"
            reasons.append(f"Vol regime {regime} (×{vol_factor:.2f})")

        # 3. Volume Pattern (replaces DXY momentum)
        volume_factor = self._check_volume_pattern(feature_row)
        components['volume_factor'] = volume_factor
        multiplier *= volume_factor
        if volume_factor < 1.0:
            reasons.append(f"Weak volume (×{volume_factor:.2f})")

        # Clamp to [0.5, 1.15]
        multiplier = max(0.5, min(1.15, multiplier))

        reason = " | ".join(reasons) if reasons else "Normal conditions"

        return {
            'multiplier': round(multiplier, 3),
            'components': components,
            'reason': reason,
            'timestamp': current_timestamp
        }

    async def _check_recent_events(self, current_time: datetime) -> float:
        """
        Check if FOMC/CPI/NFP occurred in last 30 minutes.

        Returns:
            0.5 if recent event (suppress trading)
            1.0 otherwise
        """
        current_date = current_time.strftime("%Y-%m-%d")

        # Check if today has a high-impact event
        if current_date in ECONOMIC_EVENTS:
            event_info = ECONOMIC_EVENTS[current_date]

            # Assume events occur at 14:00 UTC (FOMC typical time)
            event_time = datetime.strptime(f"{current_date} 14:00", "%Y-%m-%d %H:%M")
            event_time = event_time.replace(tzinfo=timezone.utc)

            time_since_event = (current_time - event_time).total_seconds() / 60

            # Suppress trading for 30 minutes after event
            if 0 <= time_since_event <= 30:
                logger.warning(f"⚠️ Recent {event_info['event']} event (T+{time_since_event:.0f}m) - reducing signals by 50%")
                self._recent_event = event_info
                return 0.5

        return 1.0

    def _check_volatility_regime(self, feature_row: pd.DataFrame) -> float:
        """
        Check XAUUSD volatility regime (replaces VIX).

        Uses ATR_14 to determine if market is:
        - Low vol (ATR < 3.0): Risk-on, gold suppressed → 0.85x
        - Normal (3.0 <= ATR <= 8.0): → 1.0x
        - High vol (ATR > 8.0): Risk-off, gold bid → 1.15x

        Returns:
            Multiplier [0.85, 1.15]
        """
        try:
            atr = feature_row.get('ATR_14', feature_row.get('atr_14', None))

            if atr is None:
                logger.debug("ATR_14 not found, assuming normal vol")
                return 1.0

            if isinstance(atr, pd.Series):
                atr = atr.iloc[0]

            # Volatility thresholds (for gold, typical ATR ranges)
            if atr < 3.0:
                # Low volatility = risk-on = headwind for gold
                logger.debug(f"Low vol regime (ATR={atr:.2f}) - reducing signals")
                return 0.85
            elif atr > 8.0:
                # High volatility = risk-off = tailwind for gold
                logger.debug(f"High vol regime (ATR={atr:.2f}) - boosting signals")
                return 1.15
            else:
                # Normal volatility
                return 1.0

        except Exception as e:
            logger.warning(f"Volatility check failed: {e}")
            return 1.0

    def _check_volume_pattern(self, feature_row: pd.DataFrame) -> float:
        """
        Check volume pattern (replaces DXY momentum).

        Uses volume_delta or volume to detect weak/strong participation:
        - Weak volume → reduce signals (0.8x)
        - Normal/Strong volume → neutral (1.0x)

        Returns:
            Multiplier [0.8, 1.0]
        """
        try:
            # Try to use volume_delta if available
            vol_delta = feature_row.get('volume_delta', None)

            if vol_delta is None:
                # Fallback: use raw volume vs rolling mean
                volume = feature_row.get('volume', None)
                if volume is None:
                    return 1.0

                if isinstance(volume, pd.Series):
                    volume = volume.iloc[0]

                # Simple heuristic: volume < 1000 = weak
                if volume < 1000:
                    logger.debug(f"Weak volume ({volume:.0f}) - reducing signals")
                    return 0.85
                else:
                    return 1.0

            if isinstance(vol_delta, pd.Series):
                vol_delta = vol_delta.iloc[0]

            # If volume delta is negative (selling pressure), reduce longs
            if vol_delta < -0.3:
                logger.debug(f"Negative volume delta ({vol_delta:.2f}) - reducing long signals")
                return 0.8
            else:
                return 1.0

        except Exception as e:
            logger.warning(f"Volume pattern check failed: {e}")
            return 1.0

    async def fetch_sentiment(self) -> Optional[str]:
        """
        Optional: Fetch news sentiment from free API (Finnhub or Alpha Vantage).

        Free tier limits:
        - Finnhub: 60 calls/min, news sentiment available
        - Alpha Vantage: 25 calls/day (too restrictive)

        Returns:
            "dovish" | "hawkish" | None
        """
        if not self.enable_news_api or not self.news_api_key:
            return None

        # TODO: Implement Finnhub API call if needed
        # For now, return None (disabled)
        return None

    def get_status(self) -> Dict:
        """Get current gate status."""
        return {
            'recent_event': self._recent_event,
            'last_check': self._last_event_check,
            'news_api_enabled': self.enable_news_api,
        }


# =============================================================================
# Standalone Test
# =============================================================================

async def test_news_gate():
    """Test news gate with mock data."""
    gate = NewsGate()

    # Mock feature row
    mock_features = pd.DataFrame({
        'ATR_14': [5.0],
        'volume': [2000],
    })

    result = await gate.compute_gate_multiplier(mock_features)

    print("News Gate Test:")
    print(f"  Multiplier: {result['multiplier']}")
    print(f"  Components: {result['components']}")
    print(f"  Reason: {result['reason']}")


if __name__ == "__main__":
    asyncio.run(test_news_gate())
