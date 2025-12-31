#!/usr/bin/env python3
"""
Massive WebSocket Client for Live Price Ingestion.

Uses the official Massive Python SDK (formerly Polygon.io).
Polygon.io rebranded to Massive.com on October 30, 2025.

Supports connecting to ALL stream types simultaneously:
- quotes: Real-time forex quotes (C.XAU/USD)
- aggs_minute: Minute aggregates (CA.XAU/USD)
- aggs_second: Second aggregates (CAS.XAU/USD)

The Massive library handles authentication, subscriptions, and
reconnection automatically. Includes heartbeat monitoring for
connection health tracking.
"""

import time
import logging
import threading
from datetime import datetime, timezone
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import requests

try:
    from massive import WebSocketClient
    from massive.websocket.models import WebSocketMessage, Feed, Market
    MASSIVE_AVAILABLE = True
except ImportError:
    MASSIVE_AVAILABLE = False
    logging.warning("massive not installed. Install with: pip install massive")

from .symbol_resolver import SymbolResolver

# Configure logging
logger = logging.getLogger("PolygonStream")


class WSMode(str, Enum):
    """WebSocket subscription mode."""
    QUOTES = "quotes"
    AGGS_MINUTE = "aggs_minute"
    AGGS_SECOND = "aggs_second"
    ALL = "all"  # Subscribe to all channels


@dataclass
class StreamEvent:
    """
    Normalized stream event.

    For quotes: type="quote", bid/ask populated
    For bars: type="bar", OHLCV populated
    """
    type: str  # "quote" or "bar"
    timestamp: datetime
    bid: Optional[float] = None
    ask: Optional[float] = None
    mid: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    source: Optional[str] = None  # "quotes", "aggs_minute", "aggs_second"

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "type": self.type,
            "timestamp": self.timestamp,
            "source": self.source,
        }
        if self.type == "quote":
            d.update({
                "bid": self.bid,
                "ask": self.ask,
                "mid": self.mid,
            })
        else:  # bar
            d.update({
                "open": self.open,
                "high": self.high,
                "low": self.low,
                "close": self.close,
                "volume": self.volume,
            })
        return d

    @property
    def price(self) -> float:
        """Get the primary price (mid for quotes, close for bars)."""
        if self.type == "quote":
            return self.mid or 0.0
        return self.close or 0.0


class PolygonStream:
    """
    Live price streaming using the Massive WebSocket client.

    The Massive library (formerly Polygon.io) handles authentication
    and reconnection automatically.

    Can connect to ALL channels simultaneously for maximum data.
    PERSISTENT - includes heartbeat monitoring for connection health.

    Args:
        api_key: Massive/Polygon API key
        resolver: SymbolResolver instance
        symbol: Currency pair symbol (e.g., "XAU/USD") - alternative to resolver
        mode: WebSocket mode (quotes, aggs_minute, aggs_second, or "all")
        on_event: Callback function receiving StreamEvent dict
    """

    # REST endpoint for last quote fallback
    REST_URL = "https://api.polygon.io"

    # Heartbeat interval (seconds)
    HEARTBEAT_INTERVAL = 30

    # Stale connection threshold (seconds) - force reconnect if no data
    STALE_THRESHOLD = 120

    def __init__(
        self,
        api_key: str,
        resolver: Optional[SymbolResolver] = None,
        symbol: Optional[str] = None,
        mode: WSMode = WSMode.ALL,
        on_event: Optional[Callable[[Dict], None]] = None,
    ):
        self.api_key = api_key

        # Handle both resolver and symbol parameters
        if resolver is not None:
            self.resolver = resolver
            self.symbol = f"{resolver.base}/{resolver.quote}"
        else:
            self.resolver = None
            self.symbol = symbol or "XAU/USD"

        self.mode = WSMode(mode) if isinstance(mode, str) else mode
        self.on_event = on_event or self._default_event_handler

        # State
        self._client: Optional[WebSocketClient] = None
        self._thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False
        self._events_received = 0
        self._last_event_time: Optional[datetime] = None
        self._reconnect_count = 0
        self._subscribed_channels: List[str] = []
        self._target_channels: List[str] = []

        display_name = resolver.display_name() if resolver else self.symbol
        logger.info(
            f"PolygonStream initialized: {display_name}, mode={mode.value if hasattr(mode, 'value') else mode}"
        )

    def _default_event_handler(self, event: Dict):
        """Default event handler - just logs."""
        logger.debug(f"Event: {event}")

    def _get_subscription_channels(self) -> List[str]:
        """Get the WebSocket subscription channels based on mode."""
        if self.resolver:
            # Use resolver methods for channel names
            if self.mode == WSMode.ALL:
                channels = [
                    self.resolver.ws_quotes(),      # C.XAU/USD
                    self.resolver.ws_aggs_minute(), # CA.XAU/USD
                    self.resolver.ws_aggs_second(), # CAS.XAU/USD
                ]
                logger.info(f"Mode=ALL: Will subscribe to {len(channels)} channels")
                return channels
            elif self.mode == WSMode.QUOTES:
                logger.info("Mode=QUOTES: Will subscribe to 1 channel")
                return [self.resolver.ws_quotes()]
            elif self.mode == WSMode.AGGS_MINUTE:
                logger.info("Mode=AGGS_MINUTE: Will subscribe to 1 channel")
                return [self.resolver.ws_aggs_minute()]
            elif self.mode == WSMode.AGGS_SECOND:
                logger.info("Mode=AGGS_SECOND: Will subscribe to 1 channel")
                return [self.resolver.ws_aggs_second()]
        else:
            # Build channels from symbol string
            quote = f"C.{self.symbol}"
            agg_min = f"CA.{self.symbol}"
            agg_sec = f"CAS.{self.symbol}"

            if self.mode == WSMode.ALL:
                channels = [quote, agg_min, agg_sec]
                logger.info(f"Mode=ALL: Subscribing to {len(channels)} channels")
                return channels
            elif self.mode == WSMode.QUOTES:
                logger.info("Mode=QUOTES: Subscribing to 1 channel")
                return [quote]
            elif self.mode == WSMode.AGGS_MINUTE:
                logger.info("Mode=AGGS_MINUTE: Subscribing to 1 channel")
                return [agg_min]
            elif self.mode == WSMode.AGGS_SECOND:
                logger.info("Mode=AGGS_SECOND: Subscribing to 1 channel")
                return [agg_sec]

        logger.warning(f"Unknown mode: {self.mode}, defaulting to quotes")
        return [f"C.{self.symbol}"]

    def _handle_messages(self, msgs: List[WebSocketMessage]):
        """Process incoming messages from Massive client."""
        for msg in msgs:
            ev = getattr(msg, 'ev', None)

            if ev == "C":
                self._handle_quote(msg)
            elif ev == "CA":
                self._handle_aggregate(msg, "aggs_minute")
            elif ev == "CAS":
                self._handle_aggregate(msg, "aggs_second")
            # Status messages (auth_success, subscribed, etc.) are handled
            # internally by the Massive library

    def _handle_quote(self, msg: WebSocketMessage):
        """Convert quote message to StreamEvent."""
        try:
            ts_ms = getattr(msg, 't', 0)
            timestamp = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else datetime.now(timezone.utc)

            bid = getattr(msg, 'b', None) or getattr(msg, 'bp', None)
            ask = getattr(msg, 'a', None) or getattr(msg, 'ap', None)

            if bid is None or ask is None:
                return  # Skip invalid quotes silently

            mid = (bid + ask) / 2

            event = StreamEvent(
                type="quote",
                timestamp=timestamp,
                bid=bid,
                ask=ask,
                mid=mid,
                source="quotes",
            )

            self._last_event_time = datetime.now(timezone.utc)
            self._events_received += 1
            self.on_event(event.to_dict())

        except Exception as e:
            logger.debug(f"Quote processing error: {e}")

    def _handle_aggregate(self, msg: WebSocketMessage, source: str):
        """Convert aggregate message to StreamEvent."""
        try:
            # Use end timestamp (e) or start timestamp (s)
            ts_ms = getattr(msg, 'e', 0) or getattr(msg, 's', 0)
            timestamp = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc) if ts_ms else datetime.now(timezone.utc)

            event = StreamEvent(
                type="bar",
                timestamp=timestamp,
                open=getattr(msg, 'o', None),
                high=getattr(msg, 'h', None),
                low=getattr(msg, 'l', None),
                close=getattr(msg, 'c', None),
                volume=getattr(msg, 'v', None),
                source=source,
            )

            self._last_event_time = datetime.now(timezone.utc)
            self._events_received += 1
            self.on_event(event.to_dict())

        except Exception as e:
            logger.debug(f"Aggregate processing error: {e}")

    def start(self):
        """Start the WebSocket stream."""
        if self._running:
            logger.warning("Stream already running")
            return

        if not MASSIVE_AVAILABLE:
            raise ImportError("massive not installed. Install with: pip install massive")

        self._running = True
        self._reconnect_count = 0

        # Store target channels (only set once, persist across reconnects)
        if not self._target_channels:
            self._target_channels = self._get_subscription_channels()
            logger.info(f"Target channels set: {len(self._target_channels)} channels: {', '.join(self._target_channels)}")

        self._subscribed_channels = self._target_channels.copy()

        # Create Massive WebSocket client
        self._client = WebSocketClient(
            api_key=self.api_key,
            feed=Feed.RealTime,
            market=Market.Forex,
            subscriptions=self._subscribed_channels,
        )

        # Run in background thread (client.run is blocking)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

        # Start heartbeat monitor
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self._heartbeat_thread.start()

        logger.info(f"Stream started: {self.symbol}, channels={self._subscribed_channels}")

    def _run(self):
        """Run the Massive client (blocking)."""
        try:
            self._client.run(handle_msg=self._handle_messages)
        except Exception as e:
            if self._running:
                logger.error(f"Stream error: {e}")
                self._reconnect_count += 1

    def stop(self):
        """Stop the WebSocket stream."""
        logger.info("Stopping stream...")
        self._running = False

        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.debug(f"Error closing client: {e}")

        # Wait for threads to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5)

        logger.info(f"Stream stopped. Events received: {self._events_received}")

    # =========================================================================
    # REST Fallback (Last Quote Only)
    # =========================================================================

    def fetch_last_quote(self) -> Optional[StreamEvent]:
        """
        Fetch the last forex quote via REST API.

        This is ONLY used as a keepalive/health check, not for backfilling.

        Returns:
            StreamEvent or None if failed
        """
        if self.resolver:
            base, quote = self.resolver.rest_last_quote_args()
        else:
            # Parse symbol like "XAU/USD"
            parts = self.symbol.split("/")
            if len(parts) != 2:
                logger.error(f"Invalid symbol format: {self.symbol}")
                return None
            base, quote = parts

        url = f"{self.REST_URL}/v1/last_quote/currencies/{base}/{quote}"
        params = {"apiKey": self.api_key}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "success" and data.get("last"):
                last = data["last"]

                # Timestamp may be in different formats
                ts = last.get("timestamp")
                if ts:
                    if isinstance(ts, (int, float)):
                        # Assume milliseconds if large number
                        if ts > 1e12:
                            ts = ts / 1000
                        timestamp = datetime.fromtimestamp(ts, tz=timezone.utc)
                    else:
                        timestamp = datetime.now(timezone.utc)
                else:
                    timestamp = datetime.now(timezone.utc)

                bid = last.get("bid")
                ask = last.get("ask")

                if bid and ask:
                    return StreamEvent(
                        type="quote",
                        timestamp=timestamp,
                        bid=bid,
                        ask=ask,
                        mid=(bid + ask) / 2,
                        source="rest",
                    )

        except Exception as e:
            logger.error(f"REST last quote error: {e}")

        return None

    # =========================================================================
    # Heartbeat Monitor
    # =========================================================================

    def _heartbeat_loop(self):
        """Monitor connection health and force reconnect if stale."""
        while self._running:
            time.sleep(self.HEARTBEAT_INTERVAL)

            if not self._running:
                break

            if self._last_event_time:
                age = (datetime.now(timezone.utc) - self._last_event_time).total_seconds()
                channels_str = ', '.join(self._subscribed_channels) if self._subscribed_channels else "none"
                logger.info(
                    f"Heartbeat: {self._events_received} events, "
                    f"last {age:.1f}s ago, reconnects={self._reconnect_count}, "
                    f"channels={len(self._subscribed_channels)} ({channels_str})"
                )

                # Force reconnect if connection seems stale
                if age > self.STALE_THRESHOLD and self._client:
                    logger.warning(f"Connection stale ({age:.0f}s), forcing reconnect...")
                    self._reconnect_count += 1
                    try:
                        self._client.close()
                    except:
                        pass
                    # Restart the client
                    self._restart_client()
            else:
                logger.warning("Heartbeat: no events received yet")

    def _restart_client(self):
        """Restart the WebSocket client after stale detection."""
        if not self._running:
            return

        try:
            self._client = WebSocketClient(
                api_key=self.api_key,
                feed=Feed.RealTime,
                market=Market.Forex,
                subscriptions=self._subscribed_channels,
            )

            # Run in new thread
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            logger.info("Client restarted after stale connection")
        except Exception as e:
            logger.error(f"Failed to restart client: {e}")

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._running and self._client is not None

    @property
    def events_received(self) -> int:
        """Get total events received."""
        return self._events_received

    @property
    def subscribed_channels(self) -> List[str]:
        """Get list of subscribed channels."""
        return self._subscribed_channels.copy()


# =============================================================================
# Standalone test
# =============================================================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("Set POLYGON_API_KEY in .env file")
        exit(1)

    resolver = SymbolResolver("XAU", "USD")

    def print_event(event):
        src = event.get("source", "?")
        print(f"[{event['timestamp']}] [{src}] {event['type'].upper()}: ", end="")
        if event["type"] == "quote":
            print(f"bid={event['bid']:.2f}, ask={event['ask']:.2f}, mid={event['mid']:.2f}")
        else:
            print(f"O={event['open']:.2f}, H={event['high']:.2f}, L={event['low']:.2f}, C={event['close']:.2f}")

    # Connect to ALL channels
    stream = PolygonStream(
        api_key=api_key,
        resolver=resolver,
        mode=WSMode.ALL,
        on_event=print_event,
    )

    try:
        stream.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stream.stop()
