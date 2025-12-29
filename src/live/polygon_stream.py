#!/usr/bin/env python3
"""
Polygon.io WebSocket Client for Live Price Ingestion.

Supports connecting to ALL stream types simultaneously:
- quotes: Real-time forex quotes (C.XAU/USD)
- aggs_minute: Minute aggregates (CA.XAU/USD)
- aggs_second: Second aggregates (CAS.XAU/USD)

Uses SymbolResolver for correct symbol formatting.
Includes automatic reconnection and heartbeat monitoring.
PERSISTENT connection - never disconnects voluntarily.
"""

import os
import json
import time
import logging
import threading
from datetime import datetime, timezone
from typing import Callable, Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import requests

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logging.warning("websocket-client not installed. Install with: pip install websocket-client")

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
    Live price streaming from Polygon.io WebSocket.
    
    Can connect to ALL channels simultaneously for maximum data.
    PERSISTENT - automatically reconnects forever.
    
    Args:
        api_key: Polygon.io API key
        resolver: SymbolResolver instance
        mode: WebSocket mode (quotes, aggs_minute, aggs_second, or "all")
        on_event: Callback function receiving StreamEvent dict
    """
    
    # Polygon WebSocket endpoint
    WS_URL = "wss://socket.polygon.io/forex"
    
    # REST endpoint for last quote fallback
    REST_URL = "https://api.polygon.io"
    
    # Heartbeat interval (seconds)
    HEARTBEAT_INTERVAL = 30
    
    # Reconnect settings - PERSISTENT connection (unlimited retries)
    MAX_RECONNECT_ATTEMPTS = 999999  # Effectively unlimited
    RECONNECT_DELAY_INITIAL = 1  # Start with 1 second
    RECONNECT_DELAY_MAX = 60  # Max 60 seconds between retries
    
    def __init__(
        self,
        api_key: str,
        resolver: SymbolResolver,
        mode: WSMode = WSMode.ALL,
        on_event: Optional[Callable[[Dict], None]] = None,
    ):
        self.api_key = api_key
        self.resolver = resolver
        self.mode = WSMode(mode) if isinstance(mode, str) else mode
        self.on_event = on_event or self._default_event_handler
        
        # State
        self._running = False
        self._ws: Optional[websocket.WebSocketApp] = None
        self._ws_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._last_event_time: Optional[datetime] = None
        self._reconnect_count = 0
        self._events_received = 0
        self._subscribed_channels: List[str] = []
        
        logger.info(
            f"PolygonStream initialized: {resolver.display_name()}, mode={mode.value if hasattr(mode, 'value') else mode}"
        )
    
    def _default_event_handler(self, event: Dict):
        """Default event handler - just logs."""
        logger.debug(f"Event: {event}")
    
    def _get_subscription_channels(self) -> List[str]:
        """Get the WebSocket subscription channels based on mode."""
        if self.mode == WSMode.ALL:
            # Subscribe to ALL channels
            channels = [
                self.resolver.ws_quotes(),      # C.XAU/USD
                self.resolver.ws_aggs_minute(), # CA.XAU/USD
                self.resolver.ws_aggs_second(), # CAS.XAU/USD
            ]
            logger.info(f"🔌 Mode=ALL: Will subscribe to {len(channels)} channels")
            return channels
        elif self.mode == WSMode.QUOTES:
            logger.info(f"🔌 Mode=QUOTES: Will subscribe to 1 channel")
            return [self.resolver.ws_quotes()]
        elif self.mode == WSMode.AGGS_MINUTE:
            logger.info(f"🔌 Mode=AGGS_MINUTE: Will subscribe to 1 channel")
            return [self.resolver.ws_aggs_minute()]
        elif self.mode == WSMode.AGGS_SECOND:
            logger.info(f"🔌 Mode=AGGS_SECOND: Will subscribe to 1 channel")
            return [self.resolver.ws_aggs_second()]
        else:
            logger.error(f"❌ Unknown WS mode: {self.mode}")
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def start(self):
        """Start the WebSocket stream."""
        if self._running:
            logger.warning("Stream already running")
            return
        
        if not WEBSOCKET_AVAILABLE:
            raise ImportError("websocket-client not installed")
        
        self._running = True
        self._reconnect_count = 0
        
        # Store target channels (only set once, persist across reconnects)
        if not self._target_channels:
            self._target_channels = self._get_subscription_channels()
            logger.info(f"🎯 Target channels set: {len(self._target_channels)} channels: {', '.join(self._target_channels)}")
        else:
            logger.info(f"🔄 Reusing existing target channels: {len(self._target_channels)} channels: {', '.join(self._target_channels)}")
        
        self._start_websocket()
        
        # Start heartbeat monitor
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self._heartbeat_thread.start()
    
    def stop(self):
        """Stop the WebSocket stream."""
        logger.info("Stopping Polygon stream...")
        self._running = False
        
        if self._ws:
            try:
                self._ws.close()
            except Exception as e:
                logger.error(f"Error closing WebSocket: {e}")
        
        # Wait for threads
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=5)
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5)
        
        logger.info(f"Stream stopped. Events received: {self._events_received}")
    
    # =========================================================================
    # WebSocket Implementation
    # =========================================================================
    
    def _start_websocket(self):
        """Start WebSocket connection."""
        self._ws = websocket.WebSocketApp(
            self.WS_URL,
            on_open=self._on_ws_open,
            on_message=self._on_ws_message,
            on_error=self._on_ws_error,
            on_close=self._on_ws_close,
        )
        
        self._ws_thread = threading.Thread(
            target=self._ws.run_forever,
            kwargs={"ping_interval": 20, "ping_timeout": 10},
            daemon=True
        )
        self._ws_thread.start()
    
    def _on_ws_open(self, ws):
        """Handle WebSocket connection open."""
        logger.info("WebSocket connected")
        self._reconnect_count = 0
        
        # Authenticate
        auth_msg = {"action": "auth", "params": self.api_key}
        ws.send(json.dumps(auth_msg))
    
    def _on_ws_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            
            if isinstance(data, list):
                for msg in data:
                    self._process_message(msg)
            else:
                self._process_message(data)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse message: {e}")
    
    def _process_message(self, msg: Dict):
        """Process a single WebSocket message."""
        ev = msg.get("ev")
        
        if ev == "status":
            status = msg.get("status")
            message = msg.get("message", "")
            logger.info(f"Status: {status} - {message}")
            
            if status == "auth_success":
                # Subscribe to the SAME target channels (don't regenerate list)
                if not self._target_channels:
                    # Fallback: get channels if target_channels not set (shouldn't happen)
                    self._target_channels = self._get_subscription_channels()
                    logger.warning("⚠️ Target channels not set, regenerating (shouldn't happen)")
                
                # Only subscribe to channels we're not already subscribed to
                channels_to_subscribe = [ch for ch in self._target_channels if ch not in self._subscribed_channels]
                
                if channels_to_subscribe:
                    logger.info(f"🔌 Subscribing to {len(channels_to_subscribe)} channel(s): {', '.join(channels_to_subscribe)}")
                    for channel in channels_to_subscribe:
                        sub_msg = {"action": "subscribe", "params": channel}
                        self._ws.send(json.dumps(sub_msg))
                        logger.info(f"  ✓ Subscribed to: {channel}")
                        self._subscribed_channels.append(channel)
                        time.sleep(0.1)  # Small delay between subscriptions
                    logger.info(f"✅ Successfully connected to all {len(self._subscribed_channels)} channels: {', '.join(self._subscribed_channels)}")
                else:
                    logger.info(f"✅ Already subscribed to all {len(self._subscribed_channels)} channels: {', '.join(self._subscribed_channels)}")
            
            elif status == "auth_failed":
                # Stop trying to reconnect on auth failure
                logger.error(f"AUTH FAILED: {message}")
                logger.error("Check your Polygon API key and plan. Stopping reconnection attempts.")
                self._running = False  # Stop the reconnection loop
                
        elif ev == "C":
            # Forex quote
            self._handle_quote(msg)
            
        elif ev == "CA":
            # Minute aggregate
            self._handle_aggregate(msg, source="aggs_minute")
            
        elif ev == "CAS":
            # Second aggregate
            self._handle_aggregate(msg, source="aggs_second")
    
    def _handle_quote(self, msg: Dict):
        """Handle forex quote message (ev=C)."""
        try:
            # Polygon forex quote format
            ts_ms = msg.get("t", 0)
            timestamp = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
            
            bid = msg.get("b") or msg.get("bp")
            ask = msg.get("a") or msg.get("ap")
            
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
            # Suppress quote processing errors (non-critical)
            logger.debug(f"Error processing quote: {e}")
    
    def _handle_aggregate(self, msg: Dict, source: str = "aggs"):
        """Handle aggregate bar message (ev=CA or ev=CAS)."""
        try:
            # Polygon aggregate format
            # s = start timestamp, e = end timestamp
            # Sometimes Polygon sends arrays, sometimes single values
            ts_ms = msg.get("e") or msg.get("s", 0)
            
            # Handle both single values and arrays
            if isinstance(ts_ms, (list, tuple)):
                # If timestamp is an array, process each bar
                opens = msg.get("o", [])
                highs = msg.get("h", [])
                lows = msg.get("l", [])
                closes = msg.get("c", [])
                volumes = msg.get("v", [])
                
                # Ensure all are lists
                if not isinstance(opens, (list, tuple)):
                    opens = [opens]
                if not isinstance(highs, (list, tuple)):
                    highs = [highs]
                if not isinstance(lows, (list, tuple)):
                    lows = [lows]
                if not isinstance(closes, (list, tuple)):
                    closes = [closes]
                if not isinstance(volumes, (list, tuple)):
                    volumes = [volumes]
                
                # Find the minimum length to avoid index mismatches
                # Polygon sometimes sends arrays of slightly different lengths
                n_bars = min(
                    len(ts_ms),
                    len(opens) if isinstance(opens, (list, tuple)) else 1,
                    len(highs) if isinstance(highs, (list, tuple)) else 1,
                    len(lows) if isinstance(lows, (list, tuple)) else 1,
                    len(closes) if isinstance(closes, (list, tuple)) else 1,
                    len(volumes) if isinstance(volumes, (list, tuple)) else 1,
                )
                
                # Process each bar in the batch (only up to minimum length)
                for i in range(n_bars):
                    try:
                        # Safely extract values with bounds checking
                        timestamp = datetime.fromtimestamp(ts_ms[i] / 1000, tz=timezone.utc)
                        event = StreamEvent(
                            type="bar",
                            timestamp=timestamp,
                            open=opens[i] if isinstance(opens, (list, tuple)) and i < len(opens) else (opens if not isinstance(opens, (list, tuple)) else None),
                            high=highs[i] if isinstance(highs, (list, tuple)) and i < len(highs) else (highs if not isinstance(highs, (list, tuple)) else None),
                            low=lows[i] if isinstance(lows, (list, tuple)) and i < len(lows) else (lows if not isinstance(lows, (list, tuple)) else None),
                            close=closes[i] if isinstance(closes, (list, tuple)) and i < len(closes) else (closes if not isinstance(closes, (list, tuple)) else None),
                            volume=volumes[i] if isinstance(volumes, (list, tuple)) and i < len(volumes) else (volumes if not isinstance(volumes, (list, tuple)) else None),
                            source=source,
                        )
                        self._last_event_time = datetime.now(timezone.utc)
                        self._events_received += 1
                        self.on_event(event.to_dict())
                    except (IndexError, TypeError, ValueError, KeyError) as e:
                        logger.debug(f"Skipping invalid bar in batch (index {i}/{n_bars}): {e}")
                        continue
            else:
                # Single aggregate bar
                timestamp = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
                
                # Extract values, handling both single values and arrays
                open_val = msg.get("o")
                high_val = msg.get("h")
                low_val = msg.get("l")
                close_val = msg.get("c")
                volume_val = msg.get("v")
                
                # If any value is an array, take the last element (most recent)
                if isinstance(open_val, (list, tuple)) and len(open_val) > 0:
                    open_val = open_val[-1]
                if isinstance(high_val, (list, tuple)) and len(high_val) > 0:
                    high_val = high_val[-1]
                if isinstance(low_val, (list, tuple)) and len(low_val) > 0:
                    low_val = low_val[-1]
                if isinstance(close_val, (list, tuple)) and len(close_val) > 0:
                    close_val = close_val[-1]
                if isinstance(volume_val, (list, tuple)) and len(volume_val) > 0:
                    volume_val = volume_val[-1]
                
                event = StreamEvent(
                    type="bar",
                    timestamp=timestamp,
                    open=open_val,
                    high=high_val,
                    low=low_val,
                    close=close_val,
                    volume=volume_val,
                    source=source,
                )
                
                self._last_event_time = datetime.now(timezone.utc)
                self._events_received += 1
                self.on_event(event.to_dict())
            
        except Exception as e:
            # Suppress aggregate processing errors (non-critical, handled gracefully)
            logger.debug(f"Error processing aggregate: {e}")
    
    def _on_ws_error(self, ws, error):
        """Handle WebSocket error."""
        # Log as debug (reconnection handles it automatically)
        logger.debug(f"WebSocket error: {error} (reconnecting...)")
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close - attempt reconnect with exponential backoff."""
        # Log as info (reconnection is automatic, not an error)
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg} (reconnecting...)")
        
        if self._running:
            self._reconnect_count += 1
            # Clear active subscriptions (will resubscribe to same target_channels on reconnect)
            # BUT keep _target_channels so we resubscribe to the SAME channels
            old_channels = self._subscribed_channels.copy()
            self._subscribed_channels = []
            logger.info(f"🔄 Reconnecting... Will resubscribe to same {len(self._target_channels)} channels: {', '.join(self._target_channels)}")
            
            # Limit reconnection attempts to avoid infinite loop
            if self._reconnect_count > 10:
                logger.error(f"Too many reconnection attempts ({self._reconnect_count}). Stopping.")
                logger.error("Check your API key and internet connection.")
                self._running = False
                return
            
            # Exponential backoff: 1, 2, 4, 8, 16, 32, 60, 60, 60...
            delay = min(
                self.RECONNECT_DELAY_INITIAL * (2 ** min(self._reconnect_count - 1, 5)),
                self.RECONNECT_DELAY_MAX
            )
            
            logger.info(f"Reconnecting in {delay}s (attempt #{self._reconnect_count}/10)...")
            time.sleep(delay)
            
            # Only reconnect if still running
            if self._running:
                self._start_websocket()
    
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
        base, quote = self.resolver.rest_last_quote_args()
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
        STALE_THRESHOLD = 120  # Force reconnect if no data for 2 minutes
        
        while self._running:
            time.sleep(self.HEARTBEAT_INTERVAL)
            
            if not self._running:
                break
            
            if self._last_event_time:
                age = (datetime.now(timezone.utc) - self._last_event_time).total_seconds()
                channels_str = ', '.join(self._subscribed_channels) if self._subscribed_channels else "none"
                logger.info(
                    f"♥ Heartbeat: {self._events_received} events, "
                    f"last {age:.1f}s ago, reconnects={self._reconnect_count}, "
                    f"channels={len(self._subscribed_channels)} ({channels_str})"
                )
                
                # Force reconnect if connection seems stale
                if age > STALE_THRESHOLD and self._ws:
                    logger.warning(f"Connection stale ({age:.0f}s), forcing reconnect...")
                    try:
                        self._ws.close()
                    except:
                        pass
            else:
                logger.warning("♥ Heartbeat: no events received yet")
    
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._ws is not None and self._running
    
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
        mode=WSMode.ALL,  # All channels!
        on_event=print_event,
    )
    
    try:
        stream.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stream.stop()
