#!/usr/bin/env python3
"""
REST API Historical Backfill for Live Trading.

Fetches historical bars AND quotes from Polygon REST API
to warm up the feature buffer before live streaming begins.

Features:
- Fetch OHLCV minute bars
- Fetch historical quotes (bid/ask)
- Periodic refresh (every 30-60 minutes)
- Efficient merge of quotes into bars
"""

import os
import time
import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path

import requests
import pandas as pd
import numpy as np

from .symbol_resolver import SymbolResolver

logger = logging.getLogger("Backfill")


class PolygonBackfill:
    """
    Fetch historical data from Polygon REST API.
    
    Used to warm up the feature buffer before live streaming.
    """
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self, api_key: str, resolver: SymbolResolver):
        self.api_key = api_key
        self.resolver = resolver
        
    def fetch_minute_bars(
        self,
        lookback_bars: int = 500,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical minute bars.
        
        Args:
            lookback_bars: Number of bars to fetch (max ~50,000)
            end_time: End time (default: now - ensures recent data)
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, vwap, trades
        """
        # Always use current time to ensure we get the most recent data
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        else:
            # If end_time is provided but is old, use current time instead
            now = datetime.now(timezone.utc)
            if (now - end_time).total_seconds() > 3600:  # If more than 1 hour old
                logger.warning(f"End time {end_time} is old, using current time {now} instead")
                end_time = now
        
        # Calculate start time (add buffer for weekends/holidays)
        # Assume ~400 bars per trading day for forex
        days_needed = max(3, lookback_bars // 400 + 2)
        start_time = end_time - timedelta(days=days_needed)
        
        logger.info(f"Fetching recent bars: end_time={end_time}, start_time={start_time}")
        
        symbol = self.resolver.rest_aggs_symbol()
        
        start_str = start_time.strftime("%Y-%m-%d")
        end_str = end_time.strftime("%Y-%m-%d")
        
        logger.info(f"Fetching minute bars: {symbol} from {start_str} to {end_str}")
        
        url = f"{self.BASE_URL}/v2/aggs/ticker/{symbol}/range/1/minute/{start_str}/{end_str}"
        
        all_results = []
        params = {
            "apiKey": self.api_key,
            "adjusted": "true",
            "sort": "asc",
            "limit": 50000,
        }
        
        while url and len(all_results) < lookback_bars * 2:  # Fetch extra to ensure we have enough
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"Error fetching bars: {response.status_code}")
                break
            
            data = response.json()
            
            if data.get("results"):
                all_results.extend(data["results"])
                logger.info(f"  Fetched {len(data['results'])} bars, total: {len(all_results)}")
            
            # Check for pagination
            next_url = data.get("next_url")
            if next_url and len(all_results) < lookback_bars * 2:
                url = next_url
                params = {"apiKey": self.api_key}
                time.sleep(0.15)  # Rate limiting
            else:
                url = None
        
        if not all_results:
            logger.warning("No bars returned from REST API")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Rename columns
        df = df.rename(columns={
            "v": "volume",
            "vw": "vwap",
            "o": "open",
            "c": "close",
            "h": "high",
            "l": "low",
            "t": "timestamp",
            "n": "trades",
        })
        
        # Convert timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        
        # Sort by time and take last N bars
        df = df.sort_values("timestamp").tail(lookback_bars).reset_index(drop=True)
        
        logger.info(f"Backfill complete: {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def fetch_last_quote(self) -> Optional[Dict[str, Any]]:
        """
        Fetch the last forex quote.

        Returns:
            Dict with bid, ask, mid, timestamp or None
        """
        base, quote = self.resolver.rest_last_quote_args()
        url = f"{self.BASE_URL}/v1/last_quote/currencies/{base}/{quote}"

        params = {"apiKey": self.api_key}

        try:
            response = requests.get(url, params=params, timeout=10)

            if response.status_code != 200:
                logger.error(f"Error fetching last quote: {response.status_code}")
                return None

            data = response.json()

            if data.get("status") == "success" and data.get("last"):
                last = data["last"]
                bid = last.get("bid")
                ask = last.get("ask")

                if bid and ask:
                    return {
                        "bid": bid,
                        "ask": ask,
                        "mid": (bid + ask) / 2,
                        "timestamp": datetime.now(timezone.utc),
                    }

        except Exception as e:
            logger.error(f"Error fetching last quote: {e}")

        return None

    def fetch_quotes(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit: int = 50000,
    ) -> pd.DataFrame:
        """
        Fetch historical quotes from Polygon REST API.

        Args:
            start_time: Start of quote range
            end_time: End of quote range (default: now)
            limit: Max quotes per request

        Returns:
            DataFrame with columns: timestamp, bid_price, ask_price
        """
        if end_time is None:
            end_time = datetime.now(timezone.utc)

        # Convert to nanoseconds for Polygon API
        start_ns = int(start_time.timestamp() * 1e9)
        end_ns = int(end_time.timestamp() * 1e9)

        # Get the forex pair symbol for quotes endpoint
        base, quote_sym = self.resolver.rest_last_quote_args()
        symbol = f"C:{base}{quote_sym}"

        logger.info(f"Fetching quotes: {symbol} from {start_time} to {end_time}")

        url = f"{self.BASE_URL}/v3/quotes/{symbol}"

        all_results = []
        params = {
            "apiKey": self.api_key,
            "timestamp.gte": start_ns,
            "timestamp.lte": end_ns,
            "limit": limit,
            "sort": "timestamp",
            "order": "asc",
        }

        # Paginate through results
        max_pages = 10  # Limit pages to avoid excessive API calls
        page = 0

        while url and page < max_pages:
            try:
                response = requests.get(url, params=params, timeout=30)

                if response.status_code != 200:
                    logger.error(f"Error fetching quotes: {response.status_code} - {response.text[:200]}")
                    break

                data = response.json()

                if data.get("results"):
                    all_results.extend(data["results"])
                    logger.info(f"  Fetched {len(data['results'])} quotes, total: {len(all_results)}")

                # Check for pagination
                next_url = data.get("next_url")
                if next_url:
                    url = next_url
                    params = {"apiKey": self.api_key}
                    time.sleep(0.15)  # Rate limiting
                    page += 1
                else:
                    break

            except Exception as e:
                logger.error(f"Error fetching quotes: {e}")
                break

        if not all_results:
            logger.warning("No quotes returned from REST API")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_results)

        # Extract bid/ask from nested structure if needed
        if "bid_price" not in df.columns and "bid" in df.columns:
            df["bid_price"] = df["bid"].apply(lambda x: x.get("p") if isinstance(x, dict) else x)
            df["ask_price"] = df["ask"].apply(lambda x: x.get("p") if isinstance(x, dict) else x)
        elif "bp" in df.columns:  # Alternative column names
            df["bid_price"] = df["bp"]
            df["ask_price"] = df["ap"]

        # Convert timestamp (nanoseconds to datetime)
        if "sip_timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["sip_timestamp"], unit="ns", utc=True)
        elif "participant_timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["participant_timestamp"], unit="ns", utc=True)
        elif "t" in df.columns:
            df["timestamp"] = pd.to_datetime(df["t"], unit="ns", utc=True)

        # Select only needed columns
        if "bid_price" in df.columns and "ask_price" in df.columns:
            df = df[["timestamp", "bid_price", "ask_price"]].copy()
            df = df.dropna(subset=["bid_price", "ask_price"])
            df = df.sort_values("timestamp").reset_index(drop=True)

            logger.info(f"Quotes complete: {len(df)} quotes from {df['timestamp'].min()} to {df['timestamp'].max()}")
            return df
        else:
            logger.warning(f"Could not find bid/ask columns. Available: {df.columns.tolist()}")
            return pd.DataFrame()


def merge_quotes_into_bars(bars_df: pd.DataFrame, quotes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge quotes into bars using time-based matching.

    For each bar, find the closest quote within that minute and use its bid/ask.

    Args:
        bars_df: DataFrame with OHLCV bars (timestamp column)
        quotes_df: DataFrame with quotes (timestamp, bid_price, ask_price)

    Returns:
        bars_df with bid_price and ask_price columns added
    """
    if quotes_df.empty:
        logger.warning("No quotes to merge, using spread estimation")
        return bars_df

    bars_df = bars_df.copy()

    # Ensure timestamps are datetime
    if not pd.api.types.is_datetime64_any_dtype(bars_df["timestamp"]):
        bars_df["timestamp"] = pd.to_datetime(bars_df["timestamp"], utc=True)
    if not pd.api.types.is_datetime64_any_dtype(quotes_df["timestamp"]):
        quotes_df["timestamp"] = pd.to_datetime(quotes_df["timestamp"], utc=True)

    # Sort both by timestamp
    bars_df = bars_df.sort_values("timestamp").reset_index(drop=True)
    quotes_df = quotes_df.sort_values("timestamp").reset_index(drop=True)

    # Use merge_asof to find closest quote for each bar
    merged = pd.merge_asof(
        bars_df,
        quotes_df[["timestamp", "bid_price", "ask_price"]],
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta("2min"),
    )

    # Forward-fill any remaining NaN bid/ask
    merged["bid_price"] = merged["bid_price"].ffill()
    merged["ask_price"] = merged["ask_price"].ffill()

    # If still NaN (no quotes at all), estimate from close price
    if merged["bid_price"].isna().any():
        # Estimate typical spread as 0.03% of price for gold
        avg_close = merged["close"].mean()
        spread = avg_close * 0.0003
        merged["bid_price"] = merged["bid_price"].fillna(merged["close"] - spread / 2)
        merged["ask_price"] = merged["ask_price"].fillna(merged["close"] + spread / 2)

    logger.info(f"Merged quotes into bars: {len(merged)} bars with bid/ask")
    return merged


def backfill_feature_buffer(
    feature_buffer,
    api_key: str,
    resolver: SymbolResolver,
    lookback_bars: int = 500,
    fetch_quotes: bool = True,
) -> bool:
    """
    Backfill the feature buffer with historical data (bars + quotes).

    Args:
        feature_buffer: FeatureBuffer instance to populate
        api_key: Polygon API key
        resolver: SymbolResolver instance
        lookback_bars: Number of bars to fetch
        fetch_quotes: Whether to fetch historical quotes (recommended)

    Returns:
        True if backfill successful
    """
    logger.info("=" * 60)
    logger.info("  HISTORICAL BACKFILL (REST API)")
    logger.info("=" * 60)

    backfill = PolygonBackfill(api_key, resolver)

    # Fetch bars with explicit recent end_time
    end_time = datetime.now(timezone.utc)
    bars_df = backfill.fetch_minute_bars(lookback_bars=lookback_bars, end_time=end_time)

    if bars_df.empty:
        logger.error("Failed to fetch historical bars")
        return False

    # Fetch historical quotes for accurate bid/ask
    quotes_df = pd.DataFrame()
    if fetch_quotes and not bars_df.empty:
        start_time = bars_df["timestamp"].min()
        quotes_df = backfill.fetch_quotes(start_time=start_time, end_time=end_time)

        if not quotes_df.empty:
            logger.info(f"Fetched {len(quotes_df)} historical quotes")
            # Merge quotes into bars
            bars_df = merge_quotes_into_bars(bars_df, quotes_df)
        else:
            logger.warning("No quotes fetched, using spread estimation")
            # Fallback: Get last quote and estimate spread
            last_quote = backfill.fetch_last_quote()
            if last_quote:
                spread = last_quote["ask"] - last_quote["bid"]
                bars_df["bid_price"] = bars_df["close"] - spread / 2
                bars_df["ask_price"] = bars_df["close"] + spread / 2
            else:
                # Use typical spread estimate
                spread = bars_df["close"].mean() * 0.0003
                bars_df["bid_price"] = bars_df["close"] - spread / 2
                bars_df["ask_price"] = bars_df["close"] + spread / 2

    # Feed bars into feature buffer
    logger.info(f"Loading {len(bars_df)} bars into feature buffer...")

    for idx, row in bars_df.iterrows():
        # Create bar event with bid/ask
        event = {
            "type": "bar",
            "timestamp": row["timestamp"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row.get("volume", 0),
            "bid_price": row.get("bid_price"),
            "ask_price": row.get("ask_price"),
        }

        feature_buffer.update_from_bar(event)

    logger.info(f"Feature buffer: {feature_buffer.get_bar_count()} bars, ready={feature_buffer.is_ready()}")
    logger.info("=" * 60)

    return feature_buffer.is_ready()


class BackfillManager:
    """
    Manages periodic backfill refresh during live trading.

    Periodically fetches fresh data from REST API to ensure
    the feature buffer stays synchronized with market data.
    """

    def __init__(
        self,
        feature_buffer,
        api_key: str,
        resolver: SymbolResolver,
        refresh_interval_minutes: int = 30,
        lookback_bars: int = 100,
    ):
        """
        Initialize the backfill manager.

        Args:
            feature_buffer: FeatureBuffer instance to update
            api_key: Polygon API key
            resolver: SymbolResolver instance
            refresh_interval_minutes: How often to refresh (default: 30 min)
            lookback_bars: Bars to fetch on each refresh
        """
        self.feature_buffer = feature_buffer
        self.api_key = api_key
        self.resolver = resolver
        self.refresh_interval = refresh_interval_minutes * 60  # Convert to seconds
        self.lookback_bars = lookback_bars

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_refresh: Optional[datetime] = None
        self._refresh_count = 0

        logger.info(
            f"BackfillManager initialized: refresh every {refresh_interval_minutes} min, "
            f"lookback={lookback_bars} bars"
        )

    def start(self):
        """Start the periodic refresh thread."""
        if self._running:
            logger.warning("BackfillManager already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._refresh_loop, daemon=True)
        self._thread.start()
        logger.info("BackfillManager started")

    def stop(self):
        """Stop the periodic refresh thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info(f"BackfillManager stopped (refreshed {self._refresh_count} times)")

    def _refresh_loop(self):
        """Background thread that periodically refreshes data."""
        while self._running:
            # Sleep first (initial backfill already done)
            for _ in range(self.refresh_interval):
                if not self._running:
                    return
                time.sleep(1)

            # Do refresh
            self._do_refresh()

    def _do_refresh(self):
        """Perform a single refresh."""
        try:
            logger.info("=" * 40)
            logger.info("  PERIODIC BACKFILL REFRESH")
            logger.info("=" * 40)

            backfill = PolygonBackfill(self.api_key, self.resolver)

            # Fetch recent bars
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(minutes=self.lookback_bars + 10)

            bars_df = backfill.fetch_minute_bars(
                lookback_bars=self.lookback_bars,
                end_time=end_time
            )

            if bars_df.empty:
                logger.warning("Refresh: No bars fetched")
                return

            # Fetch quotes for these bars
            quotes_df = backfill.fetch_quotes(start_time=start_time, end_time=end_time)

            if not quotes_df.empty:
                bars_df = merge_quotes_into_bars(bars_df, quotes_df)

            # Update feature buffer with fresh bars
            # Only add bars that are newer than what we have
            current_bar_count = self.feature_buffer.get_bar_count()
            new_bars = 0

            for idx, row in bars_df.iterrows():
                event = {
                    "type": "bar",
                    "timestamp": row["timestamp"],
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                    "volume": row.get("volume", 0),
                    "bid_price": row.get("bid_price"),
                    "ask_price": row.get("ask_price"),
                }

                # This will add to buffer (deque handles max size)
                self.feature_buffer.update_from_bar(event)
                new_bars += 1

            self._last_refresh = datetime.now(timezone.utc)
            self._refresh_count += 1

            logger.info(
                f"Refresh complete: processed {new_bars} bars, "
                f"buffer has {self.feature_buffer.get_bar_count()} bars"
            )
            logger.info("=" * 40)

        except Exception as e:
            logger.error(f"Refresh failed: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the backfill manager."""
        return {
            "running": self._running,
            "last_refresh": self._last_refresh,
            "refresh_count": self._refresh_count,
            "refresh_interval_min": self.refresh_interval // 60,
        }


# =============================================================================
# Standalone test
# =============================================================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    
    # Load env
    for p in [Path("/Users/omar/Desktop/ML/.env"), Path(".env")]:
        if p.exists():
            load_dotenv(p)
            break
    
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("Set POLYGON_API_KEY")
        exit(1)
    
    resolver = SymbolResolver("XAU", "USD")
    backfill = PolygonBackfill(api_key, resolver)
    
    # Test bar fetch
    bars = backfill.fetch_minute_bars(lookback_bars=100)
    print(f"\nFetched {len(bars)} bars")
    print(bars.tail())
    
    # Test quote fetch
    quote = backfill.fetch_last_quote()
    print(f"\nLast quote: {quote}")

