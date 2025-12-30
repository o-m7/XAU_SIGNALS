#!/usr/bin/env python3
"""
REST API Historical Backfill for Live Trading.

Fetches historical bars and quotes from Polygon REST API
to warm up the feature buffer before live streaming begins.
"""

import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path

import requests
import pandas as pd

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


def backfill_feature_buffer(
    feature_buffer,
    api_key: str,
    resolver: SymbolResolver,
    lookback_bars: int = 500,
) -> bool:
    """
    Backfill the feature buffer with historical data.
    
    Args:
        feature_buffer: FeatureBuffer instance to populate
        api_key: Polygon API key
        resolver: SymbolResolver instance
        lookback_bars: Number of bars to fetch
        
    Returns:
        True if backfill successful
    """
    logger.info("=" * 60)
    logger.info("  HISTORICAL BACKFILL (REST API)")
    logger.info("=" * 60)
    
    backfill = PolygonBackfill(api_key, resolver)
    
    # Fetch bars
    bars_df = backfill.fetch_minute_bars(lookback_bars=lookback_bars)
    
    if bars_df.empty:
        logger.error("Failed to fetch historical bars")
        return False
    
    # Get last quote for bid/ask
    last_quote = backfill.fetch_last_quote()
    
    if last_quote:
        logger.info(f"Last quote: bid={last_quote['bid']:.2f}, ask={last_quote['ask']:.2f}")
    
    # Feed bars into feature buffer
    logger.info(f"Loading {len(bars_df)} bars into feature buffer...")
    
    for idx, row in bars_df.iterrows():
        # Create bar event
        event = {
            "type": "bar",
            "timestamp": row["timestamp"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row.get("volume", 0),
        }
        
        # Add bid/ask from last quote (approximate)
        if last_quote:
            # Estimate historical spread based on current
            spread = last_quote["ask"] - last_quote["bid"]
            event["bid"] = row["close"] - spread / 2
            event["ask"] = row["close"] + spread / 2
        
        feature_buffer.update_from_bar(event)
    
    logger.info(f"Feature buffer: {feature_buffer.get_bar_count()} bars, ready={feature_buffer.is_ready()}")
    logger.info("=" * 60)
    
    return feature_buffer.is_ready()


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

