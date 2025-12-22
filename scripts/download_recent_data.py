#!/usr/bin/env python3
"""
Download recent XAUUSD data from Polygon.io.

Downloads minute bars and quotes for a specified date range
and saves them in the same format as existing data files.

Usage:
    python scripts/download_recent_data.py --start 2025-12-07 --end 2025-12-22
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import time

import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Paths
DATA_DIR = Path("/Users/omar/Desktop/ML/Data")
MINUTE_DIR = DATA_DIR / "ohlcv_minute"
QUOTES_DIR = DATA_DIR / "quotes"

# Polygon API
API_KEY = os.environ.get("POLYGON_API_KEY")
BASE_URL = "https://api.polygon.io"


def fetch_minute_bars(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch minute bars from Polygon REST API.
    
    Args:
        symbol: Polygon symbol (e.g., "C:XAUUSD")
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        
    Returns:
        DataFrame with minute bars
    """
    print(f"Fetching minute bars: {start_date} to {end_date}")
    
    all_results = []
    
    # Polygon aggregates endpoint
    url = f"{BASE_URL}/v2/aggs/ticker/{symbol}/range/1/minute/{start_date}/{end_date}"
    
    params = {
        "apiKey": API_KEY,
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
    }
    
    while url:
        print(f"  Requesting: {url[:80]}...")
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"  Error: {response.status_code} - {response.text}")
            break
        
        data = response.json()
        
        if data.get("results"):
            all_results.extend(data["results"])
            print(f"  Got {len(data['results'])} bars, total: {len(all_results)}")
        
        # Check for pagination
        next_url = data.get("next_url")
        if next_url:
            url = next_url
            params = {"apiKey": API_KEY}  # next_url includes other params
            time.sleep(0.15)  # Rate limiting
        else:
            url = None
    
    if not all_results:
        print("  No data returned!")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Rename columns to match existing format
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
    
    # Convert timestamp from milliseconds to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    
    # Select and order columns
    cols = ["volume", "vwap", "open", "close", "high", "low", "timestamp", "trades"]
    df = df[[c for c in cols if c in df.columns]]
    
    print(f"  Final: {len(df)} bars from {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def fetch_quotes(from_currency: str, to_currency: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch forex quotes from Polygon REST API.
    
    Args:
        from_currency: Base currency (e.g., "XAU")
        to_currency: Quote currency (e.g., "USD")
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        
    Returns:
        DataFrame with quotes
    """
    print(f"Fetching quotes: {start_date} to {end_date}")
    
    all_results = []
    
    # Need to fetch day by day for quotes (large volume)
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    current = start
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        
        # Forex quotes endpoint
        url = f"{BASE_URL}/v3/quotes/{from_currency}{to_currency}"
        
        params = {
            "apiKey": API_KEY,
            "timestamp.gte": f"{date_str}T00:00:00Z",
            "timestamp.lte": f"{date_str}T23:59:59Z",
            "order": "asc",
            "limit": 50000,
        }
        
        day_results = []
        page_url = url
        
        while page_url:
            print(f"  {date_str}: fetching...")
            response = requests.get(page_url, params=params if page_url == url else {"apiKey": API_KEY})
            
            if response.status_code != 200:
                print(f"    Error: {response.status_code}")
                break
            
            data = response.json()
            
            if data.get("results"):
                day_results.extend(data["results"])
            
            # Check pagination
            next_url = data.get("next_url")
            if next_url and len(day_results) < 500000:  # Safety limit
                page_url = next_url
                time.sleep(0.15)
            else:
                page_url = None
        
        if day_results:
            print(f"    Got {len(day_results)} quotes")
            all_results.extend(day_results)
        
        current += timedelta(days=1)
        time.sleep(0.2)  # Rate limiting between days
    
    if not all_results:
        print("  No quotes returned!")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    
    # Rename columns to match existing format
    df = df.rename(columns={
        "participant_timestamp": "timestamp",
        "sip_timestamp": "sip_timestamp",
        "ask_price": "ask_price",
        "bid_price": "bid_price",
        "ask_size": "ask_size",
        "bid_size": "bid_size",
    })
    
    # Convert timestamps
    if "timestamp" in df.columns:
        # Polygon returns nanoseconds
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ns", utc=True)
    
    print(f"  Final: {len(df)} quotes")
    
    return df


def append_to_existing(new_df: pd.DataFrame, existing_path: Path, timestamp_col: str = "timestamp") -> pd.DataFrame:
    """
    Append new data to existing parquet file, avoiding duplicates.
    """
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        print(f"  Existing file has {len(existing)} rows")
        
        # Get max timestamp from existing
        existing_max = pd.to_datetime(existing[timestamp_col]).max()
        print(f"  Existing max timestamp: {existing_max}")
        
        # Filter new data to only include records after existing max
        new_df[timestamp_col] = pd.to_datetime(new_df[timestamp_col])
        new_df = new_df[new_df[timestamp_col] > existing_max]
        
        if len(new_df) == 0:
            print("  No new data to append")
            return existing
        
        print(f"  Appending {len(new_df)} new rows")
        
        # Combine
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.sort_values(timestamp_col).reset_index(drop=True)
        
        return combined
    else:
        return new_df


def main():
    parser = argparse.ArgumentParser(description="Download recent XAUUSD data from Polygon")
    parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--minute-only", action="store_true", help="Only download minute bars")
    parser.add_argument("--quotes-only", action="store_true", help="Only download quotes")
    
    args = parser.parse_args()
    
    if not API_KEY:
        print("ERROR: POLYGON_API_KEY not set in environment or .env file")
        sys.exit(1)
    
    print("=" * 60)
    print("XAUUSD Data Download")
    print("=" * 60)
    print(f"Date range: {args.start} to {args.end}")
    print(f"API Key: {API_KEY[:8]}...")
    print()
    
    # Download minute bars
    if not args.quotes_only:
        print("\n[1] Downloading minute bars...")
        minute_df = fetch_minute_bars("C:XAUUSD", args.start, args.end)
        
        if len(minute_df) > 0:
            # Append to 2025 file
            minute_path = MINUTE_DIR / "XAUUSD_minute_2025.parquet"
            combined = append_to_existing(minute_df, minute_path)
            
            # Save
            combined.to_parquet(minute_path, index=False)
            print(f"  Saved to {minute_path}")
            print(f"  Total rows: {len(combined)}")
            print(f"  Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
    
    # Download quotes
    if not args.minute_only:
        print("\n[2] Downloading quotes...")
        quotes_df = fetch_quotes("XAU", "USD", args.start, args.end)
        
        if len(quotes_df) > 0:
            # Append to 2025 file
            quotes_path = QUOTES_DIR / "XAUUSD_quotes_2025.parquet"
            combined = append_to_existing(quotes_df, quotes_path)
            
            # Save
            combined.to_parquet(quotes_path, index=False)
            print(f"  Saved to {quotes_path}")
            print(f"  Total rows: {len(combined)}")
    
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

