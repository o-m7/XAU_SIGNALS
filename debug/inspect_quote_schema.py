#!/usr/bin/env python3
"""
Quote Schema Introspection Utility.

This script inspects the raw quotes data to find:
- Exact column names
- Sample rows
- Potential bid_size/ask_size columns

Run this BEFORE modifying data loaders to understand what we have.
"""

import sys
from pathlib import Path
import pandas as pd
import re


def inspect_quote_file(file_path: str) -> dict:
    """
    Inspect a quotes file and return schema information.
    
    Args:
        file_path: Path to quotes file (CSV or Parquet)
        
    Returns:
        Dict with schema information
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Inspecting: {file_path}")
    print(f"File size: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    
    # Load file
    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path, nrows=100000)  # Sample for CSV
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print()
    
    # Print exact column names
    print("=" * 60)
    print("EXACT COLUMN NAMES")
    print("=" * 60)
    for i, col in enumerate(df.columns):
        print(f"  {i:2d}. '{col}'")
    print()
    
    # Print sample rows
    print("=" * 60)
    print("SAMPLE ROWS (first 3)")
    print("=" * 60)
    print(df.head(3).to_string())
    print()
    
    # Search for size-related columns
    print("=" * 60)
    print("SEARCHING FOR SIZE COLUMNS")
    print("=" * 60)
    
    size_patterns = [
        (r"bid.*size", "bid + size"),
        (r"ask.*size", "ask + size"),
        (r"bid_size", "bid_size exact"),
        (r"ask_size", "ask_size exact"),
        (r"bidSize", "bidSize camel"),
        (r"askSize", "askSize camel"),
        (r"bidsz", "bidsz"),
        (r"asksz", "asksz"),
        (r"bs$", "bs"),
        (r"as$", "as"),
        (r"bid_sz", "bid_sz"),
        (r"ask_sz", "ask_sz"),
        (r"size", "contains 'size'"),
        (r"sz$", "ends with 'sz'"),
        (r"qty", "qty"),
        (r"quantity", "quantity"),
        (r"volume", "volume"),
    ]
    
    found_size_cols = []
    
    for pattern, desc in size_patterns:
        matches = [col for col in df.columns if re.search(pattern, col, re.IGNORECASE)]
        if matches:
            print(f"\n  Pattern '{desc}' matches:")
            for col in matches:
                non_null = df[col].notna().sum()
                non_null_pct = 100 * non_null / len(df)
                print(f"    - '{col}': {non_null_pct:.1f}% non-null, dtype={df[col].dtype}")
                if col not in found_size_cols:
                    found_size_cols.append(col)
    
    print()
    
    # Summary of potential size columns
    print("=" * 60)
    print("SUMMARY: POTENTIAL SIZE COLUMNS")
    print("=" * 60)
    
    if found_size_cols:
        print("\nFound these potential size-related columns:")
        for col in found_size_cols:
            non_null = df[col].notna().sum()
            non_null_pct = 100 * non_null / len(df)
            sample_vals = df[col].dropna().head(5).tolist()
            print(f"\n  '{col}':")
            print(f"    Non-null: {non_null_pct:.1f}%")
            print(f"    Dtype: {df[col].dtype}")
            print(f"    Sample values: {sample_vals}")
    else:
        print("\n❌ NO SIZE-RELATED COLUMNS FOUND")
        print("   The quotes data does not contain bid_size/ask_size.")
        print("   You need to re-ingest Polygon quotes with size data.")
    
    print()
    
    # Check specifically for bid/ask price columns
    print("=" * 60)
    print("BID/ASK PRICE COLUMNS")
    print("=" * 60)
    
    price_cols = [col for col in df.columns if 'price' in col.lower() or 'bid' in col.lower() or 'ask' in col.lower()]
    for col in price_cols:
        non_null = df[col].notna().sum()
        non_null_pct = 100 * non_null / len(df)
        print(f"  '{col}': {non_null_pct:.1f}% non-null, dtype={df[col].dtype}")
    
    print()
    
    return {
        "file_path": str(file_path),
        "row_count": len(df),
        "columns": list(df.columns),
        "potential_size_cols": found_size_cols,
        "has_bid_size": any("bid" in c.lower() and "size" in c.lower() for c in df.columns),
        "has_ask_size": any("ask" in c.lower() and "size" in c.lower() for c in df.columns),
    }


def main():
    """Inspect quote files."""
    print("=" * 70)
    print("QUOTE SCHEMA INSPECTION")
    print("=" * 70)
    print()
    
    # Default paths
    DATA_DIR = Path(__file__).parent.parent.parent / "Data"
    QUOTES_DIR = DATA_DIR / "quotes"
    
    if not QUOTES_DIR.exists():
        print(f"❌ Quotes directory not found: {QUOTES_DIR}")
        return
    
    # Find quote files
    quote_files = list(QUOTES_DIR.glob("*.parquet"))
    if not quote_files:
        quote_files = list(QUOTES_DIR.glob("*.csv"))
    
    if not quote_files:
        print(f"❌ No quote files found in {QUOTES_DIR}")
        return
    
    print(f"Found {len(quote_files)} quote file(s) in {QUOTES_DIR}")
    print()
    
    # Inspect first file (or 2024 if available)
    target_file = None
    for f in quote_files:
        if "2024" in f.name:
            target_file = f
            break
    
    if target_file is None:
        target_file = quote_files[0]
    
    results = inspect_quote_file(str(target_file))
    
    # Final verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    
    if results["has_bid_size"] and results["has_ask_size"]:
        print("\n✓ Quote data HAS bid_size and ask_size columns")
        print("  Data loader should be updated to use these columns.")
    else:
        print("\n❌ Quote data MISSING bid_size and/or ask_size")
        print("  Cannot compute true microstructure features.")
        print("  Options:")
        print("    1. Re-ingest Polygon quotes with size data")
        print("    2. Use a different data source with order book sizes")
    
    print()


if __name__ == "__main__":
    main()

