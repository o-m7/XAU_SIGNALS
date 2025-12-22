#!/usr/bin/env python3
"""
Build and Save Features Script.

This script:
1. Loads minute OHLCV + quotes data
2. Builds the full feature matrix
3. Saves to parquet

Usage:
    python scripts/build_features.py \
        --minute_path Data/ohlcv_minute/XAUUSD_minute_2024.parquet \
        --quotes_path Data/quotes/XAUUSD_quotes_2024.parquet \
        --start 2024-01-01 \
        --end 2024-12-31 \
        --out data/features/xauusd_features_2024.parquet

REQUIRES bid_size and ask_size in quotes data.
If missing, raises an error with instructions to re-ingest.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import load_minute_bars, load_quotes, align_minute_bars_with_quotes
from feature_engineering import build_feature_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and save XAUUSD feature matrix"
    )
    parser.add_argument(
        "--minute_path",
        type=str,
        required=True,
        help="Path to minute OHLCV parquet file"
    )
    parser.add_argument(
        "--quotes_path",
        type=str,
        required=True,
        help="Path to quotes parquet file"
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output parquet file path"
    )
    parser.add_argument(
        "--require-sizes",
        action="store_true",
        default=True,
        help="Require bid_size/ask_size (default: True)"
    )
    parser.add_argument(
        "--no-require-sizes",
        action="store_false",
        dest="require_sizes",
        help="Don't require bid_size/ask_size (use momentum fallback)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("BUILD FEATURES")
    print("=" * 70)
    print()
    
    # Paths
    minute_path = Path(args.minute_path)
    quotes_path = Path(args.quotes_path)
    out_path = Path(args.out)
    
    print(f"Minute data: {minute_path}")
    print(f"Quotes data: {quotes_path}")
    print(f"Output:      {out_path}")
    print(f"Date range:  {args.start or 'all'} to {args.end or 'all'}")
    print(f"Require sizes: {args.require_sizes}")
    print()
    
    # Validate inputs
    if not minute_path.exists():
        print(f"❌ Minute file not found: {minute_path}")
        return 1
    
    if not quotes_path.exists():
        print(f"❌ Quotes file not found: {quotes_path}")
        return 1
    
    # Load data
    print("Loading data...")
    try:
        minute_df = load_minute_bars(str(minute_path))
        print(f"  Minute bars: {len(minute_df):,} rows")
        
        quotes_df = load_quotes(str(quotes_path), require_sizes=args.require_sizes)
        print(f"  Quotes: {len(quotes_df):,} rows")
        print(f"  Quotes columns: {list(quotes_df.columns)}")
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        return 1
    
    # Merge
    print("\nMerging data...")
    try:
        df = align_minute_bars_with_quotes(
            minute_df, quotes_df, require_sizes=args.require_sizes
        )
        print(f"  Merged: {len(df):,} rows")
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        return 1
    
    # Filter date range
    if args.start:
        start_dt = pd.Timestamp(args.start, tz='UTC')
        df = df[df.index >= start_dt]
        print(f"  After start filter: {len(df):,} rows")
    
    if args.end:
        end_dt = pd.Timestamp(args.end, tz='UTC') + pd.Timedelta(days=1)
        df = df[df.index < end_dt]
        print(f"  After end filter: {len(df):,} rows")
    
    # Build features
    print("\nBuilding features...")
    try:
        df = build_feature_matrix(df, require_sizes=args.require_sizes)
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        return 1
    
    # Save
    print(f"\nSaving to {out_path}...")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"  ✓ Saved: {out_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nRow count: {len(df):,}")
    print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"\nColumns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")
    
    print("\nNaN percentages for key features:")
    key_features = ["mid", "spread_pct", "sigma_slope", "imbalance", "microprice", "micro_dislocation"]
    for col in key_features:
        if col in df.columns:
            nan_pct = df[col].isna().mean() * 100
            status = "✓" if nan_pct < 5 else "⚠" if nan_pct < 20 else "❌"
            print(f"  {status} {col}: {nan_pct:.1f}% NaN")
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

