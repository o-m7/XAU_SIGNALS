"""
Data Loader Module - XAUUSD Minute Bars + Quotes.

This module handles loading and merging:
- Minute OHLCV data
- Top-of-book quotes with bid/ask prices AND sizes

CRITICAL: bid_size and ask_size are REQUIRED for microstructure features.
If they are missing, the loader raises a hard error.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import warnings


# =============================================================================
# REQUIRED QUOTE COLUMNS
# =============================================================================

REQUIRED_QUOTE_COLS = ["bid_price", "ask_price", "bid_size", "ask_size"]

# Known column name mappings for different data sources
SIZE_COLUMN_MAPPINGS = {
    # Polygon.io variations
    "bid_size": ["bid_size", "bidSize", "bid_sz", "bidsz", "bs"],
    "ask_size": ["ask_size", "askSize", "ask_sz", "asksz", "as"],
    # Price columns
    "bid_price": ["bid_price", "bidPrice", "bid", "bp"],
    "ask_price": ["ask_price", "askPrice", "ask", "ap"],
}


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

def find_column_mapping(df: pd.DataFrame, target: str, candidates: List[str]) -> Optional[str]:
    """
    Find the actual column name in df that matches one of the candidates.
    
    Returns the matching column name or None if not found.
    """
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        # Also try case-insensitive match
        for col in df.columns:
            if col.lower() == candidate.lower():
                return col
    return None


def validate_quotes_schema(
    df: pd.DataFrame,
    require_sizes: bool = True
) -> Tuple[bool, dict, List[str]]:
    """
    Validate that quotes DataFrame has required columns.
    
    Args:
        df: Quotes DataFrame
        require_sizes: If True, fail if bid_size/ask_size missing
        
    Returns:
        (valid: bool, column_mapping: dict, errors: list)
    """
    errors = []
    mapping = {}
    
    # Find bid_price
    bid_price_col = find_column_mapping(df, "bid_price", SIZE_COLUMN_MAPPINGS["bid_price"])
    if bid_price_col:
        mapping["bid_price"] = bid_price_col
    else:
        errors.append("bid_price column not found")
    
    # Find ask_price
    ask_price_col = find_column_mapping(df, "ask_price", SIZE_COLUMN_MAPPINGS["ask_price"])
    if ask_price_col:
        mapping["ask_price"] = ask_price_col
    else:
        errors.append("ask_price column not found")
    
    # Find bid_size
    bid_size_col = find_column_mapping(df, "bid_size", SIZE_COLUMN_MAPPINGS["bid_size"])
    if bid_size_col:
        mapping["bid_size"] = bid_size_col
    elif require_sizes:
        errors.append(
            "bid_size column not found. "
            "Looked for: " + ", ".join(SIZE_COLUMN_MAPPINGS["bid_size"])
        )
    
    # Find ask_size
    ask_size_col = find_column_mapping(df, "ask_size", SIZE_COLUMN_MAPPINGS["ask_size"])
    if ask_size_col:
        mapping["ask_size"] = ask_size_col
    elif require_sizes:
        errors.append(
            "ask_size column not found. "
            "Looked for: " + ", ".join(SIZE_COLUMN_MAPPINGS["ask_size"])
        )
    
    valid = len(errors) == 0
    return valid, mapping, errors


# =============================================================================
# DATA LOADING
# =============================================================================

def load_minute_bars(path: str) -> pd.DataFrame:
    """
    Load minute OHLCV data.
    
    Expected columns: timestamp, open, high, low, close, volume
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Minute bars file not found: {path}")
    
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    
    return df


def load_quotes(
    path: str,
    require_sizes: bool = True
) -> pd.DataFrame:
    """
    Load top-of-book quotes data.
    
    REQUIRED columns (after mapping):
    - bid_price, ask_price
    - bid_size, ask_size (if require_sizes=True)
    
    Args:
        path: Path to quotes file
        require_sizes: If True, raise error if sizes missing
        
    Returns:
        DataFrame with standardized column names
        
    Raises:
        ValueError: If required columns are missing
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Quotes file not found: {path}")
    
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Validate schema
    valid, mapping, errors = validate_quotes_schema(df, require_sizes=require_sizes)
    
    if not valid:
        error_msg = (
            "Quote schema validation FAILED.\n"
            "Errors:\n" + "\n".join(f"  - {e}" for e in errors) + "\n\n"
            "Available columns: " + ", ".join(df.columns) + "\n\n"
        )
        
        if require_sizes and ("bid_size" not in mapping or "ask_size" not in mapping):
            error_msg += (
                "CRITICAL: bid_size/ask_size not present in raw quotes.\n"
                "Cannot compute microstructure features (imbalance, microprice).\n"
                "You must re-ingest Polygon quotes with sizes:\n"
                "  - Use 'sip_timestamp' or 'participant_timestamp'\n"
                "  - Include bid_size, ask_size in the query\n"
                "  - See: https://polygon.io/docs/forex/get_v1_historic_forex__from___to___date\n"
            )
        
        raise ValueError(error_msg)
    
    # Rename columns to standard names
    rename_map = {}
    for standard_name, raw_name in mapping.items():
        if raw_name != standard_name:
            rename_map[raw_name] = standard_name
    
    if rename_map:
        df = df.rename(columns=rename_map)
    
    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
    elif "participant_timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["participant_timestamp"], utc=True)
        df = df.set_index("timestamp")
    
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    
    # Keep only required columns
    cols_to_keep = [c for c in REQUIRED_QUOTE_COLS if c in df.columns]
    df = df[cols_to_keep]
    
    return df


# =============================================================================
# MERGE / ALIGNMENT
# =============================================================================

def align_minute_bars_with_quotes(
    minute_df: pd.DataFrame,
    quotes_df: pd.DataFrame,
    require_sizes: bool = True,
    size_non_null_threshold: float = 0.95
) -> pd.DataFrame:
    """
    Align quotes to minute bars using merge_asof.
    
    Each minute bar gets the most recent quote at or before bar close.
    
    Args:
        minute_df: Minute OHLCV data
        quotes_df: Quotes data with prices and sizes
        require_sizes: If True, require bid_size/ask_size
        size_non_null_threshold: Minimum non-null ratio for sizes
        
    Returns:
        Merged DataFrame
        
    Raises:
        ValueError: If sizes are required but missing or too sparse
    """
    minute_df = minute_df.reset_index()
    quotes_df = quotes_df.reset_index()
    
    # Determine which columns to merge
    quote_cols = ["timestamp", "bid_price", "ask_price"]
    
    if require_sizes:
        if "bid_size" not in quotes_df.columns:
            raise ValueError(
                "bid_size not in quotes data. "
                "Cannot compute microstructure features. "
                "Re-ingest Polygon quotes with size data."
            )
        if "ask_size" not in quotes_df.columns:
            raise ValueError(
                "ask_size not in quotes data. "
                "Cannot compute microstructure features. "
                "Re-ingest Polygon quotes with size data."
            )
        quote_cols.extend(["bid_size", "ask_size"])
    else:
        # Include sizes if available
        if "bid_size" in quotes_df.columns:
            quote_cols.append("bid_size")
        if "ask_size" in quotes_df.columns:
            quote_cols.append("ask_size")
    
    # Merge
    merged = pd.merge_asof(
        minute_df.sort_values("timestamp"),
        quotes_df.sort_values("timestamp")[quote_cols],
        on="timestamp",
        direction="backward"
    )
    
    merged = merged.set_index("timestamp")
    
    # Validate non-null ratios for sizes
    if require_sizes:
        for col in ["bid_size", "ask_size"]:
            if col in merged.columns:
                non_null_ratio = merged[col].notna().mean()
                
                print(f"  {col}: {non_null_ratio*100:.1f}% non-null")
                
                if non_null_ratio < size_non_null_threshold:
                    missing_timestamps = merged[merged[col].isna()].index[:10]
                    warnings.warn(
                        f"{col} is only {non_null_ratio*100:.1f}% non-null "
                        f"(threshold: {size_non_null_threshold*100:.0f}%). "
                        f"First 10 missing timestamps: {list(missing_timestamps)}"
                    )
    
    # Compute derived columns
    merged["mid"] = (merged["bid_price"] + merged["ask_price"]) / 2
    merged["spread"] = merged["ask_price"] - merged["bid_price"]
    merged["spread_pct"] = merged["spread"] / merged["mid"]
    
    # Drop rows without prices
    merged = merged.dropna(subset=["bid_price", "ask_price"])
    
    return merged


def get_combined_dataset(
    minute_path: str,
    quotes_path: str,
    require_sizes: bool = True
) -> pd.DataFrame:
    """
    Load and combine minute bars with quotes.
    
    Args:
        minute_path: Path to minute OHLCV file
        quotes_path: Path to quotes file
        require_sizes: If True, require bid_size/ask_size
        
    Returns:
        Combined DataFrame with OHLCV + quotes
    """
    print(f"Loading minute bars: {minute_path}")
    minute_df = load_minute_bars(minute_path)
    print(f"  Rows: {len(minute_df):,}")
    
    print(f"Loading quotes: {quotes_path}")
    quotes_df = load_quotes(quotes_path, require_sizes=require_sizes)
    print(f"  Rows: {len(quotes_df):,}")
    print(f"  Columns: {list(quotes_df.columns)}")
    
    print("Aligning quotes to minute bars...")
    combined = align_minute_bars_with_quotes(
        minute_df, quotes_df, require_sizes=require_sizes
    )
    print(f"  Combined rows: {len(combined):,}")
    
    return combined


# =============================================================================
# MULTI-YEAR LOADING
# =============================================================================

def load_multi_year_data(
    minute_dir: str,
    quotes_dir: str,
    years: List[int],
    require_sizes: bool = True
) -> pd.DataFrame:
    """
    Load and combine data from multiple years.
    
    Args:
        minute_dir: Directory with minute OHLCV files
        quotes_dir: Directory with quotes files
        years: List of years to load
        require_sizes: If True, require bid_size/ask_size
        
    Returns:
        Combined DataFrame
    """
    minute_dir = Path(minute_dir)
    quotes_dir = Path(quotes_dir)
    
    dfs = []
    
    for year in years:
        minute_path = minute_dir / f"XAUUSD_minute_{year}.parquet"
        quotes_path = quotes_dir / f"XAUUSD_quotes_{year}.parquet"
        
        if not minute_path.exists() or not quotes_path.exists():
            print(f"  Skipping {year}: files not found")
            continue
        
        try:
            df = get_combined_dataset(
                str(minute_path),
                str(quotes_path),
                require_sizes=require_sizes
            )
            dfs.append(df)
            print(f"  {year}: {len(df):,} rows")
        except ValueError as e:
            print(f"  {year}: ERROR - {e}")
            raise
    
    if not dfs:
        raise ValueError("No data loaded")
    
    result = pd.concat(dfs, axis=0)
    result = result.sort_index()
    result = result[~result.index.duplicated(keep="first")]
    
    return result
