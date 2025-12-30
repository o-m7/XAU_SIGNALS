#!/usr/bin/env python3
"""Debug script to check Model 4 data and feature building."""
import sys
import pandas as pd
import numpy as np

def main():
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/features/xauusd_features_2020_2025.parquet"
    
    print(f"Loading: {data_path}")
    df = pd.read_parquet(data_path)
    
    print(f"\n=== DATA INFO ===")
    print(f"Shape: {df.shape}")
    print(f"Index type: {type(df.index)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    print(f"\n=== COLUMNS ===")
    for col in sorted(df.columns):
        print(f"  {col}: {df[col].dtype}, NaN={df[col].isna().sum()}")
    
    # Check OHLCV
    ohlcv = ['open', 'high', 'low', 'close', 'volume']
    has_ohlcv = all(c in df.columns for c in ohlcv)
    print(f"\n=== OHLCV CHECK ===")
    print(f"Has OHLCV: {has_ohlcv}")
    
    if has_ohlcv:
        print(f"  open: min={df['open'].min():.2f}, max={df['open'].max():.2f}")
        print(f"  close: min={df['close'].min():.2f}, max={df['close'].max():.2f}")
        print(f"  volume: min={df['volume'].min()}, max={df['volume'].max()}, zeros={( df['volume']==0).sum()}")
    
    # Check timeframe
    if len(df) > 1:
        time_diff = (df.index[1] - df.index[0]).total_seconds()
        print(f"\n=== TIMEFRAME ===")
        print(f"Time between first 2 rows: {time_diff} seconds ({time_diff/60:.1f} minutes)")
    
    # Try building VWAP features
    if has_ohlcv:
        print(f"\n=== BUILDING VWAP FEATURES ===")
        from src.models.model4.regime import calculate_atr
        from src.models.model4.vwap import calculate_session_vwap, calculate_vwap_zscore
        
        df_test = df.copy()
        
        # ATR
        df_test = calculate_atr(df_test, period=14)
        print(f"ATR_14: min={df_test['atr_14'].min():.4f}, max={df_test['atr_14'].max():.4f}, mean={df_test['atr_14'].mean():.4f}")
        
        # VWAP  
        df_test = calculate_session_vwap(df_test, session_hours=8)
        print(f"VWAP: min={df_test['vwap'].min():.2f}, max={df_test['vwap'].max():.2f}")
        print(f"VWAP distance: min={df_test['vwap_distance'].min():.4f}, max={df_test['vwap_distance'].max():.4f}")
        
        # Z-score
        df_test = calculate_vwap_zscore(df_test)
        zscore = df_test['vwap_zscore'].dropna()
        
        print(f"\n=== VWAP Z-SCORE ===")
        print(f"Count: {len(zscore):,}")
        print(f"NaN count: {df_test['vwap_zscore'].isna().sum():,}")
        
        if len(zscore) > 0:
            print(f"Mean: {zscore.mean():.4f}")
            print(f"Std: {zscore.std():.4f}")
            print(f"Min/Max: {zscore.min():.4f} / {zscore.max():.4f}")
            print(f"Percentiles (5/25/50/75/95):")
            print(f"  {zscore.quantile(0.05):.3f} / {zscore.quantile(0.25):.3f} / "
                  f"{zscore.quantile(0.50):.3f} / {zscore.quantile(0.75):.3f} / "
                  f"{zscore.quantile(0.95):.3f}")
            
            print(f"\n=== POTENTIAL SETUPS ===")
            for thresh in [0.5, 1.0, 1.5, 2.0]:
                n = (zscore.abs() >= thresh).sum()
                pct = n / len(zscore) * 100
                print(f"  |z| >= {thresh}: {n:,} ({pct:.1f}%)")
        else:
            print("ERROR: No valid z-scores!")

if __name__ == "__main__":
    main()
