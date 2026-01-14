import pandas as pd
import numpy as np

def calculate_order_imbalance(row):
    bid_size = row.get('bid_size', 0)
    ask_size = row.get('ask_size', 0)
    if bid_size + ask_size > 0:
        return (bid_size - ask_size) / (bid_size + ask_size)
    return 0.0

def calculate_mid_price_momentum(row, lookback_minutes=1):
    mid = row.get('mid', 0)
    mid_prev = row.get(f'mid_{lookback_minutes}m_ago', mid)
    if mid > 0 and mid_prev > 0:
        return (mid - mid_prev) / mid_prev
    return 0.0

def calculate_vwap(row):
    return row.get('vwap', row.get('mid', 0))

def calculate_vwap_deviation(row):
    mid = row.get('mid', 0)
    vwap = calculate_vwap(row)
    if vwap > 0:
        return (mid - vwap) / vwap
    return 0.0

def calculate_order_features(df):
    df = df.copy()
    df['mid_1m_ago'] = df['mid'].shift(1)
    df['order_imbalance'] = df.apply(calculate_order_imbalance, axis=1)
    df['mid_price_momentum_1m'] = df.apply(calculate_mid_price_momentum, axis=1)
    df['vwap'] = df['mid']
    df['vwap_deviation'] = 0.0
    return df
