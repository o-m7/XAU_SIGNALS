"""
Model 6: Order Flow / Microstructure-Based Strategy

Uses bid_size and ask_size from quotes data to detect institutional order flow.
Strategy: BUY when heavy bid pressure, SELL when heavy ask pressure, VWAP confirmation.
"""

from .features import calculate_order_features
from .labels import create_labels

__all__ = ['calculate_order_features', 'create_labels']
