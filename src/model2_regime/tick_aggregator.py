"""
Tick-Level Data Aggregator for Model #2 (Regime Classifier)

This module processes tick-by-tick data to create advanced microstructure features:
- Order Flow Imbalance (OFI)
- Trade arrival rates
- Bid-ask dynamics
- Market depth metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TickAggregator:
    """
    Aggregates tick data into feature-rich bars for regime detection.
    
    Features computed:
    - Order flow imbalance (OFI)
    - Trade arrival rate
    - Bid-ask bounce frequency
    - Volume-weighted metrics
    """
    
    def __init__(self, bar_interval_seconds: int = 300):
        """
        Initialize tick aggregator.
        
        Args:
            bar_interval_seconds: Interval for aggregation (default 300 = 5 minutes)
        """
        self.bar_interval = bar_interval_seconds
        self.current_bar = None
        self.bar_start_time = None
        
    def process_quote_tick(
        self, 
        timestamp: datetime, 
        bid_price: float, 
        ask_price: float,
        bid_size: float = None,
        ask_size: float = None
    ) -> Optional[Dict]:
        """
        Process a single quote tick.
        
        Args:
            timestamp: Quote timestamp
            bid_price: Bid price
            ask_price: Ask price
            bid_size: Bid size (optional)
            ask_size: Ask size (optional)
            
        Returns:
            Completed bar dict if interval finished, else None
        """
        # Initialize new bar if needed
        if self.current_bar is None:
            self._init_new_bar(timestamp)
        
        # Check if bar is complete
        if (timestamp - self.bar_start_time).total_seconds() >= self.bar_interval:
            completed_bar = self._finalize_bar()
            self._init_new_bar(timestamp)
        else:
            completed_bar = None
        
        # Update current bar
        self.current_bar['quote_updates'] += 1
        self.current_bar['bid_prices'].append(bid_price)
        self.current_bar['ask_prices'].append(ask_price)
        
        if bid_size is not None and ask_size is not None:
            self.current_bar['bid_sizes'].append(bid_size)
            self.current_bar['ask_sizes'].append(ask_size)
            
            # Compute order flow imbalance
            if len(self.current_bar['bid_sizes']) > 1:
                prev_bid_size = self.current_bar['bid_sizes'][-2]
                prev_ask_size = self.current_bar['ask_sizes'][-2]
                
                # OFI = (ΔBid_size - ΔAsk_size)
                delta_bid = bid_size - prev_bid_size
                delta_ask = ask_size - prev_ask_size
                ofi = delta_bid - delta_ask
                
                self.current_bar['ofi_cumulative'] += ofi
        
        # Update spread metrics
        spread = ask_price - bid_price
        mid = (bid_price + ask_price) / 2
        
        self.current_bar['spreads'].append(spread)
        self.current_bar['mids'].append(mid)
        
        return completed_bar
    
    def process_trade_tick(
        self,
        timestamp: datetime,
        price: float,
        size: float,
        side: Optional[str] = None  # 'buy', 'sell', or None
    ) -> Optional[Dict]:
        """
        Process a single trade tick.
        
        Args:
            timestamp: Trade timestamp
            price: Trade price
            size: Trade size
            side: Trade side ('buy' or 'sell' if known)
            
        Returns:
            Completed bar dict if interval finished, else None
        """
        # Initialize new bar if needed
        if self.current_bar is None:
            self._init_new_bar(timestamp)
        
        # Check if bar is complete
        if (timestamp - self.bar_start_time).total_seconds() >= self.bar_interval:
            completed_bar = self._finalize_bar()
            self._init_new_bar(timestamp)
        else:
            completed_bar = None
        
        # Update current bar
        self.current_bar['trades'] += 1
        self.current_bar['trade_prices'].append(price)
        self.current_bar['trade_sizes'].append(size)
        self.current_bar['volume'] += size
        
        # Track aggressive vs passive (if side is known)
        if side == 'buy':
            self.current_bar['buy_volume'] += size
            self.current_bar['aggressive_buys'] += 1
        elif side == 'sell':
            self.current_bar['sell_volume'] += size
            self.current_bar['aggressive_sells'] += 1
        
        return completed_bar
    
    def _init_new_bar(self, timestamp: datetime):
        """Initialize a new bar."""
        self.bar_start_time = timestamp
        self.current_bar = {
            'timestamp': timestamp,
            'bar_start': timestamp,
            
            # Quote data
            'quote_updates': 0,
            'bid_prices': [],
            'ask_prices': [],
            'bid_sizes': [],
            'ask_sizes': [],
            'spreads': [],
            'mids': [],
            
            # Trade data
            'trades': 0,
            'trade_prices': [],
            'trade_sizes': [],
            'volume': 0.0,
            'buy_volume': 0.0,
            'sell_volume': 0.0,
            'aggressive_buys': 0,
            'aggressive_sells': 0,
            
            # Order flow
            'ofi_cumulative': 0.0,
        }
    
    def _finalize_bar(self) -> Dict:
        """
        Compute all features for completed bar.
        
        Returns:
            Dict with computed features
        """
        bar = self.current_bar
        
        # Basic OHLCV from trades
        if bar['trade_prices']:
            open_price = bar['trade_prices'][0]
            close_price = bar['trade_prices'][-1]
            high_price = max(bar['trade_prices'])
            low_price = min(bar['trade_prices'])
            vwap = np.average(bar['trade_prices'], weights=bar['trade_sizes'])
        else:
            # Fallback to mid prices if no trades
            open_price = bar['mids'][0] if bar['mids'] else np.nan
            close_price = bar['mids'][-1] if bar['mids'] else np.nan
            high_price = max(bar['mids']) if bar['mids'] else np.nan
            low_price = min(bar['mids']) if bar['mids'] else np.nan
            vwap = np.mean(bar['mids']) if bar['mids'] else np.nan
        
        # Quote metrics
        avg_spread = np.mean(bar['spreads']) if bar['spreads'] else np.nan
        spread_std = np.std(bar['spreads']) if len(bar['spreads']) > 1 else 0.0
        avg_mid = np.mean(bar['mids']) if bar['mids'] else np.nan
        
        # Microstructure metrics
        trade_arrival_rate = bar['trades'] / self.bar_interval if self.bar_interval > 0 else 0
        quote_update_rate = bar['quote_updates'] / self.bar_interval if self.bar_interval > 0 else 0
        
        # Order flow imbalance (OFI)
        ofi = bar['ofi_cumulative']
        
        # Signed volume (buy - sell)
        signed_volume = bar['buy_volume'] - bar['sell_volume']
        
        # Aggressive order ratio
        total_aggressive = bar['aggressive_buys'] + bar['aggressive_sells']
        if total_aggressive > 0:
            buy_pressure = bar['aggressive_buys'] / total_aggressive
        else:
            buy_pressure = 0.5  # neutral
        
        # Depth imbalance (if available)
        if bar['bid_sizes'] and bar['ask_sizes']:
            avg_bid_size = np.mean(bar['bid_sizes'])
            avg_ask_size = np.mean(bar['ask_sizes'])
            depth_imbalance = (avg_bid_size - avg_ask_size) / (avg_bid_size + avg_ask_size + 1e-8)
        else:
            avg_bid_size = np.nan
            avg_ask_size = np.nan
            depth_imbalance = 0.0
        
        # Bid-ask bounce (how often quotes change)
        if len(bar['bid_prices']) > 1:
            bid_changes = sum(1 for i in range(1, len(bar['bid_prices'])) 
                            if bar['bid_prices'][i] != bar['bid_prices'][i-1])
            ask_changes = sum(1 for i in range(1, len(bar['ask_prices'])) 
                            if bar['ask_prices'][i] != bar['ask_prices'][i-1])
            quote_stability = 1.0 - (bid_changes + ask_changes) / (2 * len(bar['bid_prices']))
        else:
            quote_stability = 1.0
        
        return {
            'timestamp': bar['bar_start'],
            'bar_end': bar['timestamp'],
            
            # Price data
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'vwap': vwap,
            'volume': bar['volume'],
            
            # Spread metrics
            'avg_spread': avg_spread,
            'spread_std': spread_std,
            'avg_mid': avg_mid,
            
            # Order flow
            'ofi': ofi,
            'signed_volume': signed_volume,
            'buy_volume': bar['buy_volume'],
            'sell_volume': bar['sell_volume'],
            'buy_pressure': buy_pressure,
            
            # Depth
            'avg_bid_size': avg_bid_size,
            'avg_ask_size': avg_ask_size,
            'depth_imbalance': depth_imbalance,
            
            # Activity metrics
            'trades': bar['trades'],
            'trade_arrival_rate': trade_arrival_rate,
            'quote_updates': bar['quote_updates'],
            'quote_update_rate': quote_update_rate,
            'quote_stability': quote_stability,
        }


def aggregate_from_minute_bars(
    minute_bars: pd.DataFrame,
    quotes: pd.DataFrame,
    interval_minutes: int = 5
) -> pd.DataFrame:
    """
    Aggregate existing minute bars and quotes into 5-minute bars with microstructure features.
    
    This is a simplified version for when we don't have tick data.
    
    Args:
        minute_bars: DataFrame with 1-minute OHLCV data
        quotes: DataFrame with quote data
        interval_minutes: Aggregation interval (default 5)
        
    Returns:
        DataFrame with aggregated bars and features
    """
    logger.info(f"Aggregating to {interval_minutes}-minute bars...")
    
    # Resample minute bars to 5-minute
    bars_resampled = minute_bars.resample(f'{interval_minutes}T').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    })
    
    # Add VWAP if not present
    if 'vwap' in minute_bars.columns:
        bars_resampled['vwap'] = minute_bars['vwap'].resample(f'{interval_minutes}T').mean()
    else:
        # Approximate VWAP
        bars_resampled['vwap'] = (
            (minute_bars['close'] * minute_bars['volume']).resample(f'{interval_minutes}T').sum() /
            minute_bars['volume'].resample(f'{interval_minutes}T').sum()
        )
    
    # Resample quotes
    quotes_resampled = quotes.resample(f'{interval_minutes}T').agg({
        'bid_price': ['first', 'last', 'mean'],
        'ask_price': ['first', 'last', 'mean'],
    })
    quotes_resampled.columns = ['_'.join(col).strip() for col in quotes_resampled.columns.values]
    
    # Compute spread metrics
    quotes_resampled['avg_spread'] = quotes_resampled['ask_price_mean'] - quotes_resampled['bid_price_mean']
    quotes_resampled['avg_mid'] = (quotes_resampled['bid_price_mean'] + quotes_resampled['ask_price_mean']) / 2
    
    # Merge
    df = pd.concat([bars_resampled, quotes_resampled], axis=1)
    df = df.dropna()
    
    logger.info(f"Aggregated to {len(df):,} bars")
    
    return df

