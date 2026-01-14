"""
Model 5 Configuration

Statistical mean reversion on 15M XAUUSD.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Model5Config:
    # === TIMEFRAME ===
    timeframe: str = "15T"
    max_holding_bars: int = 15  # Maximum 15 bars = 3.75 hours
    
    # === SIGNAL PARAMETERS ===
    # Z-score threshold for entry
    # Based on: P(reversal | |z| > threshold) should be > 55%
    zscore_entry_threshold: float = 2.0
    zscore_lookback: int = 20  # Bars for mean/std calculation
    
    # Variance ratio threshold
    # VR < 1 indicates mean reversion
    max_variance_ratio: float = 0.95  # Only trade when VR suggests mean reversion
    
    # === FILTERS ===
    # Spread filter (avoid illiquid periods)
    max_spread_percentile: float = 80  # Don't trade when spread is in top 20%
    spread_lookback: int = 100  # Bars for spread percentile
    
    # Volatility filter
    min_atr_percentile: float = 20  # Avoid dead markets
    max_atr_percentile: float = 90  # Avoid extreme volatility
    
    # Session filter (UTC hours)
    # Only trade during liquid sessions
    active_hours: List[int] = field(default_factory=lambda: [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    
    # === EXIT PARAMETERS ===
    # Target: reversion to mean
    target_zscore: float = 0.0  # Exit when z-score returns to 0
    
    # Stop loss
    stop_atr_multiple: float = 1.5  # Stop at 1.5 ATR from entry
    
    # Time stop
    max_bars_in_trade: int = 15  # Force exit after 15 bars
    
    # === RISK ===
    min_rr_ratio: float = 0.8  # Minimum reward:risk ratio
    cooldown_bars: int = 2  # Bars between trades
    
    # === FEATURES ===
    feature_columns: List[str] = field(default_factory=lambda: [
        # Price statistics
        'returns_1',
        'returns_2',
        'zscore_20',
        'percentile_20',
        
        # Volatility
        'atr_14',
        'parkinson_vol_20',
        'returns_std_20',
        
        # Mean reversion indicators
        'variance_ratio_2',
        'variance_ratio_4',
        'autocorr_1',
        
        # Candle structure
        'body_ratio',
        'close_location',
        
        # Distribution
        'returns_skew_20',
        'returns_kurtosis_20',
        
        # From quotes
        'spread_mean_bps',
        'spread_zscore',
        'quote_imbalance',
        
        # From second aggregates
        'intrabar_rv',
        'intrabar_jump',
        'intrabar_reversals',
        
        # Time
        'hour_sin',
        'hour_cos',
        'is_overlap',
    ])

