"""
Model 4 Configuration
Trend-Following with Entry Timing
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Model4Config:
    # Timeframe
    base_timeframe: str = "5T"

    # Trend filter parameters
    ema_fast: int = 20
    ema_slow: int = 50
    adx_threshold: float = 15.0

    # Label parameters
    horizon_bars: int = 12  # 12 bars @ 5T = 60 minutes
    threshold_atr_mult: float = 0.5  # Move must exceed 0.5 * ATR

    # Model parameters
    model_threshold: float = 0.55

    # Execution filters
    min_adx: float = 15.0
    max_spread_pct: float = 0.00015  # 1.5 bps
    cooldown_bars: int = 15

    # Session filters (hours in UTC)
    london_start: int = 7
    london_end: int = 16
    ny_start: int = 13
    ny_end: int = 21

    # Features to use
    feature_columns: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.feature_columns:
            self.feature_columns = [
                # Volatility
                'atr_ratio',
                'range_compression',
                'realized_vol_zscore',

                # Price position
                'price_in_session_range',
                'dist_from_session_high_atr',
                'dist_from_session_low_atr',

                # Momentum
                'returns_5bar_zscore',
                'returns_12bar_zscore',

                # Spread dynamics
                'spread_pct',
                'spread_zscore',

                # Quote activity
                'quote_rate_zscore',

                # Context
                'adx',
                'hour_sin',
                'hour_cos',
                'is_overlap_session',
            ]
