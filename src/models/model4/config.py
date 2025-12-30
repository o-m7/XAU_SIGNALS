"""
Model 4 Configuration: VWAP Mean Reversion

Strategy: Trade mean reversion to session VWAP when price is stretched.
ML predicts: "Given price is stretched, will it revert before hitting stop?"
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Model4Config:
    """Configuration for VWAP Mean Reversion Model."""

    # Timeframe
    base_timeframe: str = "5T"

    # VWAP parameters
    vwap_session_hours: int = 8  # Rolling window for VWAP

    # Setup detection
    entry_zscore_threshold: float = 2.0  # Distance from VWAP in ATR units

    # Regime filter
    max_adx: float = 25.0  # Only trade when ADX below this
    min_atr_percentile: float = 20.0  # Minimum volatility
    max_atr_percentile: float = 80.0  # Maximum volatility

    # ML model
    model_threshold: float = 0.50  # Minimum P(reversion) to trade

    # Risk management
    stop_atr_mult: float = 1.5  # Stop loss distance
    target_type: str = "vwap"  # "vwap" or "half_reversion"
    max_bars_in_trade: int = 30  # Time stop

    # Execution filters
    max_spread_pct: float = 0.00015  # 1.5 bps
    cooldown_bars: int = 6  # 30 min on 5T

    # Session (UTC hours)
    london_start: int = 7
    london_end: int = 16
    ny_start: int = 13
    ny_end: int = 21

    # Features
    feature_columns: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.feature_columns = [
            # Distance metrics
            'vwap_zscore',           # How stretched from VWAP
            'vwap_zscore_velocity',  # Rate of stretching
            'price_vs_session_high', # Position in session range
            'price_vs_session_low',

            # Regime context
            'adx',
            'atr_percentile',
            'range_compression',     # Recent range vs average

            # Momentum exhaustion signals
            'rsi_14',
            'rsi_divergence',        # Price vs RSI divergence
            'bars_since_extreme',    # How long at stretched level

            # Spread/activity
            'spread_zscore',
            'quote_rate_zscore',

            # Time context
            'hour_sin',
            'hour_cos',
        ]
