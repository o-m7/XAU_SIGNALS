"""
signals.py - Signal Generation

Pure statistical signal generation.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .config import Model5Config


class Signal(Enum):
    LONG = "LONG"
    NONE = "NONE"


@dataclass
class SignalResult:
    signal: Signal
    zscore: float
    entry_price: float
    stop_price: float
    target_price: float
    confidence: float
    reason: str
    timestamp: datetime = None


class Model5SignalEngine:
    """
    Statistical mean reversion signal generator.
    
    LONG when:
    - zscore < -threshold (price below mean)
    - variance_ratio < 1 (confirms mean reversion regime)
    - spread_percentile < max (not illiquid)
    - is_active == 1 (during trading hours)
    """
    
    def __init__(self, config: Model5Config = None):
        self.config = config or Model5Config()
        self.last_trade_bar = -999
        self.bar_count = 0
    
    def generate_signal(self, row: pd.Series) -> SignalResult:
        """
        Generate signal from current bar features.
        """
        self.bar_count += 1
        timestamp = row.name if hasattr(row, 'name') else datetime.now()
        
        # Extract features
        zscore = row.get('zscore_20', 0)
        vr = row.get('variance_ratio_2', 1)
        spread_pct = row.get('spread_percentile', 50)
        atr_pct = row.get('atr_percentile', 50)
        is_active = row.get('is_active', 0)
        close = row.get('close', 0)
        atr = row.get('atr_14', 1)
        
        # Default no signal
        def no_signal(reason: str) -> SignalResult:
            return SignalResult(
                signal=Signal.NONE,
                zscore=zscore,
                entry_price=0,
                stop_price=0,
                target_price=0,
                confidence=0,
                reason=reason,
                timestamp=timestamp
            )
        
        # ========== FILTERS ==========
        
        # Cooldown
        if self.bar_count - self.last_trade_bar < self.config.cooldown_bars:
            return no_signal("Cooldown")
        
        # Session filter
        if not is_active:
            return no_signal("Outside active hours")
        
        # Spread filter
        if spread_pct > self.config.max_spread_percentile:
            return no_signal(f"Spread percentile {spread_pct:.0f} > {self.config.max_spread_percentile}")
        
        # Volatility filter
        if atr_pct < self.config.min_atr_percentile:
            return no_signal(f"ATR percentile {atr_pct:.0f} too low")
        if atr_pct > self.config.max_atr_percentile:
            return no_signal(f"ATR percentile {atr_pct:.0f} too high")
        
        # Variance ratio filter (confirm mean reversion regime)
        if vr > self.config.max_variance_ratio:
            return no_signal(f"Variance ratio {vr:.3f} > {self.config.max_variance_ratio}")
        
        # Z-score validity
        if np.isnan(zscore):
            return no_signal("Invalid zscore")
        
        # ========== SIGNAL GENERATION ==========
        
        threshold = self.config.zscore_entry_threshold
        
        # LONG only: when zscore < -threshold (oversold)
        if zscore < -threshold:
            entry_price = close
            stop_price = entry_price - (self.config.stop_atr_multiple * atr)
            target_price = entry_price + (abs(zscore) * 0.5 * atr)  # Partial reversion
            
            # R:R check
            risk = entry_price - stop_price
            reward = target_price - entry_price
            rr = reward / risk if risk > 0 else 0
            
            if rr < self.config.min_rr_ratio:
                return no_signal(f"R:R {rr:.2f} < {self.config.min_rr_ratio}")
            
            # Confidence based on z-score magnitude and variance ratio
            confidence = min(1.0, (abs(zscore) - threshold) / 2.0 + (1 - vr))
            
            self.last_trade_bar = self.bar_count
            
            return SignalResult(
                signal=Signal.LONG,
                zscore=zscore,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                confidence=confidence,
                reason=f"LONG: z={zscore:.2f}, VR={vr:.3f}",
                timestamp=timestamp
            )
        
        return no_signal(f"Z-score {zscore:.2f} within threshold")
    
    def reset(self):
        self.last_trade_bar = -999
        self.bar_count = 0

