#!/usr/bin/env python3
"""
Microstructure Filters for Fake-Out Detection.

Uses Order Flow, Entropy, and Spread dynamics to filter signals.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class MicrostructureFilter:
    """
    Filters ML signals using microstructure features to detect:
    - Liquidity sweeps (fake breakouts)
    - Low-conviction moves (high entropy)
    - Effort/Result divergence
    """
    
    def __init__(
        self,
        flow_cvd_threshold: float = 0.0,  # CVD must align with direction
        entropy_max: float = 1.5,          # High entropy = choppy
        effort_min: float = 0.5,           # Min effort for valid move
        divergence_threshold: float = -1.0  # Flow divergence warning
    ):
        self.flow_cvd_threshold = flow_cvd_threshold
        self.entropy_max = entropy_max
        self.effort_min = effort_min
        self.divergence_threshold = divergence_threshold
    
    def check_flow_alignment(
        self, 
        signal: int, 
        flow_cvd_60: float,
        flow_divergence: float
    ) -> bool:
        """
        Check if order flow supports the signal direction.
        
        SHORT (-1): Requires NEGATIVE flow_cvd_60
        LONG (+1): Requires POSITIVE flow_cvd_60
        
        Also checks flow divergence to catch distribution/accumulation.
        """
        if signal == -1:  # SHORT
            # Flow must be negative (selling pressure)
            if flow_cvd_60 > self.flow_cvd_threshold:
                return False  # Bullish flow, reject short
            
            # Divergence check: Reject if price high but flow already negative (late)
            if flow_divergence < self.divergence_threshold:
                return False  # Already distributed
        
        elif signal == 1:  # LONG
            # Flow must be positive (buying pressure)
            if flow_cvd_60 < self.flow_cvd_threshold:
                return False  # Bearish flow, reject long
            
            # Divergence check: Reject if price low but flow already positive (late)
            if flow_divergence > -self.divergence_threshold:
                return False  # Already accumulated
        
        return True
    
    def check_entropy(self, micro_entropy: float) -> bool:
        """
        Reject if entropy too high (choppy, no clear direction).
        """
        if pd.isna(micro_entropy):
            return True  # Allow if missing
        return micro_entropy < self.entropy_max
    
    def check_effort(self, effort_ratio: float) -> bool:
        """
        Reject if effort_ratio too low (range relative to tick activity).
        Low effort = lots of ticks but no movement = distribution/absorption.
        """
        if pd.isna(effort_ratio):
            return True  # Allow if missing
        return effort_ratio > self.effort_min
    
    def check_wick_structure(
        self,
        signal: int,
        bearish_wick_vol: float,
        bullish_wick_vol: float,
        wick_flow_interaction: float
    ) -> bool:
        """
        Check if wick structure supports the trade.
        
        SHORT: Want bearish_wick_vol (red candle rejection)
        LONG: Want bullish_wick_vol (green candle support)
        """
        if signal == -1:  # SHORT
            # Bearish wick should be negative (red candle with upper wick)
            # And NOT have positive flow absorption
            if not pd.isna(bearish_wick_vol) and bearish_wick_vol >= 0:  # No bearish rejection
                return False
            if not pd.isna(wick_flow_interaction) and wick_flow_interaction > 0:  # Bullish absorption at top
                return False
        
        elif signal == 1:  # LONG
            # Bullish wick should be positive (green candle with lower wick)
            # And NOT have negative flow rejection
            if not pd.isna(bullish_wick_vol) and bullish_wick_vol <= 0:  # No bullish support
                return False
            if not pd.isna(wick_flow_interaction) and wick_flow_interaction < 0:  # Bearish rejection at bottom
                return False
        
        return True
    
    def filter_signal(
        self,
        signal: int,
        features: dict
    ) -> Tuple[bool, str]:
        """
        Apply all filters to a signal.
        
        Returns:
            (pass: bool, reason: str)
        """
        if signal == 0:
            return False, "No signal"
        
        # 1. Flow alignment
        if not self.check_flow_alignment(
            signal,
            features.get('flow_cvd_60', 0),
            features.get('flow_divergence', 0)
        ):
            return False, "Flow misalignment"
        
        # 2. Entropy check
        if not self.check_entropy(features.get('micro_entropy', 1.5)):
            return False, "High entropy (choppy)"
        
        # 3. Effort check
        if not self.check_effort(features.get('effort_ratio', 0)):
            return False, "Low effort (no conviction)"
        
        # 4. Wick structure
        if not self.check_wick_structure(
            signal,
            features.get('bearish_wick_vol', 0),
            features.get('bullish_wick_vol', 0),
            features.get('wick_flow_interaction', 0)
        ):
            return False, "Wick structure conflict"
        
        return True, "Pass"

