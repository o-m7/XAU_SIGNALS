#!/usr/bin/env python3
"""
Dynamic Risk Management System.

Adjusts position size and stops based on:
- Regime
- Volatility
- Account constraints (funded account rules)
"""

import numpy as np
import pandas as pd
from typing import Tuple


class RiskManager:
    """
    Handles position sizing and risk management for funded accounts.
    """
    
    def __init__(
        self,
        account_balance: float = 100000.0,
        max_risk_per_trade: float = 0.01,      # 1% max risk per trade
        max_daily_dd: float = 0.03,             # 3% daily DD limit
        max_total_dd: float = 0.06,             # 6% total DD limit
        regime_multipliers: dict = None
    ):
        self.account_balance = account_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.max_daily_dd = max_daily_dd
        self.max_total_dd = max_total_dd
        
        # Risk scaling by regime
        self.regime_multipliers = regime_multipliers or {
            'TRENDING': 1.0,      # Full size
            'RANGING': 0.7,       # 70% size
            'VOLATILE': 0.3       # 30% size (or skip)
        }
        
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.peak_balance = account_balance
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        regime: str,
        volatility_multiplier: float = 1.0
    ) -> float:
        """
        Calculate position size in ounces/contracts.
        
        Formula:
        Position Size = (Account * Risk%) / (Entry - Stop) * Regime Multiplier * Vol Multiplier
        """
        risk_amount = self.account_balance * self.max_risk_per_trade
        stop_distance = abs(entry_price - stop_loss_price)
        
        if stop_distance == 0:
            return 0.0
        
        # Base position size
        position_size = risk_amount / stop_distance
        
        # Apply regime multiplier
        regime_mult = self.regime_multipliers.get(regime, 0.5)
        position_size *= regime_mult
        
        # Apply volatility multiplier (reduce size in high vol)
        position_size *= volatility_multiplier
        
        return position_size
    
    def check_drawdown_limits(self) -> Tuple[bool, str]:
        """
        Check if within drawdown limits.
        
        Returns:
            (can_trade: bool, reason: str)
        """
        # Check daily DD
        daily_dd_pct = abs(min(0, self.daily_pnl)) / self.account_balance
        if daily_dd_pct >= self.max_daily_dd:
            return False, f"Daily DD limit hit: {daily_dd_pct:.2%}"
        
        # Check total DD
        current_balance = self.account_balance + self.total_pnl
        dd_from_peak = (self.peak_balance - current_balance) / self.peak_balance
        
        if dd_from_peak >= self.max_total_dd:
            return False, f"Total DD limit hit: {dd_from_peak:.2%}"
        
        return True, "OK"
    
    def update_pnl(self, trade_pnl: float, new_day: bool = False):
        """Update P&L tracking."""
        if new_day:
            self.daily_pnl = 0.0
        
        self.daily_pnl += trade_pnl
        self.total_pnl += trade_pnl
        
        current_balance = self.account_balance + self.total_pnl
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
    
    def get_stop_loss(
        self,
        entry_price: float,
        direction: int,
        atr: float,
        regime: str,
        sl_mult: float = 1.0
    ) -> float:
        """
        Calculate stop loss price.
        
        Uses ATR-scaled stops with regime adjustments:
        - TRENDING: Wider stops (1.5x ATR)
        - RANGING: Tighter stops (1.0x ATR)
        - VOLATILE: Skip or very tight stops (0.8x ATR)
        """
        regime_sl_mult = {
            'TRENDING': 1.5,
            'RANGING': 1.0,
            'VOLATILE': 0.8
        }
        
        sl_distance = atr * sl_mult * regime_sl_mult.get(regime, 1.0)
        
        if direction == 1:  # LONG
            return entry_price - sl_distance
        else:  # SHORT
            return entry_price + sl_distance
    
    def get_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        direction: int,
        regime: str,
        rr_ratio: float = 1.5
    ) -> float:
        """
        Calculate take profit price.
        
        Uses R:R ratio adjusted by regime:
        - TRENDING: Higher R:R (2:1 or 3:1)
        - RANGING: Lower R:R (1:1 or 1.5:1)
        """
        regime_rr = {
            'TRENDING': 2.0,
            'RANGING': 1.0,
            'VOLATILE': 1.5
        }
        
        final_rr = rr_ratio * (regime_rr.get(regime, 1.0) / 1.5)
        
        sl_distance = abs(entry_price - stop_loss_price)
        tp_distance = sl_distance * final_rr
        
        if direction == 1:  # LONG
            return entry_price + tp_distance
        else:  # SHORT
            return entry_price - tp_distance

