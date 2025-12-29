#!/usr/bin/env python3
"""
Risk Guard for Funded Account Protection.

Implements:
- Funded account rules (max drawdown, profit target)
- Confidence-based dynamic cooldowns
- Signal-change filtering

NO daily trade limits - replaced with confidence-based gating.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger("RiskGuard")


@dataclass
class RiskDecision:
    """Decision from risk guard check."""
    allow: bool
    reason: str
    cooldown_s: int = 0


class RiskGuard:
    """
    Risk management guard for funded account protection.
    
    Implements:
    - Max drawdown limit (6% -> FAIL)
    - Profit target (5% -> PASS)
    - Confidence-based dynamic cooldowns
    - Signal-change filtering (same signal blocked unless extreme confidence)
    
    Args:
        start_balance: Initial account balance (default $25,000)
        max_dd_pct: Maximum drawdown as fraction (default 0.06 = 6%)
        profit_target_pct: Profit target as fraction (default 0.05 = 5%)
        thresh_extreme: Extreme confidence threshold (default 0.75)
        thresh_high: High confidence threshold (default 0.65)
    """
    
    def __init__(
        self,
        start_balance: float = 25_000.0,
        max_dd_pct: float = 0.06,
        profit_target_pct: float = 0.05,
        thresh_extreme: float = 0.75,
        thresh_high: float = 0.65,
    ):
        self.start_balance = start_balance
        self.max_dd_pct = max_dd_pct
        self.profit_target_pct = profit_target_pct
        self.thresh_extreme = thresh_extreme
        self.thresh_high = thresh_high
        
        # Computed thresholds
        self.max_dd_level = start_balance * (1 - max_dd_pct)
        self.profit_target = start_balance * (1 + profit_target_pct)
        
        # State
        self.equity = start_balance
        self.peak_equity = start_balance
        self.account_status = "ACTIVE"  # ACTIVE, PASSED, FAILED
        
        # Signal tracking for change-filtering
        self._last_signal: Optional[str] = None
        self._last_signal_ts: Optional[datetime] = None
        
        # Trade history
        self._trade_history = []
        
        logger.info(
            f"RiskGuard initialized: "
            f"balance=${start_balance:,.0f}, "
            f"max_dd={max_dd_pct*100:.0f}%, "
            f"target={profit_target_pct*100:.0f}%, "
            f"thresh_extreme={thresh_extreme}, "
            f"thresh_high={thresh_high}"
        )
    
    def _get_confidence_cooldown(self, prob_up: float) -> int:
        """
        Calculate dynamic cooldown based on confidence.
        
        Rules:
        - Extreme confidence (p >= 0.75 or p <= 0.25): 0 seconds
        - High confidence (p >= 0.65 or p <= 0.35): 60 seconds
        - Otherwise: 300 seconds
        
        Args:
            prob_up: Probability of up move from model
            
        Returns:
            Cooldown in seconds
        """
        # Distance from 0.5 determines confidence
        # p=0.75 means 75% up, p=0.25 means 75% down
        
        # Extreme: p >= 0.75 or p <= 0.25
        if prob_up >= self.thresh_extreme or prob_up <= (1 - self.thresh_extreme):
            return 0
        
        # High: p >= 0.65 or p <= 0.35
        if prob_up >= self.thresh_high or prob_up <= (1 - self.thresh_high):
            return 60
        
        # Default: low confidence (reduced from 300s to 180s for more signals)
        return 180
    
    def _is_extreme_confidence(self, prob_up: float) -> bool:
        """Check if confidence is in extreme range."""
        return prob_up >= self.thresh_extreme or prob_up <= (1 - self.thresh_extreme)
    
    def check_signal(
        self,
        signal: str,
        prob_up: float,
        timestamp: Optional[datetime] = None
    ) -> RiskDecision:
        """
        Check if a signal is allowed under current risk rules.
        
        Args:
            signal: Signal type (LONG, SHORT, FLAT)
            prob_up: Model's probability of up move
            timestamp: Signal timestamp
            
        Returns:
            RiskDecision with allow flag, reason, and cooldown
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Calculate cooldown based on confidence
        cooldown_s = self._get_confidence_cooldown(prob_up)
        
        # Check account status
        if self.account_status == "FAILED":
            return RiskDecision(
                allow=False,
                reason="Account FAILED - max drawdown breached",
                cooldown_s=0,
            )
        
        if self.account_status == "PASSED":
            return RiskDecision(
                allow=False,
                reason="Account PASSED - profit target reached",
                cooldown_s=0,
            )
        
        # FLAT signals are always allowed (no trade)
        if signal == "FLAT":
            return RiskDecision(
                allow=True,
                reason="FLAT signal - no trade",
                cooldown_s=0,
            )
        
        # Check cooldown from last signal
        if self._last_signal_ts and cooldown_s > 0:
            elapsed = (timestamp - self._last_signal_ts).total_seconds()
            if elapsed < cooldown_s:
                remaining = cooldown_s - elapsed
                return RiskDecision(
                    allow=False,
                    reason=f"Cooldown: {remaining:.0f}s remaining",
                    cooldown_s=cooldown_s,
                )
        
        # Check signal-change filtering
        if self._last_signal is not None and signal == self._last_signal:
            # Same signal as before - only allow if extreme confidence
            if not self._is_extreme_confidence(prob_up):
                return RiskDecision(
                    allow=False,
                    reason=f"Same signal ({signal}) - need extreme confidence to repeat",
                    cooldown_s=cooldown_s,
                )
        
        # Signal allowed
        return RiskDecision(
            allow=True,
            reason="Signal allowed",
            cooldown_s=cooldown_s,
        )
    
    def record_signal(
        self,
        signal: str,
        timestamp: Optional[datetime] = None
    ):
        """
        Record that a signal was emitted (for signal-change tracking).
        
        Args:
            signal: Signal type (LONG, SHORT)
            timestamp: Signal timestamp
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        self._last_signal = signal
        self._last_signal_ts = timestamp
        
        logger.debug(f"Recorded signal: {signal} at {timestamp}")
    
    def record_trade(
        self,
        signal: str,
        r_multiple: float,
        risk_pct: float = 0.0025,
        timestamp: Optional[datetime] = None
    ):
        """
        Record a trade outcome and update equity.
        
        Args:
            signal: Trade direction (LONG, SHORT)
            r_multiple: Trade outcome in R (+1 win, -1 loss)
            risk_pct: Risk per trade as fraction of equity
            timestamp: Trade timestamp
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Calculate PnL
        risk_dollars = self.equity * risk_pct
        pnl = risk_dollars * r_multiple
        
        # Update equity
        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        
        # Record trade
        self._trade_history.append({
            "timestamp": timestamp,
            "signal": signal,
            "r_multiple": r_multiple,
            "pnl": pnl,
            "equity": self.equity,
        })
        
        # Check account status
        if self.equity <= self.max_dd_level:
            self.account_status = "FAILED"
            logger.warning(
                f"ACCOUNT FAILED: equity ${self.equity:,.2f} < ${self.max_dd_level:,.2f}"
            )
        elif self.equity >= self.profit_target:
            self.account_status = "PASSED"
            logger.info(
                f"ACCOUNT PASSED: equity ${self.equity:,.2f} >= ${self.profit_target:,.2f}"
            )
        
        logger.info(
            f"Trade recorded: {signal} {r_multiple:+.0f}R = ${pnl:+.2f}, "
            f"equity=${self.equity:,.2f}"
        )
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown as percentage of starting balance."""
        return (self.peak_equity - self.equity) / self.start_balance
    
    def get_status(self) -> Dict:
        """Get current risk status as dict."""
        return {
            "equity": self.equity,
            "start_balance": self.start_balance,
            "peak_equity": self.peak_equity,
            "drawdown_pct": self._calculate_drawdown(),
            "profit_pct": (self.equity - self.start_balance) / self.start_balance,
            "account_status": self.account_status,
            "last_signal": self._last_signal,
            "total_trades": len(self._trade_history),
        }
    
    def reset(self):
        """Reset account to initial state."""
        self.equity = self.start_balance
        self.peak_equity = self.start_balance
        self.account_status = "ACTIVE"
        self._last_signal = None
        self._last_signal_ts = None
        self._trade_history = []
        logger.info("RiskGuard reset to initial state")


# =============================================================================
# Standalone test
# =============================================================================

if __name__ == "__main__":
    from datetime import timedelta
    
    guard = RiskGuard(
        start_balance=25_000,
        max_dd_pct=0.06,
        profit_target_pct=0.05,
        thresh_extreme=0.75,
        thresh_high=0.65,
    )
    
    print("Initial status:")
    for k, v in guard.get_status().items():
        print(f"  {k}: {v}")
    
    now = datetime.now(timezone.utc)
    
    print("\n--- Testing confidence-based cooldowns ---")
    
    # Extreme confidence (0.80) - no cooldown
    decision = guard.check_signal("LONG", 0.80, now)
    print(f"LONG p=0.80: allow={decision.allow}, cooldown={decision.cooldown_s}s, reason={decision.reason}")
    guard.record_signal("LONG", now)
    
    # Try same signal immediately with lower confidence
    decision = guard.check_signal("LONG", 0.65, now + timedelta(seconds=5))
    print(f"LONG p=0.65 (5s later): allow={decision.allow}, reason={decision.reason}")
    
    # Try different signal with high confidence
    decision = guard.check_signal("SHORT", 0.30, now + timedelta(seconds=10))
    print(f"SHORT p=0.30 (10s later): allow={decision.allow}, cooldown={decision.cooldown_s}s")
    
    # Low confidence - should have 300s cooldown
    decision = guard.check_signal("LONG", 0.55, now + timedelta(seconds=15))
    print(f"LONG p=0.55: allow={decision.allow}, cooldown={decision.cooldown_s}s")
    
    print("\n--- Testing signal-change filtering ---")
    guard.reset()
    guard.record_signal("LONG", now)
    
    # Same signal, low confidence - blocked
    decision = guard.check_signal("LONG", 0.60, now + timedelta(minutes=10))
    print(f"LONG p=0.60 (after LONG): allow={decision.allow}, reason={decision.reason}")
    
    # Same signal, extreme confidence - allowed
    decision = guard.check_signal("LONG", 0.80, now + timedelta(minutes=10))
    print(f"LONG p=0.80 (after LONG): allow={decision.allow}, reason={decision.reason}")
    
    # Different signal - allowed
    decision = guard.check_signal("SHORT", 0.20, now + timedelta(minutes=10))
    print(f"SHORT p=0.20 (after LONG): allow={decision.allow}, reason={decision.reason}")
