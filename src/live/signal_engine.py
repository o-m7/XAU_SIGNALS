#!/usr/bin/env python3
"""
Signal Engine for Live Trading.

Loads the trained model and generates trading signals
based on probability thresholds.
"""

import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, List
from enum import Enum

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger("SignalEngine")


class Signal(str, Enum):
    """Trading signal types."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class SignalEngine:
    """
    ML-based signal generation engine.
    
    Loads a trained model and generates signals based on
    predicted probabilities and configurable thresholds.
    
    Args:
        model_path: Path to joblib model artifact
        threshold_long: Minimum P(up) to go long (default 0.60)
        threshold_short: Maximum P(up) to go short (default 0.40)
    """
    
    def __init__(
        self,
        model_path: str,
        threshold_long: float = 0.60,
        threshold_short: float = 0.40
    ):
        self.model_path = Path(model_path)
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short
        
        # Load model artifact
        self._load_model()
        
        logger.info(
            f"SignalEngine initialized: "
            f"long>={threshold_long}, short<={threshold_short}"
        )
    
    def _load_model(self):
        """Load model and feature list from artifact."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        artifact = joblib.load(self.model_path)
        
        self.model = artifact["model"]
        self.features = artifact["features"]
        self.params = artifact.get("best_params", {})
        
        logger.info(f"Model loaded: {len(self.features)} features")
        logger.debug(f"Features: {self.features[:5]}...")
    
    def generate_signal(
        self,
        feature_row: pd.DataFrame,
        timestamp: Optional[datetime] = None,
        current_price: Optional[float] = None
    ) -> Dict:
        """
        Generate a trading signal from features.
        
        Args:
            feature_row: Single-row DataFrame with features
            timestamp: Signal timestamp (default: now)
            current_price: Current price for TP/SL calculation
            
        Returns:
            Dict with signal, proba_up, timestamp, TP, SL, etc.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Validate features
        missing = [f for f in self.features if f not in feature_row.columns]
        if missing:
            logger.error(f"Missing features: {missing}")
            return self._make_result(Signal.FLAT, 0.5, timestamp, None, None, None, "Missing features")
        
        # Extract feature array in correct order
        X = feature_row[self.features].values
        
        # Check for NaN
        if np.isnan(X).any():
            nan_cols = [
                self.features[i] 
                for i in range(len(self.features)) 
                if np.isnan(X[0, i])
            ]
            logger.warning(f"NaN in features: {nan_cols[:5]}")
            return self._make_result(Signal.FLAT, 0.5, timestamp, None, None, None, "NaN features")
        
        # Get prediction
        try:
            proba = self.model.predict_proba(X)[0]
            
            # Probability of +1 (up) class
            # Model classes are typically [-1, 1] or [0, 1]
            classes = self.model.classes_
            if 1 in classes:
                up_idx = list(classes).index(1)
            else:
                up_idx = -1  # Last class
            
            proba_up = proba[up_idx]
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._make_result(Signal.FLAT, 0.5, timestamp, None, None, None, str(e))
        
        # Determine signal
        if proba_up >= self.threshold_long:
            signal = Signal.LONG
        elif proba_up <= self.threshold_short:
            signal = Signal.SHORT
        else:
            signal = Signal.FLAT
        
        # =====================================================================
        # WICK FILTER: Prevent shorting bullish absorption wicks
        # =====================================================================
        # If model wants to SHORT a large upper wick (bearish pattern)
        # BUT order flow is positive (buying into the wick)
        # This is a "bullish absorption" - don't short it!
        # =====================================================================
        if signal == Signal.SHORT:
            try:
                # Check if required features exist
                required_cols = ['upper_wick', 'range', 'synthetic_order_flow']
                if all(col in feature_row.columns for col in required_cols):
                    # Get current candle metrics (feature_row is a single-row DataFrame)
                    current_candle = feature_row.iloc[0]
                    upper_wick = current_candle['upper_wick']
                    range_val = current_candle['range']
                    flow = current_candle['synthetic_order_flow']
                    
                    # Calculate upper wick percentage
                    if range_val > 0:
                        upper_wick_pct = upper_wick / range_val
                        
                        # Trap condition: Large upper wick (>30%) + Positive flow = Bullish absorption
                        # This means buyers are absorbing the selling pressure - don't short!
                        if upper_wick_pct > 0.3 and flow > 0:
                            logger.warning(
                                f"⛔ FILTERED: Model shorting a Bullish Absorption Wick. "
                                f"Wick={upper_wick_pct:.2%}, Flow={flow:.4f}"
                            )
                            signal = Signal.FLAT
            except Exception as e:
                logger.debug(f"Wick filter check failed (non-critical): {e}")
                # Continue with original signal if filter check fails
        
        # =====================================================================
        # CHURN FILTER: Validate signal using close_mid_diff (Feature #2)
        # =====================================================================
        # If Volume is massive but Price isn't moving, it's a coin toss.
        # We use close_mid_diff to detect aggressive buying/selling at close.
        # =====================================================================
        conviction = 1.0  # Default: full conviction
        if signal != Signal.FLAT:
            conviction = self._validate_signal(signal, feature_row)
            
            # =====================================================================
            # VOLATILITY FILTER: High win-rate strategies avoid dead markets
            # =====================================================================
            # Reject trades when volatility is too low (spread kills you) or
            # spread is too high (cost kills you)
            # =====================================================================
            if not self._check_volatility_filter(feature_row):
                logger.warning(
                    f"⚠️ FILTERED: Volatility filter failed. Market conditions not suitable."
                )
                signal = Signal.FLAT
                conviction = 0.0
            
            # If conviction is too low, treat as FLAT
            elif conviction < 0.5:
                logger.warning(
                    f"⚠️ FILTERED: Low conviction signal ({conviction:.2f}). "
                    f"Signal changed to FLAT."
                )
                signal = Signal.FLAT
                conviction = 0.0
        
        # Calculate TP/SL if we have price and ATR
        tp_price, sl_price = None, None
        if current_price and signal != Signal.FLAT:
            tp_price, sl_price = self._calculate_tp_sl(
                signal, current_price, feature_row, proba_up
            )
        
        return self._make_result(signal, proba_up, timestamp, current_price, tp_price, sl_price, conviction=conviction)
    
    def _check_volatility_filter(self, feature_row: pd.DataFrame) -> bool:
        """
        Smart Volatility Filter: The Win Rate Booster.
        
        High win-rate strategies avoid:
        - Low volatility (where spread kills you)
        - Extreme spread (where cost kills you)
        
        Returns True if volatility is good for a High-WR trade.
        """
        try:
            required_cols = ['ATR_14', 'spread']
            if not all(col in feature_row.columns for col in required_cols):
                # If we don't have the features, allow the trade (fail open)
                return True
            
            current_bar = feature_row.iloc[0]
            
            # 1. Check ATR (Is there enough juice to cover spread?)
            min_atr = 0.50  # Minimum 50 cents move expected
            if current_bar['ATR_14'] < min_atr:
                logger.debug(f"Volatility filter: ATR too low ({current_bar['ATR_14']:.2f} < {min_atr})")
                return False  # Market is dead, don't trade
            
            # 2. Check Spread (Is cost too high?)
            max_spread = 0.20  # Max 20 cents spread
            if current_bar['spread'] > max_spread:
                logger.debug(f"Volatility filter: Spread too high ({current_bar['spread']:.2f} > {max_spread})")
                return False  # Too expensive
            
            return True
            
        except Exception as e:
            logger.debug(f"Volatility filter check failed (non-critical): {e}")
            # Default to allowing trade if check fails
            return True
    
    def _validate_signal(self, signal: Signal, feature_row: pd.DataFrame) -> float:
        """
        The "Churn" Filter: Validate signal using close_mid_diff (Feature #2).
        
        If Volume is massive but Price isn't moving, it's a coin toss.
        We use close_mid_diff to detect aggressive buying/selling at close.
        
        Args:
            signal: 1 (Long) or -1 (Short) - but we use Signal enum
            feature_row: Single-row DataFrame with features
            
        Returns:
            conviction: 1.0 (full conviction) or 0.5 (reduced risk) or 0.0 (no trade)
        """
        try:
            # Check if required features exist
            required_cols = ['close_mid_diff', 'volume', 'range']
            if not all(col in feature_row.columns for col in required_cols):
                # If we don't have the features, default to full conviction
                return 1.0
            
            current_bar = feature_row.iloc[0]
            
            # 1. Check for Volume Churn (High Vol / Low Range)
            # This detects when there's high volume but low price movement
            vol_churn = current_bar['volume'] / (current_bar['range'] + 0.0001)
            
            # 2. Check Aggressor status using close_mid_diff (Feature #2)
            # close_mid_diff = close - mid
            # If Close is below Mid, Sellers are aggressive at the close.
            # If Close is above Mid, Buyers are aggressive at the close.
            close_mid = current_bar['close_mid_diff']
            
            if signal == Signal.LONG:  # Model wants to go LONG
                # BUT Close is below Midpoint (Aggressive Selling)
                if close_mid < 0:
                    logger.warning(
                        f"⚠️ CAUTION: Model Long, but closing on the Bid (Weak Close). "
                        f"close_mid_diff={close_mid:.4f}. Reducing Risk."
                    )
                    return 0.5  # Half Size or No Trade
                    
            elif signal == Signal.SHORT:  # Model wants to go SHORT
                # BUT Close is above Midpoint (Aggressive Buying)
                if close_mid > 0:
                    logger.warning(
                        f"⚠️ CAUTION: Model Short, but closing on the Ask (Strong Close). "
                        f"close_mid_diff={close_mid:.4f}. Reducing Risk."
                    )
                    return 0.5  # Half Size or No Trade
            
            # Signal direction matches close aggressor direction - full conviction
            return 1.0
            
        except Exception as e:
            logger.debug(f"Churn filter check failed (non-critical): {e}")
            # Default to full conviction if check fails
            return 1.0
    
    def _calculate_tp_sl(
        self,
        signal: Signal,
        price: float,
        feature_row: pd.DataFrame,
        proba_up: float
    ) -> tuple:
        """
        Calculate Take Profit and Stop Loss levels.
        
        MATCHES TRAINING FORMULA (triple-barrier from features_complete.py):
            Upper barrier: price * (1 + TP_mult * ATR_14 / price)
            Lower barrier: price * (1 - SL_mult * ATR_14 / price)
        
        Default multipliers: TP_mult = 1.0, SL_mult = 1.0
        This means: TP = price + ATR, SL = price - ATR (for LONG)
        
        Args:
            signal: LONG or SHORT
            price: Current price
            feature_row: Features (must contain ATR_14)
            proba_up: Model confidence
            
        Returns:
            (tp_price, sl_price)
        """
        # =====================================================================
        # FIXED TP/SL WITH 1:1.5 RISK:REWARD RATIO
        # 
        # Account: $25,000
        # Risk: 1% = $250 per trade
        # Reward: 1.5x = $375 per trade
        #
        # For gold (XAUUSD) intraday trading:
        # SL distance: $15 (reasonable for intraday)
        # TP distance: $22.50 (1.5x SL for 1:1.5 R:R)
        # =====================================================================
        
        # Fixed distances for XAUUSD (in dollars)
        SL_DISTANCE = 15.0   # Stop loss: $15 from entry
        TP_DISTANCE = 22.50  # Take profit: $22.50 from entry (1.5x SL)
        
        # Calculate TP/SL prices
        if signal == Signal.LONG:
            # Long: TP above, SL below
            tp_price = price + TP_DISTANCE
            sl_price = price - SL_DISTANCE
        else:  # SHORT
            # Short: TP below, SL above
            tp_price = price - TP_DISTANCE
            sl_price = price + SL_DISTANCE
        
        return round(tp_price, 2), round(sl_price, 2)
    
    def _make_result(
        self,
        signal: Signal,
        proba_up: float,
        timestamp: datetime,
        price: Optional[float] = None,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        error: Optional[str] = None,
        conviction: float = 1.0
    ) -> Dict:
        """Create signal result dict."""
        return {
            "signal": signal.value,
            "proba_up": round(proba_up, 4),
            "timestamp": timestamp,
            "price": price,
            "tp": tp_price,
            "sl": sl_price,
            "threshold_long": self.threshold_long,
            "threshold_short": self.threshold_short,
            "error": error,
            "conviction": conviction,  # 1.0 = full, 0.5 = reduced, 0.0 = filtered
        }
    
    def get_feature_list(self) -> List[str]:
        """Get the list of features expected by the model."""
        return self.features.copy()
    
    def update_thresholds(self, threshold_long: float, threshold_short: float):
        """Update signal thresholds."""
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short
        logger.info(f"Thresholds updated: long>={threshold_long}, short<={threshold_short}")


# =============================================================================
# Standalone test
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    import numpy as np
    
    # Test with mock features
    model_path = Path(__file__).parent.parent.parent / "models" / "y_tb_60_hgb_tuned.joblib"
    
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        exit(1)
    
    engine = SignalEngine(
        model_path=str(model_path),
        threshold_long=0.60,
        threshold_short=0.40
    )
    
    print(f"Features expected: {len(engine.features)}")
    
    # Create mock feature row
    mock_row = pd.DataFrame({
        f: [np.random.randn()] for f in engine.features
    })
    
    result = engine.generate_signal(mock_row)
    print(f"\nMock signal result:")
    print(f"  Signal: {result['signal']}")
    print(f"  P(up): {result['proba_up']}")
    print(f"  Time: {result['timestamp']}")

