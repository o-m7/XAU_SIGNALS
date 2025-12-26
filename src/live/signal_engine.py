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
        
        # Calculate TP/SL if we have price and ATR
        tp_price, sl_price = None, None
        if current_price and signal != Signal.FLAT:
            tp_price, sl_price = self._calculate_tp_sl(
                signal, current_price, feature_row, proba_up
            )
        
        return self._make_result(signal, proba_up, timestamp, current_price, tp_price, sl_price)
    
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
        error: Optional[str] = None
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

