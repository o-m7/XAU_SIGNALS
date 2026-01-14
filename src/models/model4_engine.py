#!/usr/bin/env python3
"""
Model #4 Signal Engine (5m News-Gated Long-Only).

Integrates:
- XGBoost classifier (predicts 30m rally > 0.3%)
- News gate (multiplies signal by [0.5, 1.15])
- Long-only logic (threshold_long=0.4, no shorts)
- 30-minute max hold time tracking
- Trade logging to CSV
"""

import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional
import pandas as pd
import numpy as np
import joblib
import asyncio

from .model4_features import compute_model4_features, MODEL4_FEATURES
from ..live.news_gate import NewsGate

logger = logging.getLogger("Model4Engine")


class Model4Config:
    """Configuration for Model #4."""

    def __init__(
        self,
        threshold_long: float = 0.4,  # Min signal to enter long
        threshold_short: float = 1.0,  # Impossible (long-only)
        max_hold_minutes: int = 30,  # Hard exit after 30 minutes
        min_rsi: float = 0.0,  # Minimum RSI (filter overbought, e.g., 75)
        max_rsi: float = 100.0,  # Maximum RSI
        atr_stop_mult: float = 1.5,  # Stop loss = entry - 1.5*ATR
        atr_target_mult: float = 0.5,  # Take profit = entry + 0.5*ATR
        enable_news_gate: bool = True,  # Use news gate multiplier
    ):
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short
        self.max_hold_minutes = max_hold_minutes
        self.min_rsi = min_rsi
        self.max_rsi = max_rsi
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.enable_news_gate = enable_news_gate


class Model4SignalEngine:
    """
    Model #4 Signal Engine.

    Long-only, 5m bars, news-gated, 30-minute max hold.
    """

    def __init__(
        self,
        model_path: str,
        config: Model4Config,
        news_api_key: Optional[str] = None
    ):
        self.model_path = Path(model_path)
        self.config = config

        # Load model
        self._load_model()

        # Initialize news gate
        self.news_gate = NewsGate(
            enable_news_api=False,  # Using free alternatives only
            news_api_key=news_api_key
        )

        # State tracking for 30-minute hold rule
        self._active_position: Optional[Dict] = None

        logger.info(
            f"Model4Engine initialized: long≥{config.threshold_long}, "
            f"max_hold={config.max_hold_minutes}m"
        )

    def _load_model(self):
        """Load XGBoost model and features."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        artifact = joblib.load(self.model_path)

        self.model = artifact["model"]
        self.features = artifact.get("features", MODEL4_FEATURES)

        # Validate features
        if set(self.features) != set(MODEL4_FEATURES):
            logger.warning(
                f"Model features {self.features} != expected {MODEL4_FEATURES}"
            )

        logger.info(f"✓ Model loaded: {len(self.features)} features")

    async def generate_signal(
        self,
        feature_row: pd.DataFrame,
        timestamp: Optional[datetime] = None,
        current_price: Optional[float] = None
    ) -> Dict:
        """
        Generate trading signal from 5m features.

        Args:
            feature_row: Single-row DataFrame with Model4 features computed
            timestamp: Current timestamp (default: now)
            current_price: Current XAUUSD price

        Returns:
            Dict with:
                - signal: "LONG" | "FLAT"
                - proba_up: float [0, 1]
                - ml_signal: float (pre-news-gate)
                - news_gate_mult: float
                - final_signal: float (post-news-gate)
                - tp: Optional[float]
                - sl: Optional[float]
                - hold_minutes: int (if position active)
                - timestamp: datetime
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        # =====================================================================
        # 1. Check if we have an active position that needs to exit
        # =====================================================================
        if self._active_position is not None:
            elapsed = (timestamp - self._active_position['entry_time']).total_seconds() / 60
            if elapsed >= self.config.max_hold_minutes:
                logger.info(
                    f"⏰ Max hold time reached ({elapsed:.1f}m) - "
                    f"closing position at {current_price}"
                )
                self._log_trade_exit(
                    exit_price=current_price,
                    exit_time=timestamp,
                    exit_reason="MAX_HOLD_TIME"
                )
                self._active_position = None

        # =====================================================================
        # 2. ML Inference (XGBoost probability)
        # =====================================================================
        ml_proba = self._predict_proba(feature_row)

        # =====================================================================
        # 3. News Gate Multiplier
        # =====================================================================
        news_result = await self.news_gate.compute_gate_multiplier(
            feature_row, timestamp
        )
        news_mult = news_result['multiplier'] if self.config.enable_news_gate else 1.0

        # =====================================================================
        # 4. Gated Signal = ML × News Multiplier
        # =====================================================================
        gated_signal = ml_proba * news_mult

        # =====================================================================
        # 5. RSI Filter (avoid overbought)
        # =====================================================================
        rsi = feature_row.get('rsi_14', 50.0)
        if isinstance(rsi, pd.Series):
            rsi = rsi.iloc[0]

        if not (self.config.min_rsi <= rsi <= self.config.max_rsi):
            logger.debug(f"RSI filter: {rsi:.1f} outside [{self.config.min_rsi}, {self.config.max_rsi}]")
            gated_signal = 0.0  # Block entry

        # =====================================================================
        # 6. Long-Only Threshold
        # =====================================================================
        if gated_signal >= self.config.threshold_long and self._active_position is None:
            signal = "LONG"
        else:
            signal = "FLAT"

        # =====================================================================
        # 7. Calculate TP/SL (ATR-based)
        # =====================================================================
        tp_price, sl_price = None, None
        if signal == "LONG" and current_price:
            tp_price, sl_price = self._calculate_tp_sl(
                current_price, feature_row
            )

            # Record position entry
            self._active_position = {
                'entry_time': timestamp,
                'entry_price': current_price,
                'tp': tp_price,
                'sl': sl_price,
                'signal_strength': gated_signal,
            }
            logger.info(
                f"📈 LONG position opened at {current_price:.2f} "
                f"(TP={tp_price:.2f}, SL={sl_price:.2f}, signal={gated_signal:.3f})"
            )

        # =====================================================================
        # 8. Build result
        # =====================================================================
        hold_minutes = 0
        if self._active_position:
            hold_minutes = (timestamp - self._active_position['entry_time']).total_seconds() / 60

        return {
            'signal': signal,
            'proba_up': round(ml_proba, 4),
            'ml_signal': round(ml_proba, 4),
            'news_gate_mult': round(news_mult, 3),
            'news_gate_reason': news_result.get('reason', 'N/A'),
            'final_signal': round(gated_signal, 4),
            'tp': tp_price,
            'sl': sl_price,
            'rsi': round(rsi, 1),
            'hold_minutes': round(hold_minutes, 1),
            'timestamp': timestamp,
            'has_position': self._active_position is not None,
        }

    def _predict_proba(self, feature_row: pd.DataFrame) -> float:
        """
        Get ML probability from XGBoost.

        Returns:
            P(up) ∈ [0, 1]
        """
        # Extract features in correct order
        X = feature_row[self.features].values

        # Handle NaN
        if np.isnan(X).any():
            nan_cols = [
                self.features[i]
                for i in range(len(self.features))
                if np.isnan(X[0, i])
            ]
            logger.warning(f"NaN in features: {nan_cols}")
            X = np.nan_to_num(X, nan=0.0)

        # Predict
        proba = self.model.predict_proba(X)[0]

        # Get P(up) - assumes classes are [0, 1] or [-1, 1]
        classes = self.model.classes_
        if 1 in classes:
            up_idx = list(classes).index(1)
        else:
            up_idx = -1

        return proba[up_idx]

    def _calculate_tp_sl(
        self,
        price: float,
        feature_row: pd.DataFrame
    ) -> tuple:
        """
        Calculate TP/SL based on ATR.

        TP = entry + (atr_target_mult × ATR)
        SL = entry - (atr_stop_mult × ATR)

        Returns:
            (tp_price, sl_price)
        """
        atr = feature_row.get('atr_14', feature_row.get('ATR_14', 5.0))
        if isinstance(atr, pd.Series):
            atr = atr.iloc[0]

        tp_price = price + (self.config.atr_target_mult * atr)
        sl_price = price - (self.config.atr_stop_mult * atr)

        return round(tp_price, 2), round(sl_price, 2)

    def check_exit_conditions(
        self,
        current_price: float,
        timestamp: datetime
    ) -> Optional[str]:
        """
        Check if active position should be exited.

        Returns:
            Exit reason ("TP_HIT" | "SL_HIT" | "MAX_HOLD" | None)
        """
        if self._active_position is None:
            return None

        # Check TP
        if current_price >= self._active_position['tp']:
            logger.info(f"✅ TP hit: {current_price} >= {self._active_position['tp']}")
            self._log_trade_exit(current_price, timestamp, "TP_HIT")
            self._active_position = None
            return "TP_HIT"

        # Check SL
        if current_price <= self._active_position['sl']:
            logger.info(f"❌ SL hit: {current_price} <= {self._active_position['sl']}")
            self._log_trade_exit(current_price, timestamp, "SL_HIT")
            self._active_position = None
            return "SL_HIT"

        # Check max hold time
        elapsed = (timestamp - self._active_position['entry_time']).total_seconds() / 60
        if elapsed >= self.config.max_hold_minutes:
            logger.info(f"⏰ Max hold time: {elapsed:.1f}m >= {self.config.max_hold_minutes}m")
            self._log_trade_exit(current_price, timestamp, "MAX_HOLD")
            self._active_position = None
            return "MAX_HOLD"

        return None

    def _log_trade_exit(
        self,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str
    ):
        """
        Log trade to CSV (will be implemented in trade logger module).

        Args:
            exit_price: Exit price
            exit_time: Exit timestamp
            exit_reason: "TP_HIT" | "SL_HIT" | "MAX_HOLD"
        """
        if self._active_position is None:
            return

        entry_price = self._active_position['entry_price']
        pnl_pips = (exit_price - entry_price) * 100  # Gold pips approximation
        hold_minutes = (exit_time - self._active_position['entry_time']).total_seconds() / 60

        result = "WIN" if pnl_pips > 0 else "LOSS"

        logger.info(
            f"📊 Trade closed: {result} | "
            f"Entry={entry_price:.2f}, Exit={exit_price:.2f}, "
            f"PnL={pnl_pips:.0f} pips, Hold={hold_minutes:.1f}m, "
            f"Reason={exit_reason}"
        )

        # TODO: Append to CSV (implement in separate trade logger)

    def get_position_status(self) -> Optional[Dict]:
        """Get current position status."""
        return self._active_position


# =============================================================================
# Standalone Test
# =============================================================================

async def test_model4_engine():
    """Test Model4 engine with mock data."""
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    model_path = project_root / "models" / "model4_news_gated_5m.joblib"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run training script first to create model")
        return

    config = Model4Config()
    engine = Model4SignalEngine(str(model_path), config)

    # Mock features
    mock_features = pd.DataFrame({
        'rsi_14': [45.0],
        'macd_hist_sign': [1.0],
        'bb_width_ratio': [0.02],
        'atr_14': [5.0],
        'volume_delta_5': [0.3],
    })

    result = await engine.generate_signal(
        mock_features,
        current_price=2650.0
    )

    print("Model4 Engine Test:")
    print(f"  Signal: {result['signal']}")
    print(f"  ML Proba: {result['ml_signal']}")
    print(f"  News Gate: ×{result['news_gate_mult']}")
    print(f"  Final Signal: {result['final_signal']}")
    print(f"  TP/SL: {result['tp']}/{result['sl']}")


if __name__ == "__main__":
    asyncio.run(test_model4_engine())
