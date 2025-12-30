"""
Model 4 Signal Engine
Trend Filter + ML Entry Timing
"""
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .config import Model4Config
from .features import get_model4_feature_columns


class Signal(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class SignalResult:
    signal: Signal
    confidence: float
    trend: int
    adx: float
    reason: str
    timestamp: datetime = None
    features: Dict[str, float] = field(default_factory=dict)


class Model4SignalEngine:
    """
    Signal engine for Model 4.

    Logic:
    1. Trend filter determines direction (EMA crossover + ADX)
    2. ML model determines entry quality
    3. Execution filters validate conditions
    """

    def __init__(
        self,
        model_path: str = "models/model4_lgbm.joblib",
        config: Model4Config = None
    ):
        self.config = config or Model4Config()
        self._load_model(model_path)

        # State tracking
        self.last_signal_ts: Optional[datetime] = None
        self.last_signal: Optional[Signal] = None

    def _load_model(self, model_path: str) -> None:
        """Load trained model."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        artifact = joblib.load(model_path)
        self.model = artifact['model']
        self.feature_columns = artifact['feature_columns']
        print(f"Loaded Model 4 from {model_path}")
        print(f"  Features: {len(self.feature_columns)}")

    def generate_signal(
        self,
        features: pd.Series,
        timestamp: datetime = None
    ) -> SignalResult:
        """
        Generate trading signal.

        Parameters:
        -----------
        features : pd.Series
            Current bar features (must include all required columns)
        timestamp : datetime
            Current timestamp for cooldown tracking

        Returns:
        --------
        SignalResult with signal, confidence, and metadata
        """

        timestamp = timestamp or datetime.now()

        # ===== 1. TREND FILTER =====
        trend = features.get('trend', 0)
        adx = features.get('adx', 0)

        if adx < self.config.min_adx:
            return SignalResult(
                signal=Signal.NONE,
                confidence=0.0,
                trend=trend,
                adx=adx,
                reason=f"ADX {adx:.1f} < {self.config.min_adx} (choppy market)",
                timestamp=timestamp
            )

        # ===== 2. SESSION FILTER =====
        hour = features.get('hour', timestamp.hour if hasattr(timestamp, 'hour') else 12)
        is_london = self.config.london_start <= hour < self.config.london_end
        is_ny = self.config.ny_start <= hour < self.config.ny_end

        if not (is_london or is_ny):
            return SignalResult(
                signal=Signal.NONE,
                confidence=0.0,
                trend=trend,
                adx=adx,
                reason=f"Outside London/NY session (hour={hour})",
                timestamp=timestamp
            )

        # ===== 3. SPREAD FILTER =====
        spread_pct = features.get('spread_pct', 0)
        if spread_pct > self.config.max_spread_pct:
            return SignalResult(
                signal=Signal.NONE,
                confidence=0.0,
                trend=trend,
                adx=adx,
                reason=f"Spread {spread_pct*10000:.1f}bps > max {self.config.max_spread_pct*10000:.1f}bps",
                timestamp=timestamp
            )

        # ===== 4. COOLDOWN CHECK =====
        if self.last_signal_ts:
            # Cooldown in seconds (bars * timeframe)
            timeframe_seconds = 300 if self.config.base_timeframe == "5T" else 900
            cooldown_seconds = self.config.cooldown_bars * timeframe_seconds

            elapsed = (timestamp - self.last_signal_ts).total_seconds()
            if elapsed < cooldown_seconds:
                remaining = cooldown_seconds - elapsed
                return SignalResult(
                    signal=Signal.NONE,
                    confidence=0.0,
                    trend=trend,
                    adx=adx,
                    reason=f"Cooldown: {remaining/60:.1f} min remaining",
                    timestamp=timestamp
                )

        # ===== 5. MODEL PREDICTION =====
        try:
            X = features[self.feature_columns].values.reshape(1, -1)
            prob_good_entry = self.model.predict_proba(X)[0, 1]
        except Exception as e:
            return SignalResult(
                signal=Signal.NONE,
                confidence=0.0,
                trend=trend,
                adx=adx,
                reason=f"Model error: {str(e)}",
                timestamp=timestamp
            )

        if prob_good_entry < self.config.model_threshold:
            return SignalResult(
                signal=Signal.NONE,
                confidence=prob_good_entry,
                trend=trend,
                adx=adx,
                reason=f"Low confidence: {prob_good_entry:.2f} < {self.config.model_threshold}",
                timestamp=timestamp
            )

        # ===== 6. GENERATE SIGNAL IN TREND DIRECTION =====
        signal = Signal.LONG if trend == 1 else Signal.SHORT

        # Update state
        self.last_signal_ts = timestamp
        self.last_signal = signal

        return SignalResult(
            signal=signal,
            confidence=prob_good_entry,
            trend=trend,
            adx=adx,
            reason="Model confirmed trend entry",
            timestamp=timestamp,
            features={col: features.get(col, np.nan) for col in self.feature_columns[:5]}
        )

    def reset_state(self) -> None:
        """Reset internal state (for backtesting)."""
        self.last_signal_ts = None
        self.last_signal = None
