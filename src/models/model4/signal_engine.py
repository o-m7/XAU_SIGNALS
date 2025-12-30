"""
Signal Engine for VWAP Mean Reversion

Logic:
1. Check if price is stretched from VWAP (setup exists)
2. Check regime filter (not trending)
3. Check session filter (London/NY)
4. ML predicts probability of successful reversion
5. If P(reversion) > threshold, generate signal
"""
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from typing import Optional, Dict
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
    vwap_zscore: float
    target_price: float
    stop_price: float
    reason: str
    adx: float = 0.0
    timestamp: datetime = None
    features: Dict[str, float] = field(default_factory=dict)


class Model4SignalEngine:
    """
    Mean reversion signal engine.

    Logic:
    1. Check if price is stretched from VWAP (setup exists)
    2. Check regime filter (not trending)
    3. Check session filter (London/NY)
    4. ML predicts probability of successful reversion
    5. If P(reversion) > threshold, generate signal
    """

    def __init__(self, model_path: str, config: Model4Config = None):
        self.config = config or Model4Config()
        self._load_model(model_path)
        self.last_signal_ts = None

    def _load_model(self, path: str):
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        artifact = joblib.load(path)
        self.model = artifact['model']
        self.feature_columns = artifact['feature_columns']
        self.base_win_rate = artifact.get('base_win_rate', 0.5)
        print(f"Loaded Model 4 (VWAP Mean Reversion)")
        print(f"  Base WR: {self.base_win_rate:.1%}")
        print(f"  Features: {len(self.feature_columns)}")

    def generate_signal(
        self,
        features: pd.Series,
        timestamp: datetime = None
    ) -> SignalResult:
        """Generate trading signal from current features."""

        timestamp = timestamp or datetime.now()

        vwap_zscore = features.get('vwap_zscore', 0)
        adx = features.get('adx', 50)
        atr = features.get('atr_14', 1)
        vwap = features.get('vwap', features.get('close', 0))
        close = features.get('close', 0)
        regime = features.get('regime_tradeable', 0)

        # 1. Setup check - is price stretched?
        if abs(vwap_zscore) < self.config.entry_zscore_threshold:
            return SignalResult(
                signal=Signal.NONE,
                confidence=0,
                vwap_zscore=vwap_zscore,
                target_price=0,
                stop_price=0,
                adx=adx,
                reason=f"No setup: |z|={abs(vwap_zscore):.2f} < {self.config.entry_zscore_threshold}",
                timestamp=timestamp
            )

        # 2. Regime check (ADX < max for mean reversion)
        if adx > self.config.max_adx:
            return SignalResult(
                signal=Signal.NONE,
                confidence=0,
                vwap_zscore=vwap_zscore,
                target_price=0,
                stop_price=0,
                adx=adx,
                reason=f"Trending: ADX={adx:.1f} > {self.config.max_adx}",
                timestamp=timestamp
            )

        # 3. Session check
        hour = features.get('hour', timestamp.hour if hasattr(timestamp, 'hour') else 12)
        is_london = self.config.london_start <= hour < self.config.london_end
        is_ny = self.config.ny_start <= hour < self.config.ny_end

        if not (is_london or is_ny):
            return SignalResult(
                signal=Signal.NONE,
                confidence=0,
                vwap_zscore=vwap_zscore,
                target_price=0,
                stop_price=0,
                adx=adx,
                reason=f"Outside session: hour={hour}",
                timestamp=timestamp
            )

        # 4. Spread check
        spread = features.get('spread_pct', 0)
        if spread > self.config.max_spread_pct:
            return SignalResult(
                signal=Signal.NONE,
                confidence=0,
                vwap_zscore=vwap_zscore,
                target_price=0,
                stop_price=0,
                adx=adx,
                reason=f"Spread too wide: {spread*10000:.1f}bps",
                timestamp=timestamp
            )

        # 5. Cooldown
        if self.last_signal_ts:
            cooldown_sec = self.config.cooldown_bars * 300  # Assuming 5T timeframe
            elapsed = (timestamp - self.last_signal_ts).total_seconds()
            if elapsed < cooldown_sec:
                return SignalResult(
                    signal=Signal.NONE,
                    confidence=0,
                    vwap_zscore=vwap_zscore,
                    target_price=0,
                    stop_price=0,
                    adx=adx,
                    reason=f"Cooldown: {(cooldown_sec-elapsed)/60:.0f}min left",
                    timestamp=timestamp
                )

        # 6. ML prediction
        try:
            X = pd.DataFrame([features[self.feature_columns].values], columns=self.feature_columns)
            prob_reversion = self.model.predict_proba(X)[0, 1]
        except Exception as e:
            return SignalResult(
                signal=Signal.NONE,
                confidence=0,
                vwap_zscore=vwap_zscore,
                target_price=0,
                stop_price=0,
                adx=adx,
                reason=f"Model error: {e}",
                timestamp=timestamp
            )

        if prob_reversion < self.config.model_threshold:
            return SignalResult(
                signal=Signal.NONE,
                confidence=prob_reversion,
                vwap_zscore=vwap_zscore,
                target_price=0,
                stop_price=0,
                adx=adx,
                reason=f"Low confidence: {prob_reversion:.2f} < {self.config.model_threshold}",
                timestamp=timestamp
            )

        # 7. Generate signal (opposite direction to stretch)
        if vwap_zscore > 0:
            # Price above VWAP -> SHORT
            signal = Signal.SHORT
            target = vwap
            stop = close + (self.config.stop_atr_mult * atr)
        else:
            # Price below VWAP -> LONG
            signal = Signal.LONG
            target = vwap
            stop = close - (self.config.stop_atr_mult * atr)

        self.last_signal_ts = timestamp

        return SignalResult(
            signal=signal,
            confidence=prob_reversion,
            vwap_zscore=vwap_zscore,
            target_price=target,
            stop_price=stop,
            adx=adx,
            reason=f"Setup confirmed: z={vwap_zscore:.2f}, P={prob_reversion:.2f}",
            timestamp=timestamp,
            features={col: features.get(col, np.nan) for col in self.feature_columns[:5]}
        )

    def reset_state(self) -> None:
        """Reset internal state (for backtesting)."""
        self.last_signal_ts = None
