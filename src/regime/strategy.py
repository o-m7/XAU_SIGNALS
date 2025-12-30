#!/usr/bin/env python3
"""
Complete Regime-Aware Intraday Strategy for XAUUSD.

Combines:
- Regime detection
- ML signal generation
- Microstructure filtering
- Dynamic risk management
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Tuple, Optional, Dict

from .regime_detector import RegimeDetector
from .microstructure_filter import MicrostructureFilter
from .risk_manager import RiskManager


class RegimeAwareStrategy:
    """
    Main strategy class integrating all components.
    """
    
    def __init__(
        self,
        model_path: Path,
        regime_detector: RegimeDetector,
        micro_filter: MicrostructureFilter,
        risk_manager: RiskManager,
        min_prob_threshold: float = 0.55,  # Min probability for trade
        min_regime_strength: float = 0.3   # Min regime strength to trade
    ):
        # Load ML model
        artifact = joblib.load(model_path)
        self.model = artifact['model']
        self.feature_cols = artifact['features']
        
        # Components
        self.regime_detector = regime_detector
        self.micro_filter = micro_filter
        self.risk_manager = risk_manager
        
        # Thresholds
        self.min_prob = min_prob_threshold
        self.min_regime_strength = min_regime_strength
    
    def generate_signal(
        self,
        current_bar: pd.Series,
        regime_info: dict
    ) -> Tuple[int, float, dict]:
        """
        Generate trading signal for current bar.
        
        Returns:
            (signal: int, probability: float, metadata: dict)
            signal: -1 (short), 0 (no trade), +1 (long)
        """
        # Extract features for ML model
        available_features = [f for f in self.feature_cols if f in current_bar.index]
        if len(available_features) < len(self.feature_cols) * 0.8:  # Need at least 80% of features
            return 0, 0.0, {'filter': 'missing_features'}
        
        X = current_bar[available_features].values.reshape(1, -1)
        
        # Handle missing features by filling with 0
        if len(available_features) < len(self.feature_cols):
            X_full = np.zeros((1, len(self.feature_cols)))
            for i, feat in enumerate(self.feature_cols):
                if feat in available_features:
                    idx = available_features.index(feat)
                    X_full[0, i] = X[0, idx]
            X = X_full
        
        # Get ML prediction
        try:
            y_proba = self.model.predict_proba(X)[0]
            y_pred = self.model.predict(X)[0]
        except Exception as e:
            return 0, 0.0, {'filter': f'prediction_error: {e}'}
        
        # Map prediction: 0 -> -1 (short), 1 -> +1 (long)
        signal = 1 if y_pred == 1 else -1
        prob = y_proba[y_pred] if len(y_proba) > y_pred else y_proba[0]
        
        metadata = {
            'ml_signal': signal,
            'ml_prob': prob,
            'regime': regime_info['regime'],
            'regime_strength': regime_info['regime_strength']
        }
        
        # Filter 1: Probability threshold
        if prob < self.min_prob:
            return 0, prob, {**metadata, 'filter': 'low_probability'}
        
        # Filter 2: Regime strength
        if regime_info['regime_strength'] < self.min_regime_strength:
            return 0, prob, {**metadata, 'filter': 'weak_regime'}
        
        # Filter 3: Should we trade this regime?
        session = self._get_current_session(current_bar)
        if not self.regime_detector.should_trade(regime_info['regime'], session):
            return 0, prob, {**metadata, 'filter': 'regime_skip'}
        
        # Filter 4: Microstructure filters
        features_dict = current_bar.to_dict()
        pass_filter, reason = self.micro_filter.filter_signal(signal, features_dict)
        
        if not pass_filter:
            return 0, prob, {**metadata, 'filter': f'micro_{reason}'}
        
        # Filter 5: Risk limits
        can_trade, reason = self.risk_manager.check_drawdown_limits()
        if not can_trade:
            return 0, prob, {**metadata, 'filter': f'risk_{reason}'}
        
        # All filters passed
        return signal, prob, {**metadata, 'filter': 'PASS'}
    
    def _get_current_session(self, bar: pd.Series) -> str:
        """Determine current trading session."""
        if bar.get('is_asia', 0) == 1:
            return 'is_asia'
        elif bar.get('is_europe', 0) == 1:
            return 'is_europe'
        elif bar.get('is_us', 0) == 1:
            return 'is_us'
        return 'other'
    
    def calculate_trade_params(
        self,
        signal: int,
        current_bar: pd.Series,
        regime: str
    ) -> dict:
        """
        Calculate entry, stop, target for a signal.
        """
        entry_price = current_bar['close']
        atr = current_bar.get('ATR_14', current_bar['close'] * 0.01)  # Fallback 1%
        
        # Get stop loss
        sl_price = self.risk_manager.get_stop_loss(
            entry_price, signal, atr, regime
        )
        
        # Get take profit
        tp_price = self.risk_manager.get_take_profit(
            entry_price, sl_price, signal, regime
        )
        
        # Calculate position size
        # Volatility multiplier: reduce size if ATR is elevated
        atr_pct = current_bar.get('atr_pct', 0.5)
        vol_mult = max(0.5, 1.0 - (atr_pct - 0.5))  # Reduce size if atr_pct > 0.5
        
        position_size = self.risk_manager.calculate_position_size(
            entry_price, sl_price, regime, vol_mult
        )
        
        return {
            'entry': entry_price,
            'stop': sl_price,
            'target': tp_price,
            'position_size': position_size,
            'rr_ratio': abs((tp_price - entry_price) / (entry_price - sl_price)) if abs(entry_price - sl_price) > 0 else 0
        }

