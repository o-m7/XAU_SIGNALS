"""
Multi-Model Signal Engine.

Runs multiple models (Model #1 and Model #3) simultaneously
and generates signals from each, clearly labeled by model.
"""

import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List
from collections import defaultdict, deque
import numpy as np
import pandas as pd
import joblib

from .signal_engine import SignalEngine, Signal

logger = logging.getLogger("MultiModelEngine")


class ModelConfig:
    """Configuration for a single model."""
    def __init__(
        self,
        name: str,
        model_path: str,
        threshold_long: float,
        threshold_short: float,
        enabled: bool = True
    ):
        self.name = name
        self.model_path = Path(model_path)
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short
        self.enabled = enabled


class MultiModelSignalEngine:
    """
    Multi-model signal generation engine.
    
    Runs multiple models simultaneously and generates signals
    from each model independently.
    
    Args:
        models: List of ModelConfig objects
    """
    
    def __init__(self, models: List[ModelConfig]):
        self._models = [m for m in models if m.enabled]
        self.engines = {}

        # Signal rate monitoring (track last N signals per model)
        self.signal_history = defaultdict(lambda: deque(maxlen=1000))  # Last 1000 signals
        self.prediction_count = defaultdict(int)  # Total predictions per model
        self.signal_count = defaultdict(lambda: {'LONG': 0, 'SHORT': 0, 'FLAT': 0})
        self.last_hourly_summary = datetime.now(timezone.utc)

        # Safety thresholds
        self.MAX_SIGNALS_PER_HOUR = 50  # Alert if more than 50 signals/hour
        self.MIN_SIGNALS_PER_DAY = 1    # Alert if 0 signals in 24 hours

        for model_config in self._models:
            try:
                engine = SignalEngine(
                    model_path=str(model_config.model_path),
                    threshold_long=model_config.threshold_long,
                    threshold_short=model_config.threshold_short
                )
                self.engines[model_config.name] = {
                    'engine': engine,
                    'config': model_config
                }
                logger.info(
                    f"✓ Loaded {model_config.name} | "
                    f"Thresholds: LONG≥{model_config.threshold_long:.2f}, "
                    f"SHORT≤{model_config.threshold_short:.2f}"
                )
            except Exception as e:
                logger.error(f"✗ Failed to load {model_config.name}: {e}")

        if not self.engines:
            raise ValueError("No models successfully loaded")

        logger.info(f"MultiModelSignalEngine initialized with {len(self.engines)} models")
    
    @property
    def models(self):
        """Return list of enabled model configs."""
        return self._models
    
    def generate_signals(
        self,
        feature_row: pd.DataFrame,
        timestamp: Optional[datetime] = None,
        current_price: Optional[float] = None
    ) -> Dict[str, Dict]:
        """
        Generate signals from all models.
        
        Args:
            feature_row: Single-row DataFrame with features
            timestamp: Signal timestamp (default: now)
            current_price: Current price for TP/SL calculation
            
        Returns:
            Dict mapping model_name -> signal result dict
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        results = {}
        
        for model_name, model_data in self.engines.items():
            try:
                engine = model_data['engine']
                config = model_data['config']
                
                # Generate signal for this model
                result = engine.generate_signal(
                    feature_row=feature_row,
                    timestamp=timestamp,
                    current_price=current_price
                )
                
                # Add model name to result
                result['model_name'] = model_name
                result['model_display'] = self._get_model_display_name(model_name)
                
                results[model_name] = result

                # Track signal statistics
                self.prediction_count[model_name] += 1
                signal_type = result['signal']
                self.signal_count[model_name][signal_type] += 1

                # Record non-FLAT signals with timestamp
                if signal_type != 'FLAT':
                    self.signal_history[model_name].append({
                        'timestamp': timestamp,
                        'signal': signal_type,
                        'proba_up': result['proba_up']
                    })

                # Check for anomalies
                self._check_signal_anomalies(model_name, timestamp)

            except Exception as e:
                logger.error(f"Error generating signal for {model_name}: {e}")
                results[model_name] = {
                    'signal': Signal.FLAT,
                    'proba_up': 0.5,
                    'timestamp': timestamp,
                    'tp_price': None,
                    'sl_price': None,
                    'model_name': model_name,
                    'model_display': self._get_model_display_name(model_name),
                    'error': str(e)
                }

        # Hourly summary
        self._maybe_print_hourly_summary(timestamp)

        return results
    
    def _get_model_display_name(self, model_name: str) -> str:
        """Get display name for model."""
        display_names = {
            # Phase 5 models (60-72% WR)
            'model1_hgb': 'Model 1 HGB (72.8% WR)',
            'model3_cmf_macd': 'Model 3 CMF/MACD (71.2% WR)',
            'model_rf': 'Model RF (69.3% WR)',
            'model_5min': 'Model 5min Scalp',
            # Legacy models
            'model1': 'Model #1 (Triple-Barrier)',
            'model3': 'Model #3 (CMF/MACD)',
            'y_tb_60': 'Model #1 (Triple-Barrier)',
            'cmf_macd': 'Model #3 (CMF/MACD)',
            # Other models
            'model3_cmf_macd_v4': 'Model #3 v4 (CMF/MACD)',
            'model1_high_conf': 'Model #1 HC (High Conf)',
            'model_rf_v4': 'Model RF v4 (Random Forest)',
            'model4_news_gated_5m': 'Model #4 (5m News-Gated LONG)',
        }
        return display_names.get(model_name, model_name)

    def _check_signal_anomalies(self, model_name: str, timestamp: datetime):
        """Check for signal rate anomalies (over-firing or dead models)."""
        # Get signals in last hour
        one_hour_ago = timestamp - timedelta(hours=1)
        signals_last_hour = [
            s for s in self.signal_history[model_name]
            if s['timestamp'] > one_hour_ago
        ]

        # Check for over-firing (too many signals per hour)
        if len(signals_last_hour) > self.MAX_SIGNALS_PER_HOUR:
            logger.warning(
                f"⚠️ {model_name} OVER-FIRING: "
                f"{len(signals_last_hour)} signals in last hour "
                f"(threshold: {self.MAX_SIGNALS_PER_HOUR})"
            )

        # Check for dead models (no signals in 24 hours)
        if self.prediction_count[model_name] > 1440:  # ~24 hours of minute bars
            twenty_four_hours_ago = timestamp - timedelta(hours=24)
            signals_last_day = [
                s for s in self.signal_history[model_name]
                if s['timestamp'] > twenty_four_hours_ago
            ]

            if len(signals_last_day) < self.MIN_SIGNALS_PER_DAY:
                logger.warning(
                    f"⚠️ {model_name} DEAD MODEL: "
                    f"{len(signals_last_day)} signals in last 24 hours "
                    f"(minimum: {self.MIN_SIGNALS_PER_DAY}). "
                    f"Check thresholds!"
                )

    def _maybe_print_hourly_summary(self, timestamp: datetime):
        """Print hourly summary of signal statistics."""
        # Check if an hour has passed
        if timestamp - self.last_hourly_summary < timedelta(hours=1):
            return

        self.last_hourly_summary = timestamp

        logger.info("=" * 80)
        logger.info("📊 HOURLY SIGNAL SUMMARY")
        logger.info("=" * 80)

        for model_name in self.engines.keys():
            counts = self.signal_count[model_name]
            total = counts['LONG'] + counts['SHORT'] + counts['FLAT']
            preds = self.prediction_count[model_name]

            # Get config for threshold display
            config = self.engines[model_name]['config']

            # Calculate signal rate
            one_hour_ago = timestamp - timedelta(hours=1)
            signals_last_hour = [
                s for s in self.signal_history[model_name]
                if s['timestamp'] > one_hour_ago
            ]
            signals_per_hour = len(signals_last_hour)

            # Status check
            status = "✅"
            if signals_per_hour > self.MAX_SIGNALS_PER_HOUR:
                status = "⚠️ OVER-FIRING"
            elif signals_per_hour == 0 and preds > 60:
                status = "⚠️ DEAD"

            logger.info(
                f"{status} {model_name}: "
                f"LONG={counts['LONG']}, SHORT={counts['SHORT']}, FLAT={counts['FLAT']} "
                f"({signals_per_hour} signals/hr) | "
                f"Thresholds: ≥{config.threshold_long:.2f}/≤{config.threshold_short:.2f}"
            )

        logger.info("=" * 80)

