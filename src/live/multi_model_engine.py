"""
Multi-Model Signal Engine.

Runs multiple models (Model #1 and Model #3) simultaneously
and generates signals from each, clearly labeled by model.
"""

import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional, List
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
                logger.info(f"✓ Loaded {model_config.name} from {model_config.model_path}")
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
        
        return results
    
    def _get_model_display_name(self, model_name: str) -> str:
        """Get display name for model."""
        display_names = {
            'model1': 'Model #1 (Triple-Barrier)',
            'model3': 'Model #3 (CMF/MACD)',
            'y_tb_60': 'Model #1 (Triple-Barrier)',
            'cmf_macd': 'Model #3 (CMF/MACD)',
        }
        return display_names.get(model_name, model_name)

