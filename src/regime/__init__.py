"""
Regime-Aware Intraday Trading Strategy for XAUUSD.

This module provides:
- Regime detection (TRENDING/RANGING/VOLATILE)
- Microstructure filtering
- Dynamic risk management
- Integrated strategy execution
- Backtesting framework
"""

from .regime_detector import RegimeDetector
from .microstructure_filter import MicrostructureFilter
from .risk_manager import RiskManager
from .strategy import RegimeAwareStrategy
from .backtester import Backtester, Trade

__all__ = [
    'RegimeDetector',
    'MicrostructureFilter',
    'RiskManager',
    'RegimeAwareStrategy',
    'Backtester',
    'Trade'
]

