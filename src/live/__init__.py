"""
Live Trading Signal Engine for XAUUSD.

This package provides real-time signal generation using:
- Polygon.io WebSocket for live price streaming
- Rolling feature computation matching training exactly
- ML model inference with probability thresholds
- Confidence-based risk management (no daily limits)
- Telegram notifications for signal delivery
- Funded account protection

LIVE-ONLY: Does not backfill historical data.
Signals are generated only after warmup from live stream.
"""

from .symbol_resolver import SymbolResolver
from .polygon_stream import PolygonStream, WSMode, StreamEvent
from .feature_buffer import FeatureBuffer
from .signal_engine import SignalEngine
from .telegram_bot import TelegramBot
from .risk_guard import RiskGuard, RiskDecision
from .backfill import PolygonBackfill, backfill_feature_buffer

__all__ = [
    "SymbolResolver",
    "PolygonStream",
    "WSMode",
    "StreamEvent",
    "FeatureBuffer", 
    "SignalEngine",
    "TelegramBot",
    "RiskGuard",
    "RiskDecision",
    "PolygonBackfill",
    "backfill_feature_buffer",
]
