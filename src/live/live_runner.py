#!/usr/bin/env python3
"""
Live Trading Signal Runner.

Main orchestration script that:
1. Optionally backfills historical data via REST API
2. Streams live prices from Polygon.io WebSocket
3. Maintains rolling feature buffer
4. Generates ML-based signals
5. Enforces confidence-based risk management
6. Sends Telegram notifications

Usage:
    cd /Users/omar/Desktop/ML/xauusd_signals
    source venv/bin/activate
    python src/live/live_runner.py --backfill

Environment Variables (loaded from .env):
    POLYGON_API_KEY     - Polygon.io API key (required)
    TELEGRAM_BOT_TOKEN  - Telegram bot token (required)
    TELEGRAM_CHAT_ID    - Telegram chat/channel ID (required)
    BASE_SYMBOL         - Base currency (default: XAU)
    QUOTE_SYMBOL        - Quote currency (default: USD)
    WS_MODE             - WebSocket mode: quotes, aggs_minute, aggs_second
    THRESH_EXTREME      - Extreme confidence threshold (default: 0.75)
    THRESH_HIGH         - High confidence threshold (default: 0.65)
"""

import os
import sys
import signal
import logging
import time
import fcntl
import atexit
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Load environment variables from .env FIRST
from dotenv import load_dotenv

# Find .env file - check multiple locations
env_paths = [
    Path(__file__).parent.parent.parent / ".env",  # xauusd_signals/.env
    Path("/Users/omar/Desktop/ML/.env"),  # ML/.env
    Path.home() / ".env",
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break
else:
    load_dotenv()  # Try default

# Now import our modules
from .polygon_stream import PolygonStream, WSMode
from .feature_buffer import FeatureBuffer
from .signal_engine import SignalEngine
from .telegram_bot import TelegramBot
from .risk_guard import RiskGuard
from .symbol_resolver import SymbolResolver
from .backfill import backfill_feature_buffer

# Suppress pandas FutureWarnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, message='.*Downcasting.*')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("live_runner.log"),
    ]
)
logger = logging.getLogger("LiveRunner")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "y_tb_60_hgb_tuned.joblib"
LOCK_FILE = PROJECT_ROOT / ".live_runner.lock"
PID_FILE = PROJECT_ROOT / ".live_runner.pid"

# Global lock file handle
_lock_file_handle = None


def acquire_singleton_lock() -> bool:
    """
    Acquire a singleton lock to prevent multiple instances.
    
    Returns:
        True if lock acquired, False if another instance is running.
    """
    global _lock_file_handle
    
    try:
        _lock_file_handle = open(LOCK_FILE, 'w')
        fcntl.flock(_lock_file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        
        # Write PID
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
        
        logger.info(f"Singleton lock acquired (PID: {os.getpid()})")
        return True
        
    except (IOError, OSError):
        # Another instance is running
        try:
            with open(PID_FILE, 'r') as f:
                existing_pid = f.read().strip()
            logger.error(f"Another instance is already running (PID: {existing_pid})")
        except:
            logger.error("Another instance is already running")
        return False


def release_singleton_lock():
    """Release the singleton lock."""
    global _lock_file_handle
    
    if _lock_file_handle:
        try:
            fcntl.flock(_lock_file_handle.fileno(), fcntl.LOCK_UN)
            _lock_file_handle.close()
        except:
            pass
    
    # Clean up files
    for f in [LOCK_FILE, PID_FILE]:
        try:
            f.unlink()
        except:
            pass
    
    logger.info("Singleton lock released")


def load_config() -> dict:
    """
    Load configuration from environment variables.
    
    Fails fast if required variables are missing.
    """
    # Required variables
    polygon_api_key = os.environ.get("POLYGON_API_KEY")
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    missing = []
    if not polygon_api_key:
        missing.append("POLYGON_API_KEY")
    if not telegram_token:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not telegram_chat_id:
        missing.append("TELEGRAM_CHAT_ID")
    
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        logger.error("Create a .env file with these variables or set them in your environment")
        sys.exit(1)
    
    # Optional variables with defaults
    config = {
        "polygon_api_key": polygon_api_key,
        "telegram_token": telegram_token,
        "telegram_chat_id": telegram_chat_id,
        "base_symbol": os.environ.get("BASE_SYMBOL", "XAU"),
        "quote_symbol": os.environ.get("QUOTE_SYMBOL", "USD"),
        "ws_mode": os.environ.get("WS_MODE", "all"),  # Connect to ALL channels by default
        "thresh_extreme": float(os.environ.get("THRESH_EXTREME", "0.75")),
        "thresh_high": float(os.environ.get("THRESH_HIGH", "0.65")),
    }
    
    return config


class LiveRunner:
    """
    Main orchestration class for live signal generation.
    
    Supports both:
    - Backfill mode: Uses REST API to warm up, then streams
    - Live-only mode: Waits for stream to warm up (~65 min)
    """
    
    def __init__(
        self,
        config: dict,
        model_path: str = str(DEFAULT_MODEL_PATH),
        threshold_long: float = 0.60,
        threshold_short: float = 0.40,
        risk_pct: float = 0.01,  # 1% risk per trade
        start_balance: float = 25_000.0,
        max_dd_pct: float = 0.06,
        profit_target_pct: float = 0.05,
        backfill: bool = True,
        backfill_bars: int = 500,
    ):
        self.config = config
        self.model_path = model_path
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short
        self.risk_pct = risk_pct
        self.backfill_enabled = backfill
        self.backfill_bars = backfill_bars
        
        # Create symbol resolver
        self.resolver = SymbolResolver(
            base=config["base_symbol"],
            quote=config["quote_symbol"]
        )
        
        # Parse WS mode
        ws_mode_raw = config["ws_mode"]
        ws_mode_str = ws_mode_raw.lower().strip() if ws_mode_raw else "all"
        
        # Log environment variable check
        ws_mode_env = os.environ.get("WS_MODE", "NOT SET")
        logger.info(f"🔧 Environment WS_MODE: '{ws_mode_env}'")
        logger.info(f"🔧 Config WS_MODE: '{ws_mode_str}' (raw: '{ws_mode_raw}')")
        
        if ws_mode_str == "quotes":
            self.ws_mode = WSMode.QUOTES
            logger.warning("⚠️  WS Mode: QUOTES only (1 channel) - Set WS_MODE=all for all 3 channels")
        elif ws_mode_str == "aggs_minute":
            self.ws_mode = WSMode.AGGS_MINUTE
            logger.warning("⚠️  WS Mode: MINUTE AGGREGATES only (1 channel) - Set WS_MODE=all for all 3 channels")
        elif ws_mode_str == "aggs_second":
            self.ws_mode = WSMode.AGGS_SECOND
            logger.warning("⚠️  WS Mode: SECOND AGGREGATES only (1 channel) - Set WS_MODE=all for all 3 channels")
        elif ws_mode_str == "all":
            self.ws_mode = WSMode.ALL
            logger.info("✅ WS Mode: ALL (3 channels: quotes, minute, second)")
        else:
            logger.error(f"❌ Invalid WS_MODE: '{ws_mode_str}'. Defaulting to 'all'")
            logger.error(f"   Valid values: quotes, aggs_minute, aggs_second, all")
            self.ws_mode = WSMode.ALL  # Default to ALL instead of crashing
        
        # Initialize components
        logger.info("Initializing LiveRunner components...")
        
        # 1. Feature buffer
        self.feature_buffer = FeatureBuffer(max_window=500)
        
        # 2. Signal engine
        self.signal_engine = SignalEngine(
            model_path=model_path,
            threshold_long=threshold_long,
            threshold_short=threshold_short,
        )
        
        # 3. Risk guard with confidence-based gating
        self.risk_guard = RiskGuard(
            start_balance=start_balance,
            max_dd_pct=max_dd_pct,
            profit_target_pct=profit_target_pct,
            thresh_extreme=config["thresh_extreme"],
            thresh_high=config["thresh_high"],
        )
        
        # 4. Telegram bot
        self.telegram_bot = TelegramBot(
            token=config["telegram_token"],
            chat_id=config["telegram_chat_id"],
            enabled=True,
        )
        
        # 5. Price stream (created but not started yet)
        self.price_stream = PolygonStream(
            api_key=config["polygon_api_key"],
            resolver=self.resolver,
            mode=self.ws_mode,
            on_event=self._on_event,
        )
        
        # State
        self._running = False
        self._current_price: Optional[float] = None
        self._events_received = 0
        self._signals_generated = 0
        self._warmup_notified = False
        self._last_signal_time: Optional[datetime] = None
        
        logger.info("LiveRunner initialized successfully")
    
    def _do_backfill(self):
        """Perform REST API backfill to warm up feature buffer."""
        logger.info("=" * 60)
        logger.info("  PERFORMING REST API BACKFILL")
        logger.info("=" * 60)
        
        success = backfill_feature_buffer(
            feature_buffer=self.feature_buffer,
            api_key=self.config["polygon_api_key"],
            resolver=self.resolver,
            lookback_bars=self.backfill_bars,
        )
        
        if success:
            logger.info("✓ Backfill successful - feature buffer is ready")
            self._warmup_notified = True  # Skip warmup notification
        else:
            logger.warning("⚠ Backfill failed - falling back to live warmup")
    
    def start(self):
        """Start the live signal engine."""
        logger.info("=" * 60)
        logger.info("  STARTING LIVE SIGNAL ENGINE")
        logger.info("=" * 60)
        logger.info(f"Symbol: {self.resolver.display_name()}")
        logger.info(f"WS Mode: {self.ws_mode.value}")
        logger.info(f"WS Channel: {self._get_ws_channel()}")
        logger.info(f"Model: {self.model_path}")
        logger.info(f"Thresholds: long>={self.threshold_long}, short<={self.threshold_short}")
        logger.info(f"Risk per trade: {self.risk_pct*100:.2f}%")
        logger.info(f"Confidence: extreme={self.config['thresh_extreme']}, high={self.config['thresh_high']}")
        logger.info(f"Backfill: {'ENABLED' if self.backfill_enabled else 'DISABLED'}")
        logger.info("=" * 60)
        
        self._running = True
        
        # Do backfill if enabled
        if self.backfill_enabled:
            self._do_backfill()
            
            # Send ready notification
            if self.feature_buffer.is_ready():
                self.telegram_bot.send_alert(
                    "🚀 Signal Engine Ready",
                    f"Symbol: {self.resolver.display_name()}\n"
                    f"Mode: REST backfill + {self.ws_mode.value}\n"
                    f"Model: y_tb_60\n"
                    f"Thresholds: L≥{self.threshold_long}, S≤{self.threshold_short}\n"
                    f"Risk: {self.risk_pct*100:.2f}%\n"
                    f"Bars loaded: {self.feature_buffer.get_bar_count()}"
                )
        else:
            logger.info("LIVE-ONLY MODE: No backfill. Warming up from stream (~65 min).")
        
        # Start price stream
        logger.info("Starting WebSocket stream...")
        self.price_stream.start()
        
        # Keep running until stopped
        try:
            while self._running:
                time.sleep(1)
                
                # Periodic status log
                if self._events_received > 0 and self._events_received % 100 == 0:
                    price_str = f"{self._current_price:.2f}" if self._current_price else "N/A"
                    logger.info(
                        f"Status: events={self._events_received}, "
                        f"signals={self._signals_generated}, "
                        f"price={price_str}"
                    )
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()
    
    def _get_ws_channel(self) -> str:
        """Get the WebSocket channel(s) being used."""
        if self.ws_mode == WSMode.ALL:
            return f"{self.resolver.ws_quotes()}, {self.resolver.ws_aggs_minute()}, {self.resolver.ws_aggs_second()}"
        elif self.ws_mode == WSMode.QUOTES:
            return self.resolver.ws_quotes()
        elif self.ws_mode == WSMode.AGGS_MINUTE:
            return self.resolver.ws_aggs_minute()
        else:
            return self.resolver.ws_aggs_second()
    
    def stop(self):
        """Stop the live signal engine."""
        logger.info("Stopping LiveRunner...")
        self._running = False
        
        # Stop price stream
        self.price_stream.stop()
        
        # Send shutdown notification
        if self.telegram_bot.enabled and self._warmup_notified:
            status = self.risk_guard.get_status()
            self.telegram_bot.send_alert(
                "🛑 Signal Engine Stopped",
                f"Events: {self._events_received}\n"
                f"Signals: {self._signals_generated}\n"
                f"Account: {status['account_status']}\n"
                f"Equity: ${status['equity']:,.2f}"
            )
        
        logger.info("LiveRunner stopped")
    
    def _on_event(self, event: dict):
        """Handle incoming price event."""
        self._events_received += 1
        
        # Update price
        if event["type"] == "quote":
            self._current_price = event.get("mid")
        else:  # bar
            self._current_price = event.get("close")
        
        # Update feature buffer based on event type
        if event["type"] == "quote":
            feature_row = self.feature_buffer.update_from_quote(event)
        else:  # bar
            feature_row = self.feature_buffer.update_from_bar(event)
        
        # During warmup - no signals, no Telegram
        if self.feature_buffer.is_warming_up():
            return
        
        # Log price regularly with P(up) (every 10 events to show activity without spam)
        if not hasattr(self, '_price_log_count'):
            self._price_log_count = 0
        self._price_log_count += 1
        if self._price_log_count % 10 == 0 and self._current_price and feature_row is not None:
            # Generate signal to get P(up) for logging
            try:
                result = self.signal_engine.generate_signal(feature_row, self._current_price)
                proba_up = result.get("proba_up", 0.0)
                logger.info(f"💰 Price: {self._current_price:.2f} | P(up)={proba_up:.4f} | Source: {event.get('source', 'unknown')}")
            except Exception as e:
                logger.debug(f"Could not compute P(up) for price log: {e}")
                logger.info(f"💰 Price: {self._current_price:.2f} | Source: {event.get('source', 'unknown')}")
        
        # Warmup just completed - send notification
        if not self._warmup_notified:
            self._warmup_notified = True
            logger.info("🚀 WARMUP COMPLETE - Starting signal generation")
            self.telegram_bot.send_alert(
                "🚀 Signal Engine Ready (Warmup Complete)",
                f"Symbol: {self.resolver.display_name()}\n"
                f"Mode: {self.ws_mode.value}\n"
                f"Model: y_tb_60\n"
                f"Thresholds: L≥{self.threshold_long}, S≤{self.threshold_short}\n"
                f"Risk: {self.risk_pct*100:.2f}%\n"
                f"Bars collected: {self.feature_buffer.get_bar_count()}"
            )
        
        # Generate signal if features ready
        if feature_row is not None:
            self._process_signal(feature_row, event["timestamp"])
    
    def _process_signal(self, feature_row, timestamp):
        """Process features and potentially generate a signal."""
        # Generate signal with current price for TP/SL calculation
        result = self.signal_engine.generate_signal(
            feature_row, 
            timestamp,
            current_price=self._current_price
        )
        signal = result["signal"]
        proba_up = result["proba_up"]
        tp = result.get("tp")
        sl = result.get("sl")
        
        # Log signal (reduce noise for FLAT signals)
        price_str = f"{self._current_price:.2f}" if self._current_price else "N/A"
        tp_str = f"{tp:.2f}" if tp else "N/A"
        sl_str = f"{sl:.2f}" if sl else "N/A"
        conviction = result.get("conviction", 1.0)
        conviction_str = f"Conviction={conviction:.2f}" if conviction < 1.0 else ""
        
        # Skip FLAT signals (don't log every FLAT to reduce noise)
        if signal == "FLAT":
            # Only log FLAT occasionally (every 100th) to show system is working
            if not hasattr(self, '_flat_count'):
                self._flat_count = 0
            self._flat_count += 1
            if self._flat_count % 100 == 0:
                logger.debug(f"Signal: {signal} | P(up)={proba_up:.4f} | Price={price_str} (processed {self._flat_count} FLAT signals)")
            return
        
        # Always log non-FLAT signals
        logger.info(f"🎯 Signal: {signal} | P(up)={proba_up:.4f} | Price={price_str} | TP={tp_str} | SL={sl_str} {conviction_str}")
        
        # Check risk rules with confidence-based gating
        decision = self.risk_guard.check_signal(signal, proba_up, timestamp)
        
        if decision.allow:
            self._signals_generated += 1
            self._last_signal_time = timestamp
            
            # Record signal for change-filtering
            self.risk_guard.record_signal(signal, timestamp)
            
            # Determine confidence display
            if signal == "LONG":
                confidence = proba_up
            else:
                confidence = 1 - proba_up
            
            # Get conviction score (if available)
            conviction = result.get("conviction", 1.0)
            
            # Adjust risk based on conviction (reduce risk for low-conviction signals)
            adjusted_risk = self.risk_pct * conviction if conviction > 0 else self.risk_pct
            
            # Build extra info with conviction warning if needed
            extra_info = {
                "Cooldown": f"{decision.cooldown_s}s",
                "DD": f"{self.risk_guard.get_status()['drawdown_pct']*100:.2f}%",
            }
            if conviction < 1.0:
                extra_info["⚠️ Conviction"] = f"{conviction:.0%} (Reduced Risk)"
            
            # Send Telegram notification with TP/SL
            self.telegram_bot.send_signal(
                signal=signal,
                proba_up=proba_up,
                timestamp=timestamp,
                price=self._current_price,
                tp=tp,
                sl=sl,
                risk_pct=adjusted_risk,  # Use adjusted risk
                extra_info=extra_info
            )
            
            logger.info(f"🔔 SIGNAL SENT: {signal} @ {self._current_price:.2f} | TP={tp_str} | SL={sl_str}")
        else:
            logger.debug(f"Signal blocked: {decision.reason}")
    
    def get_status(self) -> dict:
        """Get current runner status."""
        return {
            "running": self._running,
            "events_received": self._events_received,
            "signals_generated": self._signals_generated,
            "warmup": self.feature_buffer.get_warmup_progress(),
            "current_price": self._current_price,
            "risk_status": self.risk_guard.get_status(),
        }


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Live XAUUSD Signal Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to trained model artifact"
    )
    
    parser.add_argument(
        "--threshold_long",
        type=float,
        default=0.75,
        help="Probability threshold for LONG signals"
    )
    
    parser.add_argument(
        "--threshold_short",
        type=float,
        default=0.20,
        help="Probability threshold for SHORT signals"
    )
    
    parser.add_argument(
        "--risk_pct",
        type=float,
        default=0.01,  # 1% risk per trade
        help="Risk per trade as fraction of equity"
    )
    
    parser.add_argument(
        "--start_balance",
        type=float,
        default=25000.0,
        help="Starting account balance"
    )
    
    parser.add_argument(
        "--backfill",
        action="store_true",
        default=True,
        help="Enable REST API backfill (default: True)"
    )
    
    parser.add_argument(
        "--no-backfill",
        action="store_true",
        help="Disable backfill, use live warmup only"
    )
    
    parser.add_argument(
        "--backfill_bars",
        type=int,
        default=500,
        help="Number of bars to backfill"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Acquire singleton lock - prevent multiple instances
    if not acquire_singleton_lock():
        logger.error("Cannot start: another instance is already running")
        logger.error("Kill existing process or delete .live_runner.lock file")
        sys.exit(1)
    
    # Register cleanup on exit
    atexit.register(release_singleton_lock)
    
    # Load config from environment
    config = load_config()
    
    # Validate model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    
    # Determine backfill setting
    do_backfill = args.backfill and not args.no_backfill
    
    # Create runner
    runner = LiveRunner(
        config=config,
        model_path=str(model_path),
        threshold_long=args.threshold_long,
        threshold_short=args.threshold_short,
        risk_pct=args.risk_pct,
        start_balance=args.start_balance,
        backfill=do_backfill,
        backfill_bars=args.backfill_bars,
    )
    
    # Setup graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        runner.stop()
        release_singleton_lock()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start runner
    try:
        runner.start()
    finally:
        release_singleton_lock()


if __name__ == "__main__":
    main()
