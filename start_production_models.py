#!/usr/bin/env python3
"""
Production Multi-Model Signal Runner - REBUILT MODELS (Phase 5 Complete)

Deploys all 3 rebuilt production models simultaneously:
- Model 1: Microstructure HGB (72.8% WR)
- Model 3: CMF/MACD HGB (71.2% WR)
- Model RF: Random Forest (69.3% WR)

Each model generates independent signals sent to Telegram.

Usage:
    cd /Users/omar/Desktop/ML/xauusd_signals
    source venv/bin/activate
    python start_production_models.py

    # Or specify which models to run:
    python start_production_models.py --models model1_rebuilt model3_rebuilt

    # Test mode (no Telegram):
    python start_production_models.py --test

    # No backfill (use live warmup only):
    python start_production_models.py --no-backfill
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.live.live_runner import main as run_live_engine
from models_config_production import get_production_models, get_model_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("production_runner.log"),
    ]
)
logger = logging.getLogger("ProductionRunner")


def main():
    parser = argparse.ArgumentParser(description="Production Multi-Model Signal Runner")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to run (default: all enabled)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode (no Telegram notifications)"
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        default=True,
        help="Enable REST API backfill (default: True, for backward compatibility)"
    )
    parser.add_argument(
        "--no-backfill",
        action="store_true",
        help="Disable REST API backfill (use live warmup only)"
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

    # Print configuration
    print("=" * 80)
    print("🚀 PRODUCTION MULTI-MODEL DEPLOYMENT - REBUILT MODELS")
    print("=" * 80)
    print()

    summary = get_model_summary()
    print(f"Deployment Date: {summary['deployment_date']}")
    print(f"Phase: {summary['phase']}")
    print(f"Validation: {summary['validation_method']}")
    print(f"Average Win Rate: {summary['average_win_rate']*100:.1f}%")
    print(f"Status: {summary['status']}")
    print(f"Confidence: {summary['confidence']}")
    print()

    # Get models
    all_models = get_production_models()

    if args.models:
        # Filter to specified models
        model_names = set(args.models)
        models = [m for m in all_models if m.name in model_names]
        if not models:
            logger.error(f"No models found matching: {args.models}")
            logger.info(f"Available models: {[m.name for m in all_models]}")
            sys.exit(1)
    else:
        models = all_models

    print(f"Running {len(models)} models:")
    print("-" * 80)
    for model in models:
        model_info = next((m for m in summary['models'] if m['name'] == model.name), None)
        if model_info:
            print(f"  • {model.name} ({model_info['type']})")
            print(f"      Features:  {model_info['features']}")
            print(f"      Avg WR:    {model_info['avg_wr']}")
            print(f"      Threshold: {model_info['threshold']}")
            print(f"      Signals:   {model_info['signals_day']}")
        else:
            print(f"  • {model.name}")
            print(f"      Thresholds: LONG≥{model.threshold_long}, SHORT≤{model.threshold_short}")
    print()
    print(f"Expected Combined: {summary['expected_signals_per_day']}")
    print()

    if args.test:
        print("⚠️  TEST MODE: Telegram notifications disabled")
        os.environ['TELEGRAM_ENABLED'] = '0'
        print()

    print("=" * 80)
    print("STARTING LIVE RUNNER")
    print("=" * 80)
    print()

    # Run the live engine
    try:
        # Load configuration
        from src.live.live_runner import load_config, LiveRunner

        config = load_config()

        # Verify model files exist
        missing_models = []
        for model in models:
            if not Path(model.model_path).exists():
                missing_models.append(model.model_path)

        if missing_models:
            logger.error("Missing model files:")
            for path in missing_models:
                logger.error(f"  - {path}")
            sys.exit(1)

        # Create LiveRunner with production models
        logger.info(f"Initializing LiveRunner with {len(models)} production models...")
        for model in models:
            logger.info(f"  • {model.name}: LONG≥{model.threshold_long}, SHORT≤{model.threshold_short}")

        runner = LiveRunner(
            config=config,
            risk_pct=0.01,  # 1% risk per trade
            start_balance=25000.0,  # $25k starting balance
            backfill=not args.no_backfill,  # Default to backfill enabled
            backfill_bars=500,
            production_models=models,  # Pass rebuilt production models
            model_version="v3"  # Use v3 for multi-model system
        )

        # Start the runner
        logger.info("Starting live runner...")
        runner.start()

    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
