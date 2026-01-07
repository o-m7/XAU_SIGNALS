#!/usr/bin/env python3
"""
Production Multi-Model Signal Runner

Deploys all validated production models simultaneously.
Each model generates independent signals sent to Telegram.

Usage:
    python start_production_models.py

    # Or specify which models to run:
    python start_production_models.py --models model3_cmf_macd_v4 model1_high_conf

    # Test mode (no Telegram):
    python start_production_models.py --test
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
        help="Backfill historical data on startup (default: True)"
    )

    args = parser.parse_args()

    # Print configuration
    print("=" * 80)
    print("PRODUCTION MULTI-MODEL DEPLOYMENT")
    print("=" * 80)
    print()

    summary = get_model_summary()
    print(f"Deployment Date: {summary['deployment_date']}")
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
    for model in models:
        model_info = next((m for m in summary['models'] if m['name'] == model.name), None)
        if model_info:
            print(f"  - {model.name} ({model_info['role']})")
            print(f"      WR: {model_info['wr']}, PF: {model_info['pf']}, Trades/Day: {model_info['trades_day']}")
        else:
            print(f"  - {model.name}")
        print(f"      Thresholds: {model.threshold_long}/{model.threshold_short}")
    print()

    if args.test:
        print("⚠️  TEST MODE: Telegram notifications disabled")
        os.environ['TELEGRAM_ENABLED'] = '0'
        print()

    # Set multi-model mode environment variable
    os.environ['MULTI_MODEL_MODE'] = '1'

    # Serialize model configs to environment
    model_configs_str = ";".join([
        f"{m.name},{m.model_path},{m.threshold_long},{m.threshold_short}"
        for m in models
    ])
    os.environ['MODEL_CONFIGS'] = model_configs_str

    print("=" * 80)
    print("STARTING LIVE RUNNER")
    print("=" * 80)
    print()

    # Run the live engine
    try:
        # Import and run with multi-model configuration
        sys.argv = [
            'live_runner',
            '--multi-model',
            '--backfill' if args.backfill else ''
        ]

        # The live_runner will read MODEL_CONFIGS from environment
        run_live_engine()

    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
