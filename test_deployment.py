#!/usr/bin/env python3
"""
Pre-Deployment Test Script

Validates that all production models are ready for deployment.

Tests:
1. Model files exist and load correctly
2. Features are consistent
3. Predictions work on test data
4. Thresholds are configured correctly
5. Expected performance metrics

Run before deploying to Railway!
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from models_config_production import get_production_models, get_model_summary
from src.live.signal_engine import SignalEngine

def test_model_files():
    """Test 1: All model files exist."""
    print("=" * 80)
    print("TEST 1: MODEL FILES")
    print("=" * 80)

    models = get_production_models()
    all_exist = True

    for model_config in models:
        exists = model_config.model_path.exists()
        status = "✅" if exists else "❌"
        size_mb = model_config.model_path.stat().st_size / (1024*1024) if exists else 0

        print(f"{status} {model_config.name}")
        print(f"   Path: {model_config.model_path}")
        print(f"   Size: {size_mb:.2f} MB")
        print()

        all_exist = all_exist and exists

    if all_exist:
        print("✅ ALL MODEL FILES FOUND")
    else:
        print("❌ SOME MODEL FILES MISSING")
        return False

    return True


def test_model_loading():
    """Test 2: Models load correctly."""
    print("\n" + "=" * 80)
    print("TEST 2: MODEL LOADING")
    print("=" * 80)

    models = get_production_models()
    all_loaded = True

    for model_config in models:
        try:
            artifact = joblib.load(model_config.model_path)
            model = artifact.get('model') if isinstance(artifact, dict) else artifact
            features = artifact.get('features', []) if isinstance(artifact, dict) else []

            print(f"✅ {model_config.name}")
            print(f"   Model type: {type(model).__name__}")
            print(f"   Features: {len(features)}")
            print(f"   Thresholds: {model_config.threshold_long}/{model_config.threshold_short}")
            print()

        except Exception as e:
            print(f"❌ {model_config.name}")
            print(f"   Error: {e}")
            print()
            all_loaded = False

    if all_loaded:
        print("✅ ALL MODELS LOADED SUCCESSFULLY")
    else:
        print("❌ SOME MODELS FAILED TO LOAD")
        return False

    return True


def test_signal_generation():
    """Test 3: Models generate signals on test data."""
    print("\n" + "=" * 80)
    print("TEST 3: SIGNAL GENERATION")
    print("=" * 80)

    models = get_production_models()
    all_generated = True

    for model_config in models:
        try:
            engine = SignalEngine(
                model_path=str(model_config.model_path),
                threshold_long=model_config.threshold_long,
                threshold_short=model_config.threshold_short
            )

            # Create mock feature row
            features = engine.get_feature_list()
            mock_row = pd.DataFrame({
                f: [np.random.randn()] for f in features
            })

            # Generate signal
            result = engine.generate_signal(mock_row, current_price=2650.0)

            print(f"✅ {model_config.name}")
            print(f"   Signal: {result['signal']}")
            print(f"   P(up): {result['proba_up']:.4f}")
            print(f"   TP/SL: {result.get('tp')}/{result.get('sl')}")
            print()

        except Exception as e:
            print(f"❌ {model_config.name}")
            print(f"   Error: {e}")
            print()
            all_generated = False

    if all_generated:
        print("✅ ALL MODELS GENERATE SIGNALS")
    else:
        print("❌ SOME MODELS FAILED")
        return False

    return True


def test_configuration():
    """Test 4: Configuration is valid."""
    print("\n" + "=" * 80)
    print("TEST 4: CONFIGURATION VALIDATION")
    print("=" * 80)

    summary = get_model_summary()
    models = get_production_models()

    print(f"Deployment Date: {summary['deployment_date']}")
    print(f"Total Models: {summary['total_models']}")
    print(f"Enabled: {summary['enabled_models']}")
    print(f"Status: {summary['status']}")
    print()

    # Validate thresholds
    valid_config = True
    for model_config in models:
        if not (0.5 <= model_config.threshold_long <= 0.9):
            print(f"❌ {model_config.name}: Invalid long threshold {model_config.threshold_long}")
            valid_config = False

        if not (0.1 <= model_config.threshold_short <= 0.5):
            print(f"❌ {model_config.name}: Invalid short threshold {model_config.threshold_short}")
            valid_config = False

        if model_config.threshold_long <= model_config.threshold_short:
            print(f"❌ {model_config.name}: Long threshold must be > short threshold")
            valid_config = False

    if valid_config:
        print("✅ CONFIGURATION VALID")
    else:
        print("❌ CONFIGURATION INVALID")
        return False

    return True


def test_expected_performance():
    """Test 5: Expected performance summary."""
    print("\n" + "=" * 80)
    print("TEST 5: EXPECTED PERFORMANCE")
    print("=" * 80)

    summary = get_model_summary()

    print("Model Performance (2025 OOS):")
    print("-" * 80)

    total_trades = 0
    for model in summary['models']:
        print(f"{model['role']}: {model['name']}")
        print(f"  Win Rate: {model['wr']}")
        print(f"  Profit Factor: {model['pf']}")
        print(f"  Trades/Day: {model['trades_day']}")

        # Extract numeric trades/day
        trades_day = float(model['trades_day'])
        total_trades += trades_day
        print()

    print(f"Combined Trades/Day: ~{total_trades:.1f}")
    print()

    if total_trades >= 2:
        print(f"✅ MEETS MINIMUM FREQUENCY (2+ trades/day)")
    else:
        print(f"❌ BELOW MINIMUM FREQUENCY")
        return False

    return True


def main():
    print("=" * 80)
    print("PRE-DEPLOYMENT TEST SUITE")
    print("=" * 80)
    print()

    tests = [
        ("Model Files", test_model_files),
        ("Model Loading", test_model_loading),
        ("Signal Generation", test_signal_generation),
        ("Configuration", test_configuration),
        ("Expected Performance", test_expected_performance),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ {test_name} FAILED WITH EXCEPTION:")
            print(f"   {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")
    print()

    if passed == total:
        print("🎉 " + "=" * 74)
        print("🎉 ALL TESTS PASSED - READY FOR DEPLOYMENT!")
        print("🎉 " + "=" * 74)
        print()
        print("Next steps:")
        print("1. Review DEPLOYMENT_GUIDE.md")
        print("2. Set environment variables in Railway")
        print("3. Deploy: railway up")
        print("4. Monitor logs: railway logs")
        return 0
    else:
        print("❌ " + "=" * 74)
        print("❌ SOME TESTS FAILED - FIX BEFORE DEPLOYING")
        print("❌ " + "=" * 74)
        return 1


if __name__ == "__main__":
    sys.exit(main())
