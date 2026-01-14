#!/usr/bin/env python3
"""
Master Training Script v3 - Train all 7 models
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent

TRAINING_SCRIPTS = [
    "train_model1_v3.py",
    "train_model3_v3.py",
    "train_model5_v3.py",
    "train_model6_v3.py",
    "train_model7_raw.py",
    "train_model8_momentum.py",
    "train_model9_rejection.py",
]


def main():
    print("=" * 80)
    print("MASTER TRAINING PIPELINE v3")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now()}")
    print(f"Training {len(TRAINING_SCRIPTS)} models...")
    
    scripts_dir = PROJECT_ROOT / "scripts"
    results = {}
    
    for script_name in TRAINING_SCRIPTS:
        script_path = scripts_dir / script_name
        print(f"\n{'=' * 60}")
        print(f"TRAINING: {script_name}")
        print(f"{'=' * 60}")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                timeout=1200,  # 20 min timeout per model
                capture_output=False
            )
            results[script_name] = "SUCCESS" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
        except subprocess.TimeoutExpired:
            results[script_name] = "TIMEOUT"
        except Exception as e:
            results[script_name] = f"ERROR: {e}"
    
    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    
    for script, status in results.items():
        print(f"  {script}: {status}")
    
    successful = sum(1 for s in results.values() if s == "SUCCESS")
    print(f"\nSuccessful: {successful}/{len(results)}")
    
    if successful == len(results):
        print("\nAll models trained successfully!")
        print("\nRun backtest with:")
        print("  python scripts/run_funded_backtest_v2.py")
    
    return 0 if successful == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())

