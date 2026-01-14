#!/usr/bin/env python3
"""
Simple Training Pipeline - Models 3 and 5 only

Skip Model 1 and Model 6 due to import issues.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from model3_cmf_macd.train_model3_2014_2023 import main as train_model3_main
from models.model5.train import main as train_model5_main


def main():
    print("="*80)
    print("SIMPLE TRAINING PIPELINE")
    print("="*80)
    print("\nTraining Models 3 and 5 only (skipping Model 1 and 6)")
    print("\n1. Training Model 3 (Strict CMF/MACD)...")
    train_model3_main()
    
    print("\n2. Training Model 5 (Range Reversion)...")
    train_model5_main()
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
