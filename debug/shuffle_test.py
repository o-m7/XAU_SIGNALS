#!/usr/bin/env python3
"""
Shuffled Data Test for Leakage Detection.

If the model performs well on randomly shuffled data, there's leakage.
A legitimate model should perform near random on shuffled data.

Usage:
    cd /Users/omar/Desktop/ML && source xauusd_signals/venv/bin/activate && python xauusd_signals/debug/shuffle_test.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score

PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2024.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "y_tb_60_hgb_tuned.joblib"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TARGET = "y_tb_60"


def main():
    print("=" * 70)
    print("  SHUFFLED DATA TEST FOR LEAKAGE")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    features = artifact["features"]
    print(f"  Model loaded with {len(features)} features")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(FEATURES_PATH)
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
    df = df.sort_index()
    
    # Prepare data
    df_clean = df.dropna(subset=features + [TARGET])
    df_clean = df_clean[df_clean[TARGET] != 0]
    
    n = len(df_clean)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    # Get val/test sets
    df_val = df_clean.iloc[train_end:val_end]
    df_test = df_clean.iloc[val_end:]
    
    X_val = df_val[features].to_numpy()
    y_val = df_val[TARGET].to_numpy()
    X_test = df_test[features].to_numpy()
    y_test = df_test[TARGET].to_numpy()
    
    # Map labels to binary
    y_val_bin = (y_val == 1).astype(int)
    y_test_bin = (y_test == 1).astype(int)
    
    print(f"\n  Val samples: {len(y_val):,}")
    print(f"  Test samples: {len(y_test):,}")
    
    # TEST 1: Original data performance
    print("\n" + "-" * 50)
    print("TEST 1: ORIGINAL DATA (control)")
    print("-" * 50)
    
    proba_val = model.predict_proba(X_val)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]
    
    auc_val = roc_auc_score(y_val_bin, proba_val)
    auc_test = roc_auc_score(y_test_bin, proba_test)
    
    print(f"\n  Val AUC:  {auc_val:.4f}")
    print(f"  Test AUC: {auc_test:.4f}")
    
    # TEST 2: Shuffled labels
    print("\n" + "-" * 50)
    print("TEST 2: SHUFFLED LABELS (should be ~0.50 AUC)")
    print("-" * 50)
    print("  If features leak future info, AUC should still be high")
    
    np.random.seed(42)
    y_val_shuffled = np.random.permutation(y_val_bin)
    y_test_shuffled = np.random.permutation(y_test_bin)
    
    auc_val_shuf = roc_auc_score(y_val_shuffled, proba_val)
    auc_test_shuf = roc_auc_score(y_test_shuffled, proba_test)
    
    print(f"\n  Val AUC (shuffled labels):  {auc_val_shuf:.4f}")
    print(f"  Test AUC (shuffled labels): {auc_test_shuf:.4f}")
    
    if abs(auc_val_shuf - 0.5) < 0.05 and abs(auc_test_shuf - 0.5) < 0.05:
        print("\n  ✓ As expected: random labels → ~0.50 AUC")
        print("    Model predictions are NOT correlated with random labels")
    else:
        print("\n  ⚠ Unexpected: predictions correlate with random labels")
    
    # TEST 3: Shuffled features (break temporal structure)
    print("\n" + "-" * 50)
    print("TEST 3: SHUFFLED FEATURES (should be ~0.50 AUC)")
    print("-" * 50)
    print("  If model relies on leaked future info, AUC will drop")
    
    np.random.seed(42)
    X_val_shuffled = X_val[np.random.permutation(len(X_val))]
    X_test_shuffled = X_test[np.random.permutation(len(X_test))]
    
    proba_val_shuf = model.predict_proba(X_val_shuffled)[:, 1]
    proba_test_shuf = model.predict_proba(X_test_shuffled)[:, 1]
    
    auc_val_feat_shuf = roc_auc_score(y_val_bin, proba_val_shuf)
    auc_test_feat_shuf = roc_auc_score(y_test_bin, proba_test_shuf)
    
    print(f"\n  Val AUC (shuffled features):  {auc_val_feat_shuf:.4f}")
    print(f"  Test AUC (shuffled features): {auc_test_feat_shuf:.4f}")
    
    if abs(auc_val_feat_shuf - 0.5) < 0.05 and abs(auc_test_feat_shuf - 0.5) < 0.05:
        print("\n  ✓ As expected: shuffled features → ~0.50 AUC")
        print("    Model uses temporal structure (not leakage)")
    else:
        print("\n  ⚠ WARNING: shuffled features still predictive")
        print("    Could indicate feature leakage")
    
    # TEST 4: Randomized X-y pairs (both shuffled together)
    print("\n" + "-" * 50)
    print("TEST 4: JOINTLY SHUFFLED X-y (should be ~0.50 AUC)")
    print("-" * 50)
    
    np.random.seed(42)
    shuffle_idx = np.random.permutation(len(X_val))
    X_val_joint = X_val[shuffle_idx]
    y_val_joint = y_val_bin[shuffle_idx]
    
    shuffle_idx = np.random.permutation(len(X_test))
    X_test_joint = X_test[shuffle_idx]
    y_test_joint = y_test_bin[shuffle_idx]
    
    proba_val_joint = model.predict_proba(X_val_joint)[:, 1]
    proba_test_joint = model.predict_proba(X_test_joint)[:, 1]
    
    auc_val_joint = roc_auc_score(y_val_joint, proba_val_joint)
    auc_test_joint = roc_auc_score(y_test_joint, proba_test_joint)
    
    print(f"\n  Val AUC (joint shuffle):  {auc_val_joint:.4f}")
    print(f"  Test AUC (joint shuffle): {auc_test_joint:.4f}")
    
    # This should be similar to original if X and y are jointly shuffled
    # (their relationship is preserved)
    if abs(auc_val_joint - auc_val) < 0.02:
        print("\n  ✓ Joint shuffle preserves X-y relationship")
        print("    AUC similar to original (relationship intact)")
    
    # SUMMARY
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    
    print(f"""
  Original Data:
    Val AUC:  {auc_val:.4f}
    Test AUC: {auc_test:.4f}

  Shuffled Labels (X same, y random):
    Val AUC:  {auc_val_shuf:.4f}  (expected: ~0.50)
    Test AUC: {auc_test_shuf:.4f}  (expected: ~0.50)

  Shuffled Features (X random, y same):
    Val AUC:  {auc_val_feat_shuf:.4f}  (expected: ~0.50)
    Test AUC: {auc_test_feat_shuf:.4f}  (expected: ~0.50)

  Joint Shuffle (X and y shuffled together):
    Val AUC:  {auc_val_joint:.4f}  (expected: ~original)
    Test AUC: {auc_test_joint:.4f}  (expected: ~original)
""")
    
    # Final verdict
    shuffle_test_passed = (
        abs(auc_val_shuf - 0.5) < 0.10 and
        abs(auc_test_shuf - 0.5) < 0.10 and
        abs(auc_val_feat_shuf - 0.5) < 0.10 and
        abs(auc_test_feat_shuf - 0.5) < 0.10
    )
    
    if shuffle_test_passed:
        print("  ✓ VERDICT: NO OBVIOUS LEAKAGE DETECTED")
        print("    - Shuffled labels → random predictions")
        print("    - Shuffled features → random predictions")
        print("    - Model relies on proper temporal X-y relationship")
    else:
        print("  ⚠ VERDICT: POSSIBLE LEAKAGE")
        print("    - Model predictions correlate with shuffled data")
        print("    - Further investigation needed")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

