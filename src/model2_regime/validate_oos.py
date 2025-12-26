"""
Out-of-Sample Validation for Model #2 Regime Classifier

Tests the regime classifier on December 2025 data with:
- Regime prediction accuracy
- Regime transition analysis
- Performance comparison to Model #1
- Signal quality by regime
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model2_regime.order_flow_features import build_order_flow_features
from src.model2_regime.regime_features import build_regime_features
from src.model2_regime.regime_labeling import add_regime_labels, REGIME_NAMES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = PROJECT_ROOT / "models" / "model2_regime"


def load_regime_classifier():
    """Load trained regime classifier."""
    model_path = MODELS_DIR / "regime_classifier.joblib"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Run src/model2_regime/train_regime_classifier.py first")
        return None
    
    logger.info(f"Loading model from: {model_path}")
    artifact = joblib.load(model_path)
    
    logger.info(f"Model trained: {artifact.get('trained_at', 'unknown')}")
    logger.info(f"Features: {len(artifact['features'])}")
    logger.info(f"Training accuracy: {artifact.get('val_accuracy', 0):.4f}")
    
    return artifact


def load_oos_data(start_date: str = "2025-12-01", end_date: str = "2025-12-22"):
    """Load out-of-sample data."""
    logger.info(f"\nLoading OOS data: {start_date} to {end_date}")
    
    features_path = MODELS_DIR / "regime_features_5min.parquet"
    
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        return None
    
    df = pd.read_parquet(features_path)
    df_oos = df.loc[start_date:end_date].copy()
    
    logger.info(f"Loaded {len(df_oos):,} 5-minute bars")
    logger.info(f"Date range: {df_oos.index.min()} to {df_oos.index.max()}")
    
    return df_oos


def evaluate_regime_predictions(df: pd.DataFrame, model, features: list):
    """Evaluate regime predictions vs ground truth labels."""
    logger.info("\n" + "="*80)
    logger.info("REGIME PREDICTION ACCURACY (December 2025 OOS)")
    logger.info("="*80)
    
    # Prepare data
    X = df[features].values
    y_true = df['regime'].values
    
    # Predict
    y_pred = model.predict(X)
    
    # Add predictions to dataframe
    df['regime_pred'] = y_pred
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    logger.info(f"\nOverall Accuracy: {accuracy:.4f}")
    
    # Get actual classes
    actual_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
    actual_names = [REGIME_NAMES[c] for c in actual_classes]
    
    # Classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_true, y_pred, labels=actual_classes, target_names=actual_names))
    
    # Confusion matrix
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    logger.info(f"\n{cm}")
    logger.info("\nRows: True regime | Columns: Predicted regime")
    logger.info("Order: " + ", ".join(actual_names))
    
    return df


def analyze_regime_transitions(df: pd.DataFrame):
    """Analyze how regimes transition over time."""
    logger.info("\n" + "="*80)
    logger.info("REGIME TRANSITION ANALYSIS")
    logger.info("="*80)
    
    # Count transitions
    df['regime_changed'] = (df['regime_pred'] != df['regime_pred'].shift(1))
    transitions = df['regime_changed'].sum()
    avg_duration = len(df) / transitions if transitions > 0 else len(df)
    
    logger.info(f"\nTotal regime transitions: {transitions}")
    logger.info(f"Average regime duration: {avg_duration:.1f} bars (~{avg_duration*5:.0f} minutes)")
    
    # Transition matrix (from regime X to regime Y)
    prev_regime = df['regime_pred'].shift(1)
    transition_pairs = pd.DataFrame({
        'from': prev_regime,
        'to': df['regime_pred']
    }).dropna()
    
    logger.info("\nTop 10 Regime Transitions:")
    transition_counts = transition_pairs.groupby(['from', 'to']).size().sort_values(ascending=False)
    for (from_r, to_r), count in transition_counts.head(10).items():
        from_name = REGIME_NAMES.get(from_r, str(from_r))
        to_name = REGIME_NAMES.get(to_r, str(to_r))
        logger.info(f"  {from_name:25} -> {to_name:25}: {count:4d} times")
    
    # Regime distribution
    logger.info("\nRegime Distribution (Predicted):")
    regime_dist = df['regime_pred'].value_counts()
    for regime_id, count in regime_dist.items():
        regime_name = REGIME_NAMES.get(regime_id, str(regime_id))
        pct = count / len(df) * 100
        logger.info(f"  {regime_name:25}: {count:5,} bars ({pct:5.1f}%)")
    
    return df


def analyze_model1_performance_by_regime(df: pd.DataFrame):
    """
    Analyze how Model #1 (triple-barrier) performs in different regimes.
    
    This shows which regimes Model #1 struggles with.
    """
    logger.info("\n" + "="*80)
    logger.info("MODEL #1 SIGNAL QUALITY BY REGIME")
    logger.info("="*80)
    
    # Load Model #1
    model1_path = PROJECT_ROOT / "models" / "y_tb_60_hgb_tuned.joblib"
    if not model1_path.exists():
        logger.warning("Model #1 not found, skipping this analysis")
        return df
    
    model1_artifact = joblib.load(model1_path)
    model1 = model1_artifact['model']
    model1_features = model1_artifact['features']
    
    # Map 5-min features to model1 features (need to aggregate 1-min data)
    logger.info("\nNote: Model #1 uses 1-minute features, this is approximate")
    
    # For simplicity, compute basic features from 5-min bars
    df['ret_1'] = df['close'].pct_change()
    df['vol_10'] = df['ret_1'].rolling(10).std()
    
    # Check which features are available
    available_features = [f for f in model1_features if f in df.columns]
    logger.info(f"Available Model #1 features: {len(available_features)}/{len(model1_features)}")
    
    if len(available_features) < 10:
        logger.warning("Too few features available for Model #1 analysis")
        return df
    
    # Predict with Model #1
    X_model1 = df[available_features].fillna(0).values
    proba_model1 = model1.predict_proba(X_model1)[:, 1]
    df['model1_proba'] = proba_model1
    
    # Analyze by regime
    logger.info("\nModel #1 Probability Distribution by Regime:")
    for regime_id in sorted(df['regime_pred'].unique()):
        regime_name = REGIME_NAMES.get(regime_id, str(regime_id))
        regime_mask = df['regime_pred'] == regime_id
        proba_regime = proba_model1[regime_mask]
        
        logger.info(f"\n{regime_name}:")
        logger.info(f"  Mean P(up): {proba_regime.mean():.4f}")
        logger.info(f"  Std P(up):  {proba_regime.std():.4f}")
        logger.info(f"  Min/Max:    [{proba_regime.min():.4f}, {proba_regime.max():.4f}]")
        logger.info(f"  >= 0.70:    {(proba_regime >= 0.70).sum():5d} bars ({(proba_regime >= 0.70).mean()*100:5.1f}%)")
        logger.info(f"  <= 0.20:    {(proba_regime <= 0.20).sum():5d} bars ({(proba_regime <= 0.20).mean()*100:5.1f}%)")
    
    return df


def analyze_volatility_by_regime(df: pd.DataFrame):
    """Analyze volatility characteristics by regime."""
    logger.info("\n" + "="*80)
    logger.info("VOLATILITY ANALYSIS BY REGIME")
    logger.info("="*80)
    
    df['returns'] = df['close'].pct_change()
    df['realized_vol'] = df['returns'].rolling(20).std()
    
    logger.info("\nVolatility Statistics by Regime:")
    for regime_id in sorted(df['regime_pred'].unique()):
        regime_name = REGIME_NAMES.get(regime_id, str(regime_id))
        regime_mask = df['regime_pred'] == regime_id
        vol_regime = df.loc[regime_mask, 'realized_vol']
        
        logger.info(f"\n{regime_name}:")
        logger.info(f"  Mean vol: {vol_regime.mean():.6f}")
        logger.info(f"  Median:   {vol_regime.median():.6f}")
        logger.info(f"  Std:      {vol_regime.std():.6f}")


def main():
    logger.info("="*80)
    logger.info("MODEL #2 REGIME CLASSIFIER - OUT-OF-SAMPLE VALIDATION")
    logger.info("="*80)
    
    # Load model
    artifact = load_regime_classifier()
    if artifact is None:
        return
    
    model = artifact['model']
    features = artifact['features']
    
    # Load OOS data
    df = load_oos_data(start_date="2025-12-01", end_date="2025-12-22")
    if df is None:
        return
    
    # Evaluate predictions
    df = evaluate_regime_predictions(df, model, features)
    
    # Analyze transitions
    df = analyze_regime_transitions(df)
    
    # Analyze Model #1 performance by regime
    df = analyze_model1_performance_by_regime(df)
    
    # Analyze volatility by regime
    analyze_volatility_by_regime(df)
    
    # Save results
    if df is not None:
        output_path = MODELS_DIR / "regime_predictions_dec2025.parquet"
        df.to_parquet(output_path)
        logger.info(f"\n\nSaved predictions to: {output_path}")
    
    logger.info("\n" + "="*80)
    logger.info("VALIDATION COMPLETE")
    logger.info("="*80)
    logger.info("\nKey Findings:")
    logger.info("1. Check regime prediction accuracy above")
    logger.info("2. Review regime transitions for stability")
    logger.info("3. Note which regimes Model #1 performs best/worst in")
    logger.info("4. Use this to build regime-specific strategies in Phase 2")


if __name__ == "__main__":
    main()

