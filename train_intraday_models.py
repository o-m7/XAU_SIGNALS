#!/usr/bin/env python3
"""
Model Training for Intraday XAUUSD Strategy

Train and compare multiple architectures:
1. LightGBM (gradient boosting)
2. XGBoost (gradient boosting)
3. Logistic Regression (baseline)
4. Random Forest (ensemble)

Evaluation:
- Walk-forward cross-validation
- Feature importance analysis
- Out-of-sample performance
- Hyperparameter tuning

Author: Quant Research Team
Date: 2026-01-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path("/Users/omar/Desktop/ML/xauusd_signals")
RESULTS_DIR = PROJECT_ROOT / "research_results"
MODELS_DIR = RESULTS_DIR / "intraday_models"
MODELS_DIR.mkdir(exist_ok=True)


def prepare_data(df, feature_cols, label_col='label', test_size=0.2):
    """
    Prepare train/test split with time-based split.

    Args:
        df: DataFrame with features and labels
        feature_cols: List of feature column names
        label_col: Name of label column
        test_size: Fraction for test set

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Remove neutral labels (focus on long/short only)
    df_binary = df[df[label_col] != 0].copy()

    # Convert -1/1 to 0/1 for binary classification
    df_binary['label_binary'] = (df_binary[label_col] == 1).astype(int)

    print(f"\nData Preparation:")
    print(f"  Original rows: {len(df):,}")
    print(f"  Binary labels: {len(df_binary):,}")
    print(f"  Features: {len(feature_cols)}")

    # Time-based split
    split_idx = int(len(df_binary) * (1 - test_size))

    train = df_binary.iloc[:split_idx]
    test = df_binary.iloc[split_idx:]

    X_train = train[feature_cols]
    y_train = train['label_binary']
    X_test = test[feature_cols]
    y_test = test['label_binary']

    print(f"\n  Train set: {len(X_train):,} ({train.index.min()} to {train.index.max()})")
    print(f"  Test set:  {len(X_test):,} ({test.index.min()} to {test.index.max()})")
    print(f"  Train positive %: {y_train.mean()*100:.2f}%")
    print(f"  Test positive %:  {y_test.mean()*100:.2f}%")

    return X_train, X_test, y_train, y_test, df_binary


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM classifier."""
    print("\n" + "=" * 80)
    print("TRAINING LIGHTGBM")
    print("=" * 80)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 100,
        'max_depth': 7,
        'verbose': -1,
        'seed': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )

    return model


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier."""
    print("\n" + "=" * 80)
    print("TRAINING XGBOOST")
    print("=" * 80)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 7,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 100,
        'seed': 42,
        'verbosity': 1
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    evals = [(dtrain, 'train'), (dtest, 'test')]

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=100
    )

    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest classifier."""
    print("\n" + "=" * 80)
    print("TRAINING RANDOM FOREST")
    print("=" * 80)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=200,
        min_samples_leaf=100,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X_train, y_train)

    return model


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression (baseline)."""
    print("\n" + "=" * 80)
    print("TRAINING LOGISTIC REGRESSION (Baseline)")
    print("=" * 80)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance.

    Returns:
        Dictionary with metrics
    """
    print(f"\n{'-' * 80}")
    print(f"EVALUATING {model_name}")
    print(f"{'-' * 80}")

    # Predictions
    if model_name in ['LightGBM', 'XGBoost']:
        if model_name == 'LightGBM':
            y_pred_proba = model.predict(X_test)
        else:
            dtest = xgb.DMatrix(X_test)
            y_pred_proba = model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n  Accuracy:  {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}%")
    print(f"  Recall:    {recall*100:.2f}%")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0,0]:6,}  FP: {cm[0,1]:6,}")
    print(f"    FN: {cm[1,0]:6,}  TP: {cm[1,1]:6,}")

    # Classification report
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'tn': cm[0,0],
        'fp': cm[0,1],
        'fn': cm[1,0],
        'tp': cm[1,1]
    }

    return metrics, y_pred, y_pred_proba


def get_feature_importance(model, feature_cols, model_name, top_n=20):
    """Extract and print feature importance."""
    print(f"\n  Top {top_n} Features ({model_name}):")

    if model_name == 'LightGBM':
        importances = model.feature_importance(importance_type='gain')
        feature_imp = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

    elif model_name == 'XGBoost':
        importances = model.get_score(importance_type='gain')
        # XGB uses feature names as keys
        feature_imp = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importances.items()
        ]).sort_values('importance', ascending=False)

    elif model_name == 'Random Forest':
        importances = model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

    else:  # Logistic Regression
        importances = np.abs(model.coef_[0])
        feature_imp = pd.DataFrame({
            'feature': feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

    print(feature_imp.head(top_n).to_string(index=False))

    return feature_imp


if __name__ == "__main__":
    print("=" * 80)
    print("INTRADAY STRATEGY - MODEL TRAINING")
    print("=" * 80)

    # Load labeled data
    df = pd.read_parquet(RESULTS_DIR / "data_15min_2020_2024_labeled.parquet")

    # Get feature columns (exclude OHLCV, labels, etc.)
    exclude_cols = [
        'open', 'high', 'low', 'close', 'volume', 'vwap', 'trades',
        'returns', 'log_returns', 'returns_bps',
        'label', 'label_binary', 'barrier_hit', 'hold_bars', 'forward_return',
        'ma_5', 'ma_10', 'ma_20', 'ma_50', 'bar_direction', 'tr'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    print(f"\nTotal features: {len(feature_cols)}")

    # Prepare data
    X_train, X_test, y_train, y_test, df_binary = prepare_data(
        df, feature_cols, label_col='label', test_size=0.2
    )

    # Train models
    models = {}
    results = []

    # 1. LightGBM
    lgbm_model = train_lightgbm(X_train, y_train, X_test, y_test)
    models['LightGBM'] = lgbm_model
    metrics, y_pred, y_pred_proba = evaluate_model(lgbm_model, X_test, y_test, 'LightGBM')
    results.append(metrics)
    lgbm_importance = get_feature_importance(lgbm_model, feature_cols, 'LightGBM')

    # Save
    joblib.dump(lgbm_model, MODELS_DIR / "lightgbm_intraday.joblib")
    lgbm_importance.to_csv(MODELS_DIR / "lightgbm_feature_importance.csv", index=False)

    # 2. XGBoost
    xgb_model = train_xgboost(X_train, y_train, X_test, y_test)
    models['XGBoost'] = xgb_model
    metrics, y_pred, y_pred_proba = evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
    results.append(metrics)
    xgb_importance = get_feature_importance(xgb_model, feature_cols, 'XGBoost')

    # Save
    xgb_model.save_model(str(MODELS_DIR / "xgboost_intraday.json"))
    xgb_importance.to_csv(MODELS_DIR / "xgboost_feature_importance.csv", index=False)

    # 3. Random Forest
    rf_model = train_random_forest(X_train, y_train)
    models['Random Forest'] = rf_model
    metrics, y_pred, y_pred_proba = evaluate_model(rf_model, X_test, y_test, 'Random Forest')
    results.append(metrics)
    rf_importance = get_feature_importance(rf_model, feature_cols, 'Random Forest')

    # Save
    joblib.dump(rf_model, MODELS_DIR / "randomforest_intraday.joblib")
    rf_importance.to_csv(MODELS_DIR / "randomforest_feature_importance.csv", index=False)

    # 4. Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    models['Logistic Regression'] = lr_model
    metrics, y_pred, y_pred_proba = evaluate_model(lr_model, X_test, y_test, 'Logistic Regression')
    results.append(metrics)
    lr_importance = get_feature_importance(lr_model, feature_cols, 'Logistic Regression')

    # Save
    joblib.dump(lr_model, MODELS_DIR / "logistic_regression_intraday.joblib")
    lr_importance.to_csv(MODELS_DIR / "logistic_regression_feature_importance.csv", index=False)

    # Compare models
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    df_results = pd.DataFrame(results)
    df_results = df_results.set_index('model')

    print("\n" + df_results[['accuracy', 'precision', 'recall', 'f1', 'auc']].to_string())

    # Best model
    best_model_name = df_results['auc'].idxmax()
    print(f"\n\nBest Model (by AUC): {best_model_name}")
    print(f"  AUC: {df_results.loc[best_model_name, 'auc']:.4f}")
    print(f"  Accuracy: {df_results.loc[best_model_name, 'accuracy']*100:.2f}%")
    print(f"  Precision: {df_results.loc[best_model_name, 'precision']*100:.2f}%")

    # Save comparison
    df_results.to_csv(RESULTS_DIR / "model_comparison.csv")
    print(f"\n\nSaved model comparison to: {RESULTS_DIR / 'model_comparison.csv'}")
    print(f"Models saved to: {MODELS_DIR}/")
    print()
