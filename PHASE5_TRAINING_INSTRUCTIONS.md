# Phase 5: Model Retraining Instructions

## Status: Ready to Execute

All code changes for Phases 1-4 are complete and committed. The training pipeline is ready to use the unified feature engineering module.

## What Will Happen

The `rebuild_from_scratch.py` script will:

1. **Load Data**: `data/features/xauusd_features_2020_2025_fixed.parquet` (522 MB, 2014-2025)

2. **Generate Features** (NEW - uses unified module):
   - Calls `build_features()` from `src.features`
   - Generates all 59 unique features
   - **CRITICAL**: Features now computed with fixed lookahead bias
   - Microstructure: `synthetic_order_flow = net_tick_pressure` (NOT * volume)

3. **Create Labels** (Triple-Barrier):
   - TP: 1.5 ATR
   - SL: 1.0 ATR
   - Max holding: 30 bars (~7.5 hours on 15-min data)
   - Min holding: 2 bars (30 min)

4. **Walk-Forward Validation** (4 periods):
   - 2022: Train on 2014-2021, test on 2022
   - 2023: Train on 2014-2022, test on 2023
   - 2024: Train on 2014-2023, test on 2024
   - 2025: Train on 2014-2024, test on 2025

5. **Train 3 Models** (if validation passes):
   - **Model 1** (model1_hgb): HistGradientBoostingClassifier, 37 microstructure features
   - **Model 3** (model3_cmf_macd): HistGradientBoostingClassifier, 31 CMF/MACD features
   - **Model RF** (model_rf): RandomForestClassifier, 37 microstructure features

## How to Run

```bash
cd /Users/omar/Desktop/ML/xauusd_signals

# Activate your Python environment (if using virtualenv/conda)
# source venv/bin/activate  # or conda activate your_env

# Run the training script
python3 rebuild_from_scratch.py
```

## Expected Output

### Phase 1: Feature Engineering
```
================================================================================
UNIFIED FEATURE ENGINEERING
================================================================================
Input bars: XXX,XXX rows
Input quotes: XXX,XXX rows
Feature set: all

Step 1: Merging quotes into bars...
  ✓ Quotes merged: XX columns

Step 2: Computing microstructure features...
  ✓ Microstructure complete: 7 features

Step 3: Computing price action features...
  ✓ Price action complete

Step 4: Computing technical indicators...
  ✓ Technical indicator features complete

Step 5: Computing regime features...
  ✓ Regime features complete

Step 6: Computing interaction features...

================================================================================
FEATURE ENGINEERING COMPLETE: 59 features, XXX,XXX rows
================================================================================
```

### Phase 2: Labeling
```
================================================================================
STEP 2: TRIPLE-BARRIER LABELING
================================================================================
Label config:
  TP: 1.5 ATR
  SL: 1.0 ATR
  Max bars: 30
  Min bars: 2

Distribution:
  WIN:     XX,XXX (XX%)
  LOSS:    XX,XXX (XX%)
  NEUTRAL: XX,XXX (XX%)
```

### Phase 3: Walk-Forward Validation (per model)
```
================================================================================
TRAINING: model1_hgb (microstructure features)
================================================================================

[FOLD 1/4] 2022-01-01 to 2022-12-31
  Train samples: XX,XXX | Test samples: X,XXX
  Training model...
  Testing at threshold 0.55:
    Win Rate: 68.5%
    Profit Factor: 1.45
    Signals: 350

[FOLD 2/4] 2023-01-01 to 2023-12-31
  ...

Average Win Rate: 67.8%
Average Profit Factor: 1.38
✅ PASSES validation (67.8% ≥ 52.0%)
```

### Phase 4: Final Model Training
```
Training final model on ALL data (2014-2025)...
Training on XXX,XXX samples
  WIN:  XX,XXX
  LOSS: XX,XXX

✓ Model saved: models/rebuilt_from_scratch/model1_hgb.joblib
  Thresholds: LONG≥0.55, SHORT≤0.45
```

## Success Criteria

For each model to PASS validation:
- **Minimum Win Rate**: 52% (conservative)
- **Target Win Rate**: 65-70% (realistic based on previous results)
- **Profit Factor**: ≥1.3
- **Signals per day**: 10-50 (reasonable frequency)
- **Consistent across folds**: Similar WR in all 4 periods

## Expected Training Time

- **Total**: 15-30 minutes (depending on hardware)
- Per model: 5-10 minutes
- Per fold: 1-2 minutes

## What to Check After Training

1. **Model Files Created**:
   ```bash
   ls -lh models/rebuilt_from_scratch/
   # Should see:
   # model1_hgb.joblib
   # model3_cmf_macd.joblib
   # model_rf.joblib
   ```

2. **Training Log** (saved automatically):
   ```bash
   cat training_log_phase5.txt
   ```

3. **Win Rates**: All models should show ≥52% (ideally 65-70%)

4. **Feature Counts**:
   - Model 1: Should use 37 microstructure features
   - Model 3: Should use 31 CMF/MACD features
   - Model RF: Should use 37 microstructure features

## If Training Fails

### Common Issues:

1. **Data file not found**:
   ```
   FileNotFoundError: data/features/xauusd_features_2020_2025_fixed.parquet
   ```
   - Check if file exists
   - Verify path is correct

2. **Memory error**:
   - Reduce data size in Config.DATA_PATH
   - Use fewer walk-forward splits

3. **Module import errors**:
   ```
   ModuleNotFoundError: No module named 'src.features'
   ```
   - Run from project root: `/Users/omar/Desktop/ML/xauusd_signals`
   - Check PYTHONPATH

4. **Feature generation errors**:
   - Check unified features module: `python3 -c "from src.features import build_features; print('OK')"`
   - Verify quotes data exists in parquet file

5. **Low win rates** (<52%):
   - This indicates models can't generalize
   - May need to adjust label config or feature engineering
   - Check if data quality is good

## Next Steps After Training

If all 3 models pass validation:

1. **Verify models**:
   ```bash
   python3 -c "
   import joblib
   m1 = joblib.load('models/rebuilt_from_scratch/model1_hgb.joblib')
   print(f\"Model 1 features: {len(m1['features'])}\")
   print(f\"Threshold: {m1['threshold_long']:.3f}\")
   "
   ```

2. **Update production config**:
   - Edit `models_config_production.py`
   - Point to new model paths in `models/rebuilt_from_scratch/`

3. **Proceed to Phase 6**: Deploy and monitor

## Troubleshooting

If you encounter issues, check:
- Python version: 3.8+ required
- Dependencies installed: numpy, pandas, sklearn, joblib
- Working directory: Must be project root
- Disk space: At least 2GB free for models

## Notes

- **Training uses unified features**: Same as live (perfect parity!)
- **Lookahead bias fixed**: `synthetic_order_flow` no longer uses volume multiplication
- **Conservative hyperparameters**: Designed to prevent overfitting
- **Walk-forward validation**: Ensures models generalize to future data
- **All changes committed**: Ready for version control tracking

---

**Status**: Ready to run
**Last updated**: 2026-01-13
**Phase**: 5 of 6
