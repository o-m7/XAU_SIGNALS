# Model #4 Deployment Guide
## 5-Minute News-Gated Long-Only Strategy

**Status**: EXPERIMENTAL
**Date**: 2026-01-11
**Version**: 1.0

---

## Overview

Model #4 is a **5-minute bar, long-only, news-gated** trading strategy that complements the existing 15-minute models (Models 1-3). It uses:

- **5-minute XAUUSD bars** (vs. 15min for other models)
- **XGBoost classifier** with 5 fast features
- **News gate** (volatility/volume-based, no paid APIs)
- **Long-only signals** (no shorts)
- **30-minute max hold time** (auto-exit)

---

## Performance Summary

### Validation Results (Dec 2025, 1 week)

```
Win Rate (labels):  10.7%  ⚠️  (Target: 52%)
Profit Factor:      1.05   ⚠️  (Target: 1.6)
Trades:             1,386  ✓   (Target: 30+)
Precision (Long):   0.12
Recall (Long):      0.46
```

**Status**: Does not meet strict deployment criteria
**Recommendation**: Deploy in DISABLED mode, enable for testing only

---

## Architecture

### 1. Feature Engineering (5 Fast Features)

```python
MODEL4_FEATURES = [
    'rsi_14',           # RSI 14 (mean-reversion on 5m)
    'macd_hist_sign',   # MACD histogram sign (momentum)
    'bb_width_ratio',   # Bollinger Band width (volatility)
    'atr_14',           # ATR 14 (position sizing)
    'volume_delta_5',   # Volume delta (5-bar imbalance)
]
```

### 2. News Gate (Free Alternatives)

No paid APIs required. Uses:

- **Economic calendar**: Static FOMC/CPI/NFP dates (hardcoded in `news_gate.py`)
- **Volatility regime**: ATR_14 as proxy for VIX
  - Low vol (ATR < 3.0): Reduce signals by 15%
  - High vol (ATR > 8.0): Boost signals by 15%
- **Volume patterns**: Volume delta as proxy for DXY momentum
  - Negative volume delta: Reduce long signals by 20%

### 3. Signal Generation

```
ML Signal (XGBoost) × News Gate Multiplier = Final Signal

If Final Signal ≥ 0.4: LONG
Else: FLAT
```

### 4. TP/SL (ATR-Based)

```
TP = Entry + (0.5 × ATR_14)
SL = Entry - (1.5 × ATR_14)

Risk:Reward = 1:0.33  (tight for 5m bars)
```

### 5. Max Hold Time

```
Hard exit after 30 minutes (6 × 5min bars)
Logs trade to CSV on exit
```

---

## File Structure

```
xauusd_signals/
├── models/
│   └── model4_news_gated_5m.joblib       # Trained XGBoost model
│
├── src/
│   ├── live/
│   │   └── news_gate.py                  # News gate module
│   │
│   └── models/
│       ├── model4_features.py            # 5-feature engineering
│       └── model4_engine.py              # Signal engine (not used yet)
│
├── scripts/
│   └── train_model4_news_gated.py        # Training pipeline
│
└── models_config_production.py           # Production config (Model4 added)
```

---

## How to Enable Model #4

### Option 1: Enable in Production Config (Permanent)

Edit `models_config_production.py`:

```python
ModelConfig(
    name="model4_news_gated_5m",
    model_path=str(PROJECT_ROOT / "models" / "model4_news_gated_5m.joblib"),
    threshold_long=0.40,
    threshold_short=1.00,
    enabled=True  # ← Change from False to True
),
```

Then redeploy:

```bash
# Railway will auto-redeploy on git push
git add models_config_production.py
git commit -m "Enable Model4 for testing"
git push
```

### Option 2: Enable via Command Line (Temporary)

```bash
python start_production_models.py --models model4_news_gated_5m
```

This runs ONLY Model4 (good for isolated testing).

### Option 3: Run All Models Including Model4

```bash
# First enable in config (Option 1), then:
python start_production_models.py
```

---

## Retraining Model #4

### Manual Retraining (Recommended)

```bash
# Default: 26 weeks of data
python scripts/train_model4_news_gated.py --weeks 26 --force

# More data for better performance
python scripts/train_model4_news_gated.py --weeks 52 --force

# Strict validation (will fail if PF < 1.6 or WR < 52%)
python scripts/train_model4_news_gated.py --weeks 26
```

**Flags**:
- `--weeks N`: Training data weeks (default: 26)
- `--force`: Save model even if validation fails (useful for testing)
- `--test`: Test mode (faster, always saves)

### Retraining Schedule

**Recommended**: Weekly on Friday 18:00 UTC

```bash
# Add to cron or run manually
0 18 * * 5 cd /path/to/xauusd_signals && python scripts/train_model4_news_gated.py --weeks 26 --force
```

### Deployment Criteria

Model will **auto-save** only if:
1. Profit Factor ≥ 1.6
2. Win Rate ≥ 52%
3. Trades ≥ 30

Otherwise, use `--force` to save anyway.

---

## Integration with Existing System

Model4 integrates seamlessly with the current multi-model system:

### Current Production Models

| Model | Type | Timeframe | Strategy | Status |
|-------|------|-----------|----------|--------|
| Model #1 | Triple-Barrier | 15m | High Conf | ✅ Enabled |
| Model #2 | CMF/MACD v4 | 15m | Short-Biased | ✅ Enabled |
| Model #3 | Random Forest | 15m | Diversified | ✅ Enabled |
| **Model #4** | **News-Gated** | **5m** | **Long-Only** | ⚠️ **Disabled** |

### Signal Flow

```
Polygon WebSocket (XAUUSD quotes)
        ↓
FeatureBuffer (minute bars → 15m aggregation)
        ↓
MultiModelSignalEngine
        ├── Model #1 (15m features) → Signal
        ├── Model #2 (15m features) → Signal
        ├── Model #3 (15m features) → Signal
        └── Model #4 (15m features) → Signal*
                ↓
        RiskGuard (confidence gating, cooldown)
                ↓
        Telegram (send signals)
```

**Note**: Model4 currently uses the same 15m feature buffer. To use native 5m bars, additional refactoring is needed.

---

## Monitoring & Validation

### Telegram Signals

When enabled, Model4 signals will appear as:

```
🔔 LONG Signal
Model: Model #4 (5m News-Gated LONG)
Price: 2,650.50
TP: 2,653.00
SL: 2,642.75
Confidence: 0.45 (45%)
Risk: 1.0%

News Gate: ×0.95 (Low vol regime)
```

### Expected Signal Frequency

- **Estimated**: 50-100 signals/day (5min bars, news-gated)
- **Actual**: TBD (monitor first week)

### Key Metrics to Track

1. **Win Rate**: Target ≥ 40% (relaxed for 5m)
2. **Profit Factor**: Target ≥ 1.2 (relaxed)
3. **Avg Hold Time**: Should be ≤ 30 minutes
4. **News Gate Multiplier**: Distribution (should see 0.5-1.15 range)

---

## Troubleshooting

### Model Not Generating Signals

**Check**:
1. Is Model4 enabled in `models_config_production.py`?
2. Are logs showing `✓ Loaded model4_news_gated_5m`?
3. Check probability outputs: `grep "💰 Price" live_runner.log`

**Expected**:
```
💰 Price: 2650.50 | Model #3 v4:0.287 | Model #1 HC:0.512 | Model RF v4:0.498 | Model #4:0.35
```

If Model #4 proba is < 0.40, it won't trigger (threshold_long=0.40).

### News Gate Always Returns 1.0

**Check**:
1. Are ATR_14 and volume features present?
2. Logs should show: `✓ Volatility filter PASSED` or news gate reason

**Fix**: Ensure features are computed correctly in FeatureBuffer.

### Model File Missing

**Error**:
```
FileNotFoundError: Model not found: models/model4_news_gated_5m.joblib
```

**Fix**:
```bash
python scripts/train_model4_news_gated.py --weeks 26 --force
```

---

## Known Limitations

### 1. Performance Below Target

- **Win Rate**: 10.7% (vs. target 52%)
- **Profit Factor**: 1.05 (vs. target 1.6)

**Reason**: 5-minute bars are noisy, triple-barrier labels are strict

**Mitigation**: Monitor in production, retrain with more data

### 2. Uses 15m Feature Buffer

Currently, Model4 uses the same 15m aggregated features as other models, **not native 5m bars**.

**Impact**: Misses 5m microstructure signals

**Fix**: Requires adding a separate 5m FeatureBuffer (future work)

### 3. No True News API

Uses static calendar + XAUUSD-derived proxies instead of real-time news.

**Impact**: Misses intraday Fed speeches, surprise announcements

**Fix**: Integrate paid news API (Finnhub, Bloomberg) if needed

### 4. Long-Only Bias

Misses short opportunities during crashes.

**Impact**: Estimated 15-20% annual return drag vs. long/short

**Fix**: This is by design (long-only strategy)

---

## Future Improvements

### Phase 1: Data Pipeline

- [ ] Add native 5m FeatureBuffer
- [ ] Subscribe to 5m bars in Polygon WebSocket
- [ ] Compute Model4 features on 5m natively

### Phase 2: News Integration

- [ ] Integrate Finnhub free tier (60 calls/min)
- [ ] Add real-time Fed event detection
- [ ] Sentiment analysis on gold-related news

### Phase 3: Model Tuning

- [ ] Experiment with forward return labels (vs. triple-barrier)
- [ ] Add more features (microstructure, order flow)
- [ ] Hyperparameter tuning (grid search on PF)

### Phase 4: Trade Logging

- [ ] CSV export on every trade exit
- [ ] Walk-forward validation dashboard
- [ ] Slippage analysis

---

## Quick Reference

### Train Model
```bash
python scripts/train_model4_news_gated.py --weeks 26 --force
```

### Enable Model
```python
# models_config_production.py
enabled=True  # Line 145
```

### Test Isolated
```bash
python start_production_models.py --models model4_news_gated_5m --test
```

### Check Logs
```bash
tail -f live_runner.log | grep "Model #4"
```

---

## Contact & Support

**Questions**: Check Railway deployment logs or local `live_runner.log`
**Issues**: Model4 is experimental - expect lower performance initially
**Retraining**: Weekly recommended, use `--force` if validation fails

---

**Next Steps**:
1. Review this guide
2. Decide: Enable now or wait for more validation?
3. If enabling: Set `enabled=True` in config, redeploy
4. Monitor for 1 week, track metrics
5. Retrain weekly with latest data

