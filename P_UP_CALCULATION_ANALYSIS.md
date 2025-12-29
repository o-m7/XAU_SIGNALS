# P(up) Calculation Analysis

## How P(up) is Calculated

### 1. Model Prediction (`signal_engine.py` lines 114-125)

```python
# Get probability array from model
proba = self.model.predict_proba(X)[0]

# Find index of "up" class (class 1)
classes = self.model.classes_
if 1 in classes:
    up_idx = list(classes).index(1)
else:
    up_idx = -1  # Last class

# Extract probability of up class
proba_up = proba[up_idx]
```

**Key Points:**
- `p(up)` is the model's predicted probability of the "up" class (class 1)
- It comes directly from `model.predict_proba()` which outputs probabilities for each class
- The model is loaded from `models/y_tb_60_hgb_tuned.joblib`

### 2. Signal Generation (`signal_engine.py` lines 132-137)

```python
if proba_up >= threshold_long:      # Default: 0.60, Production: 0.70
    signal = Signal.LONG
elif proba_up <= threshold_short:   # Default: 0.40, Production: 0.30
    signal = Signal.SHORT
else:
    signal = Signal.FLAT
```

**Current Production Thresholds:**
- `threshold_long = 0.70` (LONG if p(up) >= 0.70)
- `threshold_short = 0.30` (SHORT if p(up) <= 0.30)
- FLAT otherwise (0.30 < p(up) < 0.70)

## Why Signals Might Be Missed

### 1. **High Thresholds** ⚠️ PRIMARY SUSPECT

**Production thresholds are HIGH:**
- LONG requires p(up) >= 0.70 (70% confidence)
- SHORT requires p(up) <= 0.30 (30% confidence = 70% down confidence)

**Impact:** Only very confident predictions generate signals. Many valid setups with p(up) in 0.60-0.70 range are filtered out.

**Recommendation:** Lower thresholds to capture more signals:
- `threshold_long = 0.60` (was 0.70)
- `threshold_short = 0.40` (was 0.30)

### 2. **Signal Engine Filters**

#### A. Volatility Filter (`signal_engine.py` lines 214-249)
**Blocks signals if:**
- `ATR_14 < 0.50` (market too dead)
- `spread > 0.20` (spread too wide)

**Impact:** Filters out low-volatility periods and high-spread conditions.

#### B. Churn Filter (`signal_engine.py` lines 251-308)
**Reduces conviction if:**
- LONG signal but `close_mid_diff < 0` (weak close)
- SHORT signal but `close_mid_diff > 0` (strong close)

**Impact:** Blocks signals when price action contradicts model prediction.

#### C. Wick Filter (`signal_engine.py` lines 146-171)
**Blocks SHORT signals if:**
- Large upper wick (>30% of range) AND positive order flow
- This is "bullish absorption" - buyers absorbing selling pressure

**Impact:** Prevents shorting bullish absorption patterns.

#### D. Conviction Filter (`signal_engine.py` lines 197-203)
**Blocks signals if:**
- Conviction < 0.5 (after churn filter reduction)

**Impact:** Filters out low-conviction signals.

### 3. **Risk Guard Filters** (`risk_guard.py`)

#### A. Confidence-Based Cooldowns (lines 86-113)
- **Extreme confidence** (p >= 0.75 or p <= 0.25): **0 seconds** cooldown
- **High confidence** (p >= 0.65 or p <= 0.35): **60 seconds** cooldown
- **Low confidence**: **300 seconds** (5 minutes) cooldown

**Impact:** Low-confidence signals have 5-minute cooldowns, reducing signal frequency.

#### B. Signal-Change Filter (lines 177-184)
**Blocks same signal unless:**
- Extreme confidence (p >= 0.75 or p <= 0.25)

**Impact:** Prevents repeating the same signal unless confidence is very high.

**Example:**
- Signal LONG with p(up)=0.70 → sent
- Next signal LONG with p(up)=0.65 → **BLOCKED** (same signal, not extreme confidence)
- Next signal LONG with p(up)=0.80 → **ALLOWED** (extreme confidence)

## Diagnostic Steps

### 1. Check Current P(up) Values
Look at logs for `P(up)=` values. If many are in 0.60-0.70 range, thresholds are too high.

### 2. Check Filter Logs
Look for these log messages:
- `⚠️ FILTERED: Volatility filter failed`
- `⚠️ FILTERED: Low conviction signal`
- `⛔ FILTERED: Model shorting a Bullish Absorption Wick`
- `Signal blocked: Cooldown: Xs remaining`
- `Signal blocked: Same signal (LONG) - need extreme confidence to repeat`

### 3. Check Risk Guard Status
The risk guard logs when signals are blocked. Check `live_runner.log` for:
- `Signal blocked: Cooldown: Xs remaining`
- `Signal blocked: Same signal (X) - need extreme confidence to repeat`

## Recommendations

### Immediate Fix: Lower Thresholds
```bash
# In railway.json or start_signal_engine.sh:
--threshold_long 0.60   # Was 0.70
--threshold_short 0.40  # Was 0.30
```

### Monitor Filter Activity
Add more detailed logging to see which filters are blocking signals most often.

### Adjust Risk Guard Cooldowns
If cooldowns are too aggressive, consider:
- Reducing low-confidence cooldown from 300s to 180s
- Allowing same-signal repeats with high confidence (0.65+) instead of only extreme (0.75+)

### Relax Volatility Filter
If market conditions are normal but ATR is low:
- Reduce `min_atr` from 0.50 to 0.30
- Increase `max_spread` from 0.20 to 0.30

## Code Locations

- **P(up) calculation**: `src/live/signal_engine.py:114-125`
- **Signal thresholds**: `src/live/signal_engine.py:132-137`
- **Filters**: `src/live/signal_engine.py:146-249`
- **Risk guard**: `src/live/risk_guard.py:119-191`
- **Production config**: `railway.json:7`, `start_signal_engine.sh:47-48`
