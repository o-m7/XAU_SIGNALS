# Why No Signals During Big Drop - Complete Analysis

## Problem
No signals generated during a significant market drop, despite the system being active.

## All Signal Blockers (In Order of Execution)

### 1. **Insufficient Bars Collected** ⚠️ CRITICAL
**Location:** `src/live/feature_buffer.py`
- **Requirement:** 65 bars minimum
- **Current:** Only 1 bar collected
- **Impact:** Features can't be computed properly → P(up) unreliable → No signals

**Fix Status:** ✅ Enhanced logging added, but root cause needs investigation

### 2. **High Probability Thresholds** ⚠️ PRIMARY SUSPECT
**Location:** `railway.json:7`, `src/live/signal_engine.py:175-180`
- **Current:** `threshold_long=0.70`, `threshold_short=0.30`
- **Meaning:** 
  - LONG requires P(up) >= 0.70 (70% confidence)
  - SHORT requires P(up) <= 0.30 (30% confidence = 70% down confidence)
- **Impact:** Only very confident predictions generate signals
- **During drop:** P(up) might be 0.40-0.50 → FLAT signal (no trade)

**Recommendation:** Lower to `threshold_long=0.60`, `threshold_short=0.40`

### 3. **Volatility Filter** ⚠️ BLOCKER
**Location:** `src/live/signal_engine.py:214-249`
- **Blocks if:**
  - `ATR_14 < 0.50` (market too dead)
  - `spread > 0.20` (spread too wide)
- **Impact:** Filters out low-volatility periods and high-spread conditions
- **During drop:** If spread widens significantly, signals blocked

**Log Message:** `⚠️ FILTERED: Volatility filter failed`

### 4. **Churn Filter (Conviction Reduction)** ⚠️ BLOCKER
**Location:** `src/live/signal_engine.py:251-308`
- **Reduces conviction if:**
  - LONG signal but `close_mid_diff < 0` (weak close)
  - SHORT signal but `close_mid_diff > 0` (strong close)
- **Impact:** If conviction < 0.5, signal changed to FLAT
- **During drop:** If price closes strong despite drop, SHORT signals blocked

**Log Message:** `⚠️ FILTERED: Low conviction signal (X.XX)`

### 5. **Wick Filter** ⚠️ BLOCKER (For SHORT signals)
**Location:** `src/live/signal_engine.py:189-210`
- **Blocks SHORT if:**
  - Large upper wick (>30% of range) AND positive order flow
  - This is "bullish absorption" - buyers absorbing selling pressure
- **Impact:** Prevents shorting bullish absorption patterns
- **During drop:** If there's buying into the drop, SHORT signals blocked

**Log Message:** `⛔ FILTERED: Model shorting a Bullish Absorption Wick`

### 6. **Risk Guard Cooldowns** ⚠️ BLOCKER
**Location:** `src/live/risk_guard.py:86-113`
- **Cooldown rules:**
  - Extreme confidence (p >= 0.75 or p <= 0.25): 0 seconds
  - High confidence (p >= 0.65 or p <= 0.35): 60 seconds
  - Low confidence: 300 seconds (5 minutes)
- **Impact:** Low-confidence signals have 5-minute cooldowns
- **During drop:** If P(up) is 0.40-0.50 (low confidence), 5-minute cooldown blocks signals

**Log Message:** `Signal blocked: Cooldown: Xs remaining`

### 7. **Signal-Change Filter** ⚠️ BLOCKER
**Location:** `src/live/risk_guard.py:177-184`
- **Blocks same signal unless:**
  - Extreme confidence (p >= 0.75 or p <= 0.25)
- **Impact:** Prevents repeating the same signal unless very confident
- **During drop:** If last signal was SHORT, next SHORT blocked unless extreme confidence

**Log Message:** `Signal blocked: Same signal (SHORT) - need extreme confidence to repeat`

### 8. **P(up) Stuck at 0.4-0.5** ⚠️ ROOT CAUSE
**Location:** `src/live/signal_engine.py:154-168`
- **Problem:** P(up) not updating with market conditions
- **Causes:**
  - Features not updating (only 1 bar collected)
  - Model seeing stale features
- **Impact:** P(up) stays constant → No signals generated

**Log Message:** `⚠️ P(up) STUCK: 0.4XXX (change=0.0000)`

## Signal Generation Flow

```
1. Event received → Update feature buffer
2. Get feature_row (None if bar not completed)
3. Generate signal from features:
   - Compute P(up) from model
   - Check thresholds: LONG if P(up) >= 0.70, SHORT if P(up) <= 0.30
   - If FLAT → STOP (no signal)
4. Apply filters:
   - Wick filter (for SHORT)
   - Churn filter (conviction check)
   - Volatility filter
   - If filtered → FLAT → STOP
5. Check risk guard:
   - Cooldown check
   - Signal-change filter
   - If blocked → STOP
6. Send signal (if passed all checks)
```

## Why No Signals During Drop

### Most Likely Scenario:
1. **Only 1 bar collected** → Features can't compute → P(up) unreliable
2. **P(up) stuck at 0.4-0.5** → Doesn't meet thresholds (0.70/0.30) → FLAT
3. **Even if P(up) changes:**
   - Spread widens → Volatility filter blocks
   - Low confidence → 5-minute cooldown blocks
   - Same signal → Signal-change filter blocks

## Immediate Fixes Needed

### Fix 1: Lower Thresholds (HIGH PRIORITY)
**File:** `railway.json`
```json
"startCommand": "python -m src.live.live_runner --backfill --threshold_long 0.60 --threshold_short 0.40"
```

**Impact:** Captures more signals during market moves

### Fix 2: Fix Bar Collection (CRITICAL)
**Files:** `src/live/backfill.py`, `src/live/feature_buffer.py`
- Already added diagnostic logging
- Need to investigate why only 1 bar collected
- Check API key, network, symbol resolver

### Fix 3: Relax Volatility Filter (MEDIUM PRIORITY)
**File:** `src/live/signal_engine.py:233-242`
- Current: `min_atr = 0.50`, `max_spread = 0.20`
- During crashes, spread can widen significantly
- Consider: `min_atr = 0.30`, `max_spread = 0.30`

### Fix 4: Reduce Cooldowns (MEDIUM PRIORITY)
**File:** `src/live/risk_guard.py:86-113`
- Current: Low confidence = 300 seconds
- Consider: Low confidence = 180 seconds (3 minutes)
- Or: Remove cooldown for SHORT signals during drops

### Fix 5: Relax Signal-Change Filter (LOW PRIORITY)
**File:** `src/live/risk_guard.py:177-184`
- Current: Blocks same signal unless extreme confidence (0.75/0.25)
- Consider: Allow same signal with high confidence (0.65/0.35)

## Diagnostic Checklist

### Check Logs For:
1. ✅ **"Backfill complete: X bars added"** - How many bars collected?
2. ✅ **"P(up)=X.XXXX"** - What's the actual P(up) value?
3. ✅ **"Signal: FLAT"** - Is signal being generated but filtered?
4. ✅ **"FILTERED:"** - Which filter is blocking?
5. ✅ **"Signal blocked:"** - Why is risk guard blocking?
6. ✅ **"⚠️ P(up) STUCK"** - Is P(up) updating?

### Expected During Drop:
- **P(up) should decrease** (more likely to go down)
- **SHORT signals should be generated** if P(up) <= threshold_short
- **If P(up) = 0.30-0.40** → Should generate SHORT signal (if threshold_short = 0.40)

## Quick Fixes to Deploy

### 1. Lower Thresholds Immediately
```bash
# Update railway.json
"startCommand": "python -m src.live.live_runner --backfill --threshold_long 0.60 --threshold_short 0.40"
```

### 2. Check Bar Collection
```bash
# Check logs
grep "bars" live_runner.log | tail -20
grep "Backfill complete" live_runner.log
```

### 3. Monitor Signal Generation
```bash
# Check what signals are being generated
grep "Signal:" live_runner.log | tail -20
grep "FILTERED\|blocked" live_runner.log | tail -20
```

## Summary

**Primary Blockers:**
1. ⚠️ Only 1 bar collected (can't compute features)
2. ⚠️ High thresholds (0.70/0.30) too restrictive
3. ⚠️ P(up) stuck at 0.4-0.5 (doesn't meet thresholds)
4. ⚠️ Multiple filters blocking signals
5. ⚠️ Risk guard cooldowns blocking signals

**Fix Priority:**
1. **CRITICAL:** Fix bar collection (need 65+ bars)
2. **HIGH:** Lower thresholds to 0.60/0.40
3. **MEDIUM:** Relax volatility filter and cooldowns
4. **LOW:** Relax signal-change filter
