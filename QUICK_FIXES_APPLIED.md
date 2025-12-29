# Quick Fixes Applied for Missing Signals

## Changes Made

### 1. ✅ Lowered Thresholds (railway.json)
**Before:** `threshold_long=0.70`, `threshold_short=0.30`
**After:** `threshold_long=0.60`, `threshold_short=0.40`

**Impact:** 
- LONG signals: P(up) >= 0.60 (was 0.70) - captures more opportunities
- SHORT signals: P(up) <= 0.40 (was 0.30) - captures more opportunities
- **During drop:** If P(up) drops to 0.40, SHORT signal will be generated

### 2. ✅ Relaxed Volatility Filter (src/live/signal_engine.py)
**Before:** `min_atr=0.50`, `max_spread=0.20`
**After:** `min_atr=0.30`, `max_spread=0.30`

**Impact:**
- Allows signals during lower volatility periods
- Allows signals when spread widens up to 30 cents (important during crashes)
- **During drop:** Spread can widen significantly, now won't block signals

### 3. ✅ Reduced Cooldowns (src/live/risk_guard.py)
**Before:** Low confidence = 300 seconds (5 minutes)
**After:** Low confidence = 180 seconds (3 minutes)

**Impact:**
- Signals can be generated more frequently
- Low-confidence signals (P(up) 0.35-0.65) have shorter cooldown
- **During drop:** More opportunities to catch the move

## What's Still Needed

### ⚠️ CRITICAL: Fix Bar Collection
- Currently only collecting 1 bar instead of 65+
- Features can't be computed properly without enough bars
- **Check logs for:** "Backfill complete: X bars added"
- **If < 65 bars:** Investigate API key, network, symbol resolver

### ⚠️ Monitor P(up) Updates
- Check if P(up) is updating with market conditions
- **Check logs for:** "⚠️ P(up) STUCK" warnings
- **If stuck:** Features aren't updating → Fix bar collection first

## Expected Behavior After Fixes

### During Market Drop:
1. **P(up) should decrease** (e.g., from 0.50 → 0.40 → 0.30)
2. **When P(up) <= 0.40:** SHORT signal generated
3. **Volatility filter:** Won't block if spread < 30 cents
4. **Cooldown:** 3 minutes instead of 5 minutes
5. **Signal sent:** If passes all checks

### Signal Generation:
- **LONG:** P(up) >= 0.60 (was 0.70)
- **SHORT:** P(up) <= 0.40 (was 0.30)
- **FLAT:** 0.40 < P(up) < 0.60

## Next Steps

1. **Deploy these fixes** to Railway
2. **Monitor logs** for:
   - Bar collection count
   - P(up) values during moves
   - Signal generation
   - Filter blocking messages
3. **If still no signals:**
   - Check bar collection (must be 65+)
   - Check P(up) values (should change with market)
   - Check filter logs (which filters are blocking)

## Diagnostic Commands

```bash
# Check bar collection
grep "bars" live_runner.log | tail -20

# Check P(up) values
grep "P(up)=" live_runner.log | tail -20

# Check signal generation
grep "Signal:" live_runner.log | tail -20

# Check what's blocking signals
grep "FILTERED\|blocked" live_runner.log | tail -20
```
