# P(up) Stuck at 0.4-0.5 Diagnosis

## Problem
P(up) is staying constant at 0.4-0.5 even though the market crashed to 4330 (significant price movement).

## Root Cause Analysis

### How P(up) is Calculated
1. **Feature Buffer** aggregates ticks/quotes into minute bars
2. **Features** are computed from the bar history (returns, volatility, etc.)
3. **Model** predicts P(up) from features
4. **Signal** is generated based on P(up) thresholds

### Why P(up) Might Be Stuck

#### 1. **Features Only Update When Bars Complete** ⚠️ PRIMARY SUSPECT
- `update_from_quote()` returns `feature_row` **only when a bar completes**
- If quotes come in but bars don't complete frequently, features stay stale
- The model sees the same feature values repeatedly → same P(up)

**Code Location:** `src/live/feature_buffer.py:166-201`
- `_update()` returns `None` when updating current bar
- Only returns `get_feature_row()` when a new bar starts

#### 2. **Feature Buffer Not Receiving New Bars**
- If WebSocket is only sending quotes (not bars), bars might not complete
- Bar completion requires timestamp to cross minute boundary
- If quotes are sparse or timestamps are wrong, bars never complete

#### 3. **Stale Feature Values**
- Rolling features (vol_10, vol_60, ATR_14) use historical data
- If new bars aren't being added, rolling windows don't update
- Features stay constant → P(up) stays constant

## Fixes Applied

### Fix 1: Always Use Latest Features
**File:** `src/live/live_runner.py:413-421`

```python
# CRITICAL FIX: If feature_row is None (bar not completed yet), 
# but buffer is ready, get the latest features anyway
if feature_row is None and self.feature_buffer.is_ready():
    try:
        feature_row = self.feature_buffer.get_feature_row()
        logger.debug("Using latest features (bar not yet completed)")
    except Exception as e:
        logger.debug(f"Could not get latest features: {e}")
```

**What it does:** Ensures we always use the most recent features, even if a bar hasn't completed yet.

### Fix 2: Diagnostic Logging
**Files:** 
- `src/live/live_runner.py:427-456` - Logs feature changes
- `src/live/signal_engine.py:113-150` - Logs when features don't change
- `src/live/feature_buffer.py:301-334` - Logs bar completion and feature values

**What it does:** Helps identify when features are stale or not updating.

## Diagnostic Checks

### Check Logs For:
1. **"⚠️ NO FEATURE CHANGES detected"** - Features aren't updating
2. **"⚠️ P(up) STUCK"** - Prediction isn't changing despite feature updates
3. **"FeatureBuffer: Computing features for bar X"** - Bars are completing
4. **Feature changes in logs** - Should see `ret_1`, `ret_5`, etc. changing

### Manual Check:
```bash
# Check if bars are being collected
grep "Bars=" live_runner.log | tail -20

# Check if features are changing
grep "Changes:" live_runner.log | tail -20

# Check for warnings about stale features
grep "STUCK\|NO FEATURE" live_runner.log | tail -20
```

## Additional Recommendations

### 1. Check WebSocket Data Flow
- Verify that bars are being received from Polygon
- Check if `WS_MODE=all` is set (should receive quotes, minute bars, second bars)
- If only quotes are coming, bars might not complete frequently

### 2. Verify Bar Completion
- Bars complete when timestamp crosses minute boundary
- Check logs for "FeatureBuffer: Computing features for bar X"
- If bars aren't completing, check timestamp handling

### 3. Check Feature Calculation
- Features should update when new bars are added
- Rolling windows (vol_10, vol_60, ATR_14) should change with new data
- If features are constant, check `_compute_features()` logic

### 4. Monitor Feature Buffer State
- Check `feature_buffer.get_bar_count()` - should increase over time
- Check `feature_buffer.is_ready()` - should be True after warmup
- If bar count isn't increasing, bars aren't being added

## Expected Behavior

### Normal Operation:
- New bar completes → features recomputed → P(up) updates
- P(up) should vary with market conditions
- During a crash, P(up) should decrease (more likely to go down)

### If P(up) is Stuck:
- Features aren't updating → check bar completion
- Features updating but P(up) constant → check model/features
- No bars being added → check WebSocket data flow

## Next Steps

1. **Deploy fixes** and monitor logs for diagnostic messages
2. **Check logs** for warnings about stale features
3. **Verify** that bars are completing and features are updating
4. **If still stuck**, check WebSocket data flow and bar aggregation logic
