# Volatility Filter Diagnosis

## Problem
No long signals are occurring because the volatility filter is rejecting all signals with the error:
```
⚠️ FILTERED: Volatility filter failed. Market conditions not suitable.
```

## Root Cause
The volatility filter in `signal_engine.py` has two strict conditions that must BOTH pass:

1. **ATR Check**: `ATR_14 >= 0.50` (minimum 50 cents expected movement)
2. **Spread Check**: `spread <= 0.20` (maximum 20 cents spread)

If either condition fails, the signal is filtered out.

## Current Filter Logic (UPDATED)
Located in: `src/live/signal_engine.py` → `_check_volatility_filter()`

**Before (Too Strict):**
```python
min_atr = 0.50      # Minimum 50 cents ATR
max_spread = 0.20   # Maximum 20 cents spread (TOO STRICT for gold!)
```

**After (Fixed):**
```python
min_atr = 0.30      # Minimum 30 cents ATR (relaxed)
max_spread_pct = 0.001  # Maximum 0.1% spread (percentage-based, matches config)
# Fallback: max_spread = 2.00  # Maximum $2.00 spread (if spread_pct not available)
```

## Changes Made

1. **Switched to percentage-based spread** (`spread_pct`) instead of absolute spread
   - Aligns with `config.py` which uses `max_spread_pct: 0.001`
   - More appropriate for gold trading where spreads vary with price

2. **Relaxed ATR threshold** from 0.50 to 0.30
   - 30 cents is more reasonable for intraday gold trading

3. **Improved logging** to show which condition fails and actual values

## Why Signals Are Being Filtered

### Possible Reasons:
1. **ATR too low** (< 0.50): Market volatility is insufficient for profitable trading
   - During low-volatility periods (e.g., Asian session, weekends)
   - Gold might have ATR values below 50 cents

2. **Spread too high** (> 0.20): Trading costs are too expensive
   - During volatile periods, spreads can widen
   - During low-liquidity periods (e.g., after-hours)

## Diagnostic Improvements Made

I've updated the logging to show **which specific condition is failing** and the **actual values**:

- **Before**: Generic message "Volatility filter failed"
- **After**: Specific messages like:
  - `⚠️ Volatility filter FAILED: ATR too low. ATR_14=0.35 < min_atr=0.50`
  - `⚠️ Volatility filter FAILED: Spread too high. spread=0.25 > max_spread=0.20`

## Next Steps to Diagnose

1. **Check the logs** - The new logging will show which condition is failing and the actual values
2. **Review actual market conditions** - Check if current ATR/spread values are typical for XAUUSD
3. **Adjust thresholds if needed** - If thresholds are too strict, we can relax them

## Potential Solutions

### Option 1: Relax Thresholds (if too strict)
If the current market conditions are normal but thresholds are too strict:
```python
min_atr = 0.30      # Lower from 0.50 to 0.30 (30 cents)
max_spread = 0.30   # Increase from 0.20 to 0.30 (30 cents)
```

### Option 2: Make Thresholds Configurable
Add configuration parameters so thresholds can be adjusted without code changes.

### Option 3: Use Relative Thresholds
Instead of absolute values, use relative thresholds:
- ATR relative to price (e.g., ATR/price > 0.0001)
- Spread as percentage (e.g., spread_pct < 0.001)

### Option 4: Session-Aware Thresholds
Different thresholds for different trading sessions:
- Asian session: Lower ATR threshold (less volatility expected)
- London/NY overlap: Standard thresholds

## How to Check Current Values

Run the system and check the logs. The new logging will show:
- Actual ATR_14 value
- Actual spread value
- Which condition failed

Example log output:
```
⚠️ Volatility filter FAILED: ATR too low. ATR_14=0.35 < min_atr=0.50 (Market volatility too low for profitable trading)
```

## Related Configuration

Note: The config file (`src/config.py`) uses different spread filtering:
- `max_spread_pct: 0.001` (0.1% of price)
- This is a percentage-based filter, while the signal engine uses absolute values

For gold at ~$2650/oz:
- 0.1% spread = ~$2.65
- Current filter: 0.20 = $0.20

This suggests the absolute spread filter (0.20) might be too strict compared to the percentage-based filter (0.1% = $2.65).

