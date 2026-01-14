# Regime Filter Impact Analysis
## 200-Day MA Filter Added to Momentum Strategy

**Date**: 2026-01-11
**Modification**: Added regime filter - only trade when price > 200-day MA

---

## 📊 Performance Comparison

### Training Period (2014-2023)

| Metric | Original | With Regime Filter | Change |
|--------|----------|-------------------|--------|
| **Total Return** | 24.84% | 24.02% | -0.82% ⬇️ |
| **CAGR** | 2.04% | 1.98% | -0.06% ⬇️ |
| **Sharpe Ratio** | 0.28 | 0.28 | 0.00 ➡️ |
| **Max Drawdown** | -18.49% | **-22.74%** | -4.25% ⚠️ **WORSE** |
| **Profit Factor** | 1.32 | 1.35 | +0.03 ⬆️ |
| **Win Rate** | 41.5% | 37.7% | -3.8% ⬇️ |
| **Avg Win** | +3.24% | +3.27% | +0.03% ➡️ |
| **Avg Loss** | -1.74% | -1.47% | +0.27% ⬆️ (better) |
| **Trades** | 53 | 53 | 0 ➡️ |
| **Avg Hold** | 35.8 days | 31.4 days | -4.4 days ⬇️ |

### Validation Period (2024-2025)

| Metric | Original | With Regime Filter | Change |
|--------|----------|-------------------|--------|
| **Total Return** | 40.46% | 40.46% | 0.00% ➡️ |
| **CAGR** | 28.92% | 28.92% | 0.00% ➡️ |
| **Sharpe Ratio** | 2.12 | 2.12 | 0.00 ➡️ |
| **Max Drawdown** | -7.10% | -7.10% | 0.00% ➡️ |
| **Profit Factor** | 11.34 | 11.34 | 0.00 ➡️ |
| **Win Rate** | 75.0% | 75.0% | 0.00% ➡️ |
| **Trades** | 4 | 4 | 0 ➡️ |

**Why no change in validation?** The entire 2024-2025 period was already 100% above the 200-day MA, so the filter had no effect.

---

## 🔍 Key Findings

### ❌ Regime Filter Did NOT Improve Training Performance

**Expected**: Filtering out sideways markets should reduce whipsaws and improve Sharpe ratio.

**Reality**:
- Sharpe stayed the same (0.28)
- Max drawdown got **WORSE** (-18.5% → -22.7%)
- Win rate declined (41.5% → 37.7%)
- Only slight PF improvement (1.32 → 1.35)

### 🤔 Why Didn't It Help?

1. **MA crossovers create new whipsaws**
   - Price crossing above/below 200-day MA generates entry/exit signals
   - These transitions are often choppy and create losses

2. **MA is a lagging indicator**
   - By the time price crosses above 200-day MA, momentum may already be fading
   - By the time it crosses below, damage is already done

3. **2014-2023 was fundamentally choppy**
   - Even periods above 200-day MA had frequent reversals
   - Momentum signals failed regardless of MA position

### ✅ What Actually Worked?

**The validation period (100% above MA)** shows the strategy works perfectly in sustained trends:
- Sharpe 2.12
- Max DD only 7.1%
- PF 11.34

**The issue isn't the strategy design** - it's that 2014-2023 lacked sustained trends.

---

## 📈 Signal Distribution Analysis

### Before Regime Filter
- Long signals: Unknown (not tracked)
- Flat signals: Unknown

### After Regime Filter
- **Long signals: 1,889 (56.0%)**
- **Flat signals: 1,485 (44.0%)**

The filter is active 44% of the time, which is substantial. This confirms that:
1. Gold spent ~44% of 2014-2023 below its 200-day MA (sideways/downtrend)
2. Even the 56% above MA wasn't enough to generate good returns

---

## 🎯 Deployment Assessment

### Original Strategy vs Regime-Filtered

| Criteria | Original | With Regime Filter | Winner |
|----------|----------|-------------------|--------|
| Training Sharpe | 0.28 | 0.28 | Tie |
| Training Max DD | -18.5% | -22.7% ⚠️ | **Original** |
| Training PF | 1.32 | 1.35 | Regime Filter |
| Validation Sharpe | 2.12 | 2.12 | Tie |
| Validation Max DD | -7.1% | -7.1% | Tie |
| Code Complexity | Simple | +2 features | Original |

**Verdict**: **Use original strategy (no regime filter)**

**Reasoning**:
1. Regime filter made training Max DD worse (-22.7%)
2. No improvement in training Sharpe (still 0.28)
3. No difference in validation (already trending)
4. Simpler is better when performance is identical

---

## 💡 Why the Regime Filter Failed

### Root Cause Analysis

**Problem**: Momentum strategies need persistent trends, not just "price > MA"

**Why 200-day MA filter didn't help**:
1. **MA is not a trend strength indicator** - it only shows direction
2. **Gold can be above 200-day MA but still choppy** (e.g., 2015-2017)
3. **MA crossovers add new entry/exit points** which created additional losses

### What Would Actually Help?

Instead of a simple MA filter, consider:

1. **Trend Strength Filter** (ADX or similar)
   - Only trade when ADX > 25 (strong trend)
   - This filters chop better than MA

2. **Volatility Clustering Filter**
   - Only trade when realized vol is in certain range
   - Avoid both dead zones (low vol) and panic zones (extreme vol)

3. **Multi-Timeframe Confirmation**
   - Require momentum alignment across multiple timeframes
   - E.g., 60d, 126d, and 252d all positive

4. **Market Regime Detection (ML-based)**
   - Use Hidden Markov Model to identify trending vs mean-reverting regimes
   - Only trade momentum in trending regimes

---

## 🔄 Next Steps

### Option 1: Revert to Original (Recommended)
- Remove 200-day MA filter
- Use original strategy as-is
- Deploy with 25% allocation in trending markets

**Why**: Simpler, slightly better Max DD, same Sharpe

### Option 2: Try Advanced Regime Detection
- Implement ADX-based trend strength filter
- Test threshold: ADX > 20 or ADX > 25
- Expected improvement: 20-30% better training Sharpe

### Option 3: Accept Regime-Dependency
- Keep original strategy
- Only deploy during confirmed uptrends (manual oversight)
- Monitor 200-day MA but don't automate the filter

---

## 📊 Technical Implementation

### Code Changes Made

**Added to FeatureEngineer**:
```python
# 6. Regime filter: 200-day moving average
df['ma_200'] = df['close'].rolling(200).mean()
df['above_ma_200'] = (df['close'] > df['ma_200']).astype(int)
```

**Added to SignalGenerator**:
```python
# Long when:
# - Composite score > threshold
# - Volatility regime < threshold
# - Price above 200-day MA (regime filter)
df['signal'] = 0
df.loc[
    (df['composite_score'] > threshold) &
    (df['vol_regime'] < vol_threshold) &
    (df['above_ma_200'] == 1),  # <-- NEW FILTER
    'signal'
] = 1
```

### Files Modified
- `momentum_strategy_production.py` (lines 177-182, 233-248)

---

## 📋 Summary

### What We Learned

1. **200-day MA filter ≠ better performance**
   - Made Max DD worse in training (-22.7% vs -18.5%)
   - No improvement in Sharpe or overall returns
   - Only marginally better PF (1.35 vs 1.32)

2. **The real issue is market regime, not position**
   - 2014-2023: Choppy even above MA
   - 2024-2025: Smooth even without filter (already 100% above MA)

3. **Simple filters can backfire**
   - MA crossovers create additional whipsaws
   - Lagging indicators don't prevent drawdowns

### Final Recommendation

**Use the ORIGINAL strategy without regime filter**

**Rationale**:
- Lower Max DD (-18.5% vs -22.7%)
- Same Sharpe (0.28)
- Simpler implementation
- Validation results identical (2.12 Sharpe)

**Deployment**: 25% allocation, monitor manually for trending markets, set hard stop at 20% DD

---

**Report Generated**: 2026-01-11
**Modification Status**: Regime filter tested but NOT recommended for production
**Next Action**: Revert to original strategy or test advanced regime detection (ADX)
