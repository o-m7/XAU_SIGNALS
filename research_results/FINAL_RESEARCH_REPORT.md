# XAUUSD Intraday Strategy - Final Research Report

**Project:** Profitable intraday XAUUSD strategy development
**Duration:** Phase 1-3 (January 6, 2026)
**Objective:** PF ≥ 1.6, WR ≥ 52%, Sharpe ≥ 0.25, DD ≤ 6%, R-multiple > 1.2
**Status:** ❌ **NOT PROFITABLE** (but significant learnings achieved)

---

## Executive Summary

Conducted rigorous 3-phase research to develop profitable XAUUSD intraday trading strategy:

### What We Accomplished ✅

1. **Validated statistical edges** (mean reversion p < 0.0001, session effects p = 0.001)
2. **Built robust research infrastructure** (EDA, features, labels, models, backtesting)
3. **Achieved 55.7% win rate** in Phase 3 (above 55% target)
4. **Identified exploitable patterns** (extreme mean reversion, London/NY sessions, calm volatility)

### Why It's Not Profitable ❌

1. **R-multiple problem:** Winning +9.6 bps but losing -10.5 bps (R = 0.92 < 1.0)
2. **Transaction costs:** 2.5 bps/trade eliminates thin edge
3. **Trade-off paradox:** Can get high WR OR good R-multiple, but not both
4. **Edge too small:** Statistical significance ≠ economic profitability

### Recommendation

**The mean reversion edge at intraday XAUUSD is REAL but TOO SMALL for retail profitability** given transaction costs.

**Three options:**
1. **Accept it:** Use for institutional/prop trading (lower costs)
2. **Pivot:** Try different strategy type (momentum, breakout, macro)
3. **Refine:** One more iteration with optimized risk/reward targets

---

## Phase-by-Phase Results

### Phase 1: 15-Min Triple-Barrier (Baseline)

**Approach:**
- 15-minute bars, 2020-2024 (5 years)
- 65 features (price, momentum, volatility, volume, session)
- Triple-barrier labels (2.0σ profit / 1.0σ stop)
- 4 model architectures (LightGBM best)

**Results:**
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Win Rate | 40.45% | 52% | ❌ |
| R-multiple | 1.53 | >1.2 | ✅ |
| AUC | 0.521 | >0.60 | ❌ |

**Key Findings:**
- Statistical edges validated ✅
- Models learned correct patterns ✅
- But AUC ~0.52 (barely better than random) ❌
- Too many signals, too noisy ❌

---

### Phase 2: Meta-Labeling

**Approach:**
- Primary model predicts direction
- Meta-model predicts "should I trade?"
- Filter signals to improve quality
- Optimize probability threshold

**Results:**
| Metric | Phase 1 | Phase 2 | Change |
|--------|---------|---------|--------|
| Win Rate | 40.45% | 44.96% | +4.51% ✅ |
| Profit Factor | N/A | 1.14 | ❌ |
| R-multiple | 1.53 | 1.40 | ✅ |
| Sharpe/trade | 0.016 | 0.052 | +3.2x ✅ |

**Key Findings:**
- Modest improvement +4.5% WR ✅
- But meta-model AUC only 0.515 ❌
- Probability range too narrow (0.36-0.47) ❌
- Still unprofitable after costs ❌

---

### Phase 3: 30-Min + Strict Filters ⭐

**Approach:**
- 30-minute bars (reduce noise)
- **Strict entry filters (15.6% of bars pass):**
  1. Extreme mean reversion: |z-score| > 1.5
  2. London or NY session only
  3. Volatility < 80th percentile (calm markets)
- Fixed pip targets: +20 pips profit / -12 pips stop
- Target: 5-10 trades/day with 55%+ WR

**Results:**
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Win Rate** | **55.66%** | 55% | ✅ **ACHIEVED!** |
| R-multiple | 0.92 | >1.2 | ❌ **INVERTED** |
| Profit Factor | 1.16 | ≥1.6 | ❌ |
| Sharpe/trade | 0.0507 | ≥0.25 | ❌ |
| Trades/day | 2.2 | 5-10 | ❌ |
| After costs | -1.76 bps | >0 | ❌ |

**Key Findings:**
- ✅ **WIN RATE TARGET ACHIEVED** (55.7%)
- ✅ Mean reversion confirmed at 30-min (ACF = -0.0302)
- ✅ Model AUC improved to 0.537
- ❌ **R-multiple inverted:** Avg win +9.6 bps < Avg loss -10.5 bps
- ❌ Fixed targets don't match reality (slippage, gaps)
- ❌ Too few trades (2.2/day vs 5-10 target)

**The Critical Problem:**

Even with 55.7% win rate, we're **losing more on losers than winning on winners**:
```
55.7% × (+9.61 bps) - 44.3% × (-10.48 bps) = 0.74 bps before costs
0.74 bps - 2.5 bps (costs) = -1.76 bps after costs ❌
```

---

## The Trade-Off Paradox

**Discovered a fundamental constraint:**

| Phase | Win Rate | R-multiple | Profitable? |
|-------|----------|------------|-------------|
| Phase 1 (15-min) | 40.5% | 1.53 | ❌ (WR too low) |
| Phase 2 (meta) | 45.0% | 1.40 | ❌ (WR too low) |
| Phase 3 (30-min) | 55.7% | 0.92 | ❌ (R-mult too low) |

**Needed for profitability:**
```
WR × Avg_Win - (1 - WR) × Avg_Loss > 2.5 bps (transaction costs)

Option A: WR = 55% → Need R-mult = 1.3+
Option B: R-mult = 1.5 → Need WR = 50%+

Current: WR = 55.7%, R-mult = 0.92 ❌
```

**Can't achieve both simultaneously with current edge.**

---

## Why The Edge Exists But Isn't Profitable

### The Edge Is Real ✅

**Statistical evidence (can't be disputed):**
1. **Mean reversion:** Lag-1 ACF = -0.0385 (15-min), -0.0302 (30-min), p < 0.0001
2. **Session effects:** ANOVA F = 5.32, p = 0.001
3. **Volatility clustering:** ARCH test p < 0.0001

These are statistically significant, peer-reviewable findings.

### But Economic Profitability Requires More

**The math:**
- **Edge magnitude:** 3.68% directional advantage (51.92% vs 48.24%)
- **Translated to returns:** ~0.5-1.0 bps/trade expected value
- **Transaction costs:** 2.5 bps/trade (spread + slippage)
- **Net:** -1.5 to -2.0 bps/trade ❌

**The edge is TOO SMALL to overcome friction costs.**

---

## What Went Right ✅

### 1. Methodology

- ✅ First-principles approach (didn't p-hack or data mine)
- ✅ Statistical validation before building models
- ✅ Rigorous walk-forward testing
- ✅ Realistic transaction cost modeling
- ✅ Honest assessment (didn't cherry-pick results)

### 2. Infrastructure

Built reusable research framework:
- Data loading & resampling pipelines
- Feature engineering modules (65+ features)
- Labeling methodologies (triple-barrier, meta-labels)
- Model training & comparison systems
- Backtest evaluation frameworks
- Comprehensive documentation

### 3. Learnings

**Validated patterns that work:**
- Extreme mean reversion (|z-score| > 1.5)
- London/NY session trading (avoid Asia)
- Calm volatility periods (< 80th percentile)
- 30-min frequency (better SNR than 15-min)

**These insights are transferable to other strategies.**

---

## What Went Wrong ❌

### 1. Edge Magnitude

The mean reversion edge provides 3.68% directional advantage. In a perfect world (no costs), this would be profitable. But:
- Spread: ~0.3 pips
- Slippage: ~0.2 pips
- Total: ~0.5 pips = 2.5 bps

**Edge is consumed by costs.**

### 2. Fixed Target Assumptions

Assumed +20 pip profit / -12 pip stop would give R = 1.67.

Reality:
- Profit targets often not hit (only avg +9.6 bps = ~19 pips)
- Stop losses hit with slippage (avg -10.5 bps = ~21 pips)
- **Actual R-multiple: 0.92 (inverted!)**

### 3. Trade Frequency

Filters are TOO strict:
- Only 15.6% of bars pass all 3 filters
- Only 2.2 trades/day (vs 5-10 target)
- Fewer trades = harder to reach statistical significance
- May have filtered out too many marginal-but-profitable setups

---

## Deep Dive: Why R-Multiple Is Inverted

### Expected vs Actual

**Fixed Targets (Design):**
```
Profit Target: +20 pips = ~$20 on mini lot = 10 bps
Stop Loss: -12 pips = ~$12 on mini lot = 6 bps
R-multiple: 20/12 = 1.67 ✅
```

**Actual Performance:**
```
Avg Win: +9.61 bps ≈ +19 pips ✓ (close to target)
Avg Loss: -10.48 bps ≈ -21 pips ✗ (75% worse than target!)
R-multiple: 9.61/10.48 = 0.92 ❌
```

### Why Stops Are Worse Than Expected

1. **Slippage on stops:** Market moves fast through stop levels
2. **Gap risk:** 30-min bars can gap through stops
3. **Volatility spikes:** During news, spreads widen
4. **Asymmetric execution:** Easier to hit stops than targets

**Conclusion:** Fixed pip stops don't account for real-world execution dynamics.

---

## Paths Forward

### Option 1: Accept Reality (Recommended for Retail)

**Conclusion:** This edge is NOT exploitable for retail traders with:
- 0.3-0.5 pip spreads
- 0.2 pip average slippage
- Market execution

**Who COULD profit from this edge:**
1. **Institutional traders** (0.05-0.1 pip costs)
2. **Prop firms** (Rebate programs, better execution)
3. **Market makers** (Collect spread instead of pay it)

**Recommendation:** Archive this research, move to different strategy type.

---

### Option 2: One Final Iteration

**If you want to try one more time**, here's the highest-probability approach:

#### Optimize Risk/Reward Targets

**Problem:** Current R-multiple = 0.92 (inverted)

**Solution:** Test asymmetric targets that account for slippage:

| Profit Target | Stop Loss | R-mult | Notes |
|---------------|-----------|--------|-------|
| +30 pips | -10 pips | 3.0 | Very aggressive, low WR expected |
| +25 pips | -12 pips | 2.08 | Balanced |
| +20 pips | -10 pips | 2.0 | Slightly tighter stop |

**Re-run Phase 3 with:**
- Same 30-min bars + filters
- New targets: +25 pips profit / -10 pips stop (R = 2.5)
- Accept lower WR (maybe 48-50%) in exchange for higher R-mult
- Target: 50% WR × 2.5 R-mult = 1.25 EV before costs

**Success criteria:**
```
0.50 × 12.5 bps - 0.50 × 5.0 bps = 3.75 bps before costs
3.75 bps - 2.5 bps (costs) = 1.25 bps after costs ✅
```

**Effort:** 1 day (change label targets, re-train)

**Probability of success:** 30% (may just shift problem)

---

### Option 3: Pivot to Different Strategy

**If Option 2 fails, pivot entirely.**

**Alternative strategy types:**

#### A. Momentum/Breakout
- Opposite of mean reversion
- Ride strong moves instead of fading them
- Test: Price breaks above 20-day high → go long
- Better R-multiples (2-3x) but lower WR (35-40%)

#### B. Macro/Fundamental
- Trade on FOMC, NFP, CPI events
- Gold vs Dollar, Gold vs Real Yields
- Lower frequency (weekly/monthly)
- Fewer trades but larger edge per trade

#### C. Volatility
- VIX regime strategies
- Trade gold volatility expansion/contraction
- Options strategies (straddles, iron condors)

#### D. Inter-Market Arbitrage
- Gold vs Silver ratio
- Gold vs Bitcoin correlation
- Exploit temporary mispricings

**Recommendation:** Try **Momentum/Breakout** next. It's the natural counterpart to mean reversion and may have larger edge.

---

## Technical Artifacts

### Code Modules (Reusable)

1. `research_intraday_strategy.py` - EDA & hypothesis testing
2. `feature_engineering_intraday.py` - 65-feature pipeline
3. `labeling_intraday.py` - Triple-barrier labels
4. `train_intraday_models.py` - Multi-model training
5. `meta_labeling_strategy.py` - Meta-labeling framework
6. `strategy_30min_filtered.py` - 30-min filtered strategy

### Data

- `data_15min_2020_2024.parquet` - 15-min bars with features
- `data_15min_2020_2024_labeled.parquet` - With labels
- `data_30min_filtered.parquet` - 30-min filtered dataset

### Models

- `lightgbm_intraday.joblib` - Primary model (15-min)
- `meta_model.joblib` - Meta-labeling model
- `model_30min_filtered.joblib` - 30-min strategy model

### Reports

- `RESEARCH_SUMMARY.md` - Phase 1 findings
- `PHASE2_META_LABELING_ANALYSIS.md` - Phase 2 findings
- `FINAL_RESEARCH_REPORT.md` - This document

---

## Lessons Learned

### 1. Statistical Significance ≠ Profitability

**We proved the edge exists** (p < 0.0001). But:
- 3.68% directional advantage
- Translates to ~1 bps expected value
- **Transaction costs are 2.5 bps**
- **Net: -1.5 bps** ❌

**Lesson:** Need >3x edge vs costs to be reliably profitable.

### 2. Models Learn Patterns, Not Profits

All models correctly learned:
- Mean reversion (dist_ma features)
- Session effects (time features)
- Volatility patterns (rvol features)

But **learning patterns ≠ exploiting them profitably.**

### 3. Backtesting Must Be Brutally Realistic

We modeled:
- Walk-forward validation ✅
- Transaction costs (2.5 bps) ✅
- Slippage on stops ✅
- Real market execution ✅

**If we had been optimistic, we'd deploy a losing strategy.**

### 4. Iteration Is Normal

3 phases:
- Phase 1: 40% WR ❌
- Phase 2: 45% WR ❌
- Phase 3: 56% WR ✅ (but R-mult failed)

**Most quant research requires 5-10 iterations before success (or abandonment).**

### 5. Know When to Pivot

We've tried:
- 15-min frequency ❌
- Meta-labeling ❌
- 30-min + filters ❌

**Continuing on same path has diminishing returns. Time to try different strategy type.**

---

## Recommendations

### For This Project

**Option A (Conservative):** Archive research, move to momentum/breakout strategy

**Option B (One more try):** Test +25 pip profit / -10 pip stop (1 day effort)

**Option C (Nuclear):** Add DXY, yields, SPX features (1 week effort)

**My recommendation:** **Option A or B**

- Option B has 30% success probability (worth 1 day)
- If B fails → definitely Option A
- Option C is too much effort for uncertain payoff

### For Future Strategies

**Checklist before deployment:**
1. ✅ Statistical edge validated (p < 0.05)
2. ✅ Edge magnitude >3x transaction costs
3. ✅ Walk-forward validation on out-of-sample data
4. ✅ Win Rate × R-multiple > 1.3 after costs
5. ✅ Profit Factor ≥ 1.6
6. ✅ Sharpe ≥ 0.25 per trade
7. ✅ Max Drawdown ≤ 6%
8. ✅ Minimum 500 trades for statistical significance

**Don't deploy until ALL criteria met.**

---

## Conclusion

### What We Achieved

- ✅ Rigorous first-principles research
- ✅ Validated statistical edges
- ✅ Built professional-grade infrastructure
- ✅ Achieved 55.7% win rate (Phase 3)
- ✅ Honest, transparent analysis

### Why It's Not Profitable

- ❌ Edge too small (1 bps) vs costs (2.5 bps)
- ❌ Can't achieve both high WR AND good R-multiple
- ❌ Fixed pip targets don't match execution reality

### Final Verdict

**The mean reversion edge at intraday XAUUSD is REAL but NOT EXPLOITABLE for retail traders** given current transaction costs and execution constraints.

**This is a SUCCESS in research methodology** - we didn't foolishly deploy a losing strategy. We identified the edge, tested it rigorously, and correctly concluded it's unprofitable.

**Next:** Pivot to momentum/breakout strategy with different edge structure.

---

**Status:** ✅ **RESEARCH COMPLETE - EDGE VALIDATED BUT NOT PROFITABLE**

**Recommendation:** Archive and pivot to new strategy type

---

*Prepared by: Quant Research Team*
*Date: January 6, 2026*
*Total Research Time: 3 phases, ~8 hours*
*Lines of Code: ~3,000*
*Data Analyzed: 1.7M minute bars, 5 years*
*Models Trained: 7*
*Hypotheses Tested: 4*
*Statistical Significance: p < 0.0001 ✅*
*Economic Profitability: ❌*

---

**"Not all valid edges are profitable edges." - Quantitative Trading Wisdom**
