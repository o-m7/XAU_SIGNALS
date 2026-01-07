# XAUUSD Intraday Strategy Research - Summary Report

**Date:** 2026-01-06
**Objective:** Develop profitable 15-30min strategy meeting PF ≥ 1.6, WR ≥ 52%, Sharpe ≥ 0.25, DD ≤ 6%
**Approach:** First-principles research with rigorous statistical validation

---

## Executive Summary

Completed systematic research to develop intraday XAUUSD strategy from first principles. **Validated statistical edges exist**, but current implementation **does not meet profitability targets**. The mean reversion and session effects are statistically significant (p < 0.05), but converting these to profitable ML models requires iteration on labeling methodology and model architecture.

### Key Findings
✅ **Statistical edges validated** (mean reversion p < 0.0001, session effects p = 0.001)
✅ **Feature engineering captures validated edges** (65 features)
❌ **Current models fail to meet targets** (AUC ≈ 0.52, WR ~40%)
⚠️ **Label design is bottleneck** (triple-barrier produces 40% WR vs 52% target)

---

## 1. Data Exploration (Phase 1)

### Dataset
- **Period:** 2020-2024 (5 years)
- **Frequency:** 15-minute bars
- **Observations:** 116,382 bars
- **Avg bars/day:** 92.4

### Statistical Properties
- **Mean return:** +0.0523 bps per 15-min bar (slight upward drift)
- **Std deviation:** 10.35 bps (high intraday volatility)
- **Skewness:** -0.76 (left tail, outlier crashes)
- **Kurtosis:** 57.86 (extreme fat tails - expect rare but large moves)
- **Normality:** REJECTED (p < 0.001) - not normal distribution
- **Stationarity:** Returns are stationary ✅ (ADF p < 0.001)

---

## 2. Edge Hypothesis Testing (Phase 2)

### ✅ HYPOTHESIS 1: MEAN REVERSION EDGE (VALIDATED)

**Statistical Evidence:**
- **Lag-1 ACF:** -0.0385, p < 0.0001 ⭐⭐⭐
- **Ljung-Box test:** p < 0.0001 at lags 1, 5, 10, 20
- **Directional persistence:** Chi² = 157.20, p < 0.0001

**Effect Size:**
- P(Up | Previous Down) = **51.92%**
- P(Up | Previous Up) = **48.24%**
- **Edge:** 3.68% higher probability of reversal

**Interpretation:** Strong short-term mean reversion. Prices tend to reverse after moves.

---

### ✅ HYPOTHESIS 2: SESSION EFFECTS (VALIDATED)

**Statistical Evidence:**
- **ANOVA:** F = 5.32, p = **0.001** ⭐⭐

**Returns by Session:**
| Session | Mean Return (bps) | Volatility (bps) | Win % |
|---------|-------------------|------------------|-------|
| After-hours | +0.47 | 11.05 | 51.4% |
| Asia/Pacific | +0.03 | 7.75 | 50.0% |
| NY | +0.02 | 9.05 | 49.4% |
| London | +0.01 | 12.50 | 50.3% |

**Interpretation:** Sessions differ significantly. London has highest volatility, After-hours has best returns.

---

### ✅ HYPOTHESIS 3: VOLATILITY CLUSTERING (VALIDATED)

**Statistical Evidence:**
- **ARCH test:** LM = 2744.31, p < 0.0001 ⭐⭐⭐
- **Interpretation:** Volatility is predictable and clusters (high vol follows high vol)

**Application:** Use for dynamic position sizing and risk management.

---

### ❌ HYPOTHESIS 4: VOLATILITY REGIME DEPENDENCY (REJECTED)

**Statistical Evidence:**
- **ANOVA:** F = 0.26, p = **0.77** (not significant)

**Interpretation:** Strategy performance does NOT depend on volatility regime. Edge works across regimes.

---

## 3. Feature Engineering (Phase 3)

### Feature Categories (65 total features)

| Category | Count | Key Features |
|----------|-------|--------------|
| **Price/Mean Reversion** | 18 | ROC, dist_ma, z-score, pct_rank |
| **Volatility** | 21 | rvol, ATR, Parkinson vol, vol_rank |
| **Momentum** | 10 | lagged returns, cum_ret, reversion_signal |
| **Volume** | 10 | volume ratios, volume rank |
| **Microstructure** | 5 | close_position, shadows, body_size |
| **Session/Time** | 9 | session indicators, hour, day_of_week |

### Design Principles
✅ All features computed on bar close (no lookahead bias)
✅ Capture validated statistical edges
✅ Normalized for regime independence
✅ Multi-timeframe awareness (optional)

---

## 4. Label Design (Phase 4)

### Triple-Barrier Method

Tested 4 configurations:

| Config | Profit Target | Stop Loss | Win Rate | R-multiple | Score |
|--------|---------------|-----------|----------|------------|-------|
| **Aggressive** | 2.0σ | 1.0σ | 40.45% | **1.53** ✅ | **BEST** |
| Default | 1.5σ | 1.0σ | 45.45% | 1.24 ✅ | Good |
| Conservative | 1.2σ | 0.8σ | 45.91% | 1.21 ✅ | Good |
| Tight | 1.0σ | 0.5σ | 43.21% | 1.34 ✅ | OK |

### Best Configuration: AGGRESSIVE
- **Win Rate:** 40.45% ❌ (below 52% target)
- **R-multiple:** 1.53 ✅ (above 1.2 target)
- **Avg Hold:** 6.2 bars (~1.5 hours)

### Label Distribution
- **Long signals:** 40.4%
- **Short signals:** 59.5%
- **Neutral:** 0.1%

### ⚠️ PROBLEM IDENTIFIED
**None of the label configurations achieve 52% win rate.** Best is 45.91% (conservative).

This indicates that:
1. Triple-barrier method may not be optimal for this edge
2. Need more selective entry criteria (meta-labeling)
3. Or different label structure (fixed pip targets, time-based, etc.)

---

## 5. Model Training (Phase 5)

### Models Tested
1. **LightGBM** (gradient boosting)
2. **XGBoost** (gradient boosting)
3. **Random Forest** (ensemble)
4. **Logistic Regression** (baseline)

### Results

| Model | AUC | Accuracy | Precision | Recall | F1 |
|-------|-----|----------|-----------|--------|-----|
| **LightGBM** | **0.521** | 56.6% | 62.8% | **0.9%** | 0.018 |
| XGBoost | 0.517 | 56.2% | 47.2% | 4.7% | 0.086 |
| Random Forest | 0.521 | 56.7% | 59.4% | 1.6% | 0.032 |
| Logistic Reg | 0.518 | 56.4% | 40.8% | 0.5% | 0.010 |

### ⚠️ CRITICAL ISSUE: MODELS FAIL TO MEET TARGETS

**Problem:** All models have:
- **AUC ≈ 0.52** (barely better than random 0.50)
- **Extremely low recall** (<5%) - models predict almost NO trades
- **Models are overly conservative** - they learned "when in doubt, don't trade"

**Why this happened:**
1. Labels have only 40% win rate - models can't beat this
2. Class imbalance (40% positive / 60% negative)
3. Features may need more engineering
4. Need different modeling approach (meta-labeling, probability thresholds)

### Top Features by Importance

**LightGBM:**
1. minutes_since_midnight (time of day - session effect!)
2. dist_ma_50 (mean reversion signal!)
3. rvol_50 (volatility - position sizing!)
4. roc_50 (rate of change - reversion!)
5. volume_ma_20

**XGBoost:**
1. session (validated session edge!)
2. dist_ma_50 (mean reversion!)
3. close_position (microstructure)
4. trades_ma_10
5. day_of_week (time effect)

**✅ Models ARE learning the validated edges** (mean reversion, session, time, volatility).
**❌ But current labels prevent profitable exploitation.**

---

## 6. Performance vs. Targets

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| **Profit Factor** | ≥ 1.6 | **Not calculable** | ❌ (too few predictions) |
| **Win Rate** | ≥ 52% | **40-46%** (labels) | ❌ FAIL |
| **Sharpe/trade** | ≥ 0.25 | **0.01-0.02** (labels) | ❌ FAIL |
| **Max Drawdown** | ≤ 6% | **Not tested** | ⏸️ Pending |
| **R-multiple** | > 1.2 | **1.53** (aggressive) | ✅ PASS |
| **Trades/day** | 15-30 | **92 opportunities** | ✅ PASS |

### Current Status: **NOT READY FOR LIVE DEPLOYMENT**

---

## 7. Recommendations & Next Steps

### Immediate Actions

#### A. **Improve Label Design** (HIGHEST PRIORITY)

**Problem:** Triple-barrier labels produce only 40-46% win rate.

**Solutions to test:**

1. **Meta-labeling approach:**
   - Primary model: predicts DIRECTION (current labels)
   - Meta-model: predicts WHEN to trade (filters low-confidence signals)
   - Hypothesis: This can boost win rate to >52% by being more selective

2. **Fixed pip targets:**
   - Instead of volatility-scaled barriers, use fixed pip targets
   - Test: +15 pip profit / -10 pip stop (1.5 R-multiple)
   - May produce more stable win rates

3. **Time-weighted labels:**
   - Label = profitability over next N bars (continuous)
   - Then convert to binary based on threshold
   - Captures edge better than hit-or-miss barriers

4. **Forward returns + quantiles:**
   - Label top 40% forward returns as "long", bottom 40% as "short"
   - Middle 20% as "neutral" (don't trade)
   - Ensures balanced classes by construction

#### B. **Model Improvements**

1. **Probability threshold optimization:**
   - Don't use default 0.5 threshold
   - Find optimal threshold that maximizes (WR × R-multiple)
   - Example: If threshold=0.7, model only trades when 70% confident

2. **Ensemble meta-learner:**
   - Stack LightGBM + XGBoost predictions
   - Train meta-learner on "should I take this signal?"
   - Acts as confidence filter

3. **Cost-sensitive learning:**
   - Modify loss function to penalize false positives more
   - Forces model to be selective (higher precision)

#### C. **Feature Engineering Round 2**

1. **Add signal strength indicators:**
   - How far is price from MA? (already have dist_ma_50)
   - Consecutive bars in same direction
   - Volume confirmation (high volume on reversal?)

2. **Regime indicators:**
   - Even though regime ANOVA failed, create binary "high_vol" feature
   - Model might find non-linear relationships

3. **Interaction features:**
   - dist_ma_50 × session (reversion stronger in certain sessions?)
   - roc_50 × rvol_50 (big moves in low vol = stronger reversion?)

#### D. **Alternative Approaches**

1. **Rule-based filters + ML confirmation:**
   - Hard rules: "Only trade when |roc_50| > 2σ AND session = London"
   - Then use ML to decide long vs short
   - Reduces sample size but increases signal quality

2. **Reinforcement learning:**
   - Frame as RL problem: agent learns optimal entry/exit
   - Reward = (PF × WR) - DrawdownPenalty
   - May discover non-obvious patterns

3. **Survival analysis:**
   - Model "time to profit target" and "time to stop"
   - Use to optimize holding periods

---

### Walk-Forward Validation Plan (WHEN ready)

Once model achieves WR ≥ 52% on static test set:

1. **Train/test windows:**
   - Train: 12 months
   - Test: 3 months
   - Step: 1 month (rolling window)
   - Minimum 20 windows

2. **Transaction costs:**
   - Spread: 0.3 pips (realistic for XAUUSD)
   - Slippage: 0.2 pips per side
   - Total cost: ~0.5 pips per round-trip

3. **Position sizing:**
   - ATR-based: risk 0.5% per trade
   - Max 2 concurrent positions
   - Enforce 6% account DD hard stop

4. **Performance by regime:**
   - Calculate PF, WR, Sharpe for each of:
     - Low / Medium / High volatility
     - London / NY / Asia sessions
     - Trending / Ranging markets
   - Require PF > 1.4 in ALL regimes (or document where edge breaks)

5. **Robustness tests:**
   - Parameter sensitivity (±10% on all hyperparameters)
   - Random permutation test (shuffle labels, should fail)
   - Synthetic data test (if works on synthetic, might be overfit)

---

## 8. Technical Artifacts Generated

### Code Modules
- ✅ `research_intraday_strategy.py` - EDA and hypothesis testing
- ✅ `feature_engineering_intraday.py` - Feature pipeline
- ✅ `labeling_intraday.py` - Label generation and optimization
- ✅ `train_intraday_models.py` - Model training and comparison

### Data Artifacts
- ✅ `data_15min_2020_2024.parquet` - Raw resampled data
- ✅ `data_15min_2020_2024_features.parquet` - Featured dataset
- ✅ `data_15min_2020_2024_labeled.parquet` - Labeled dataset
- ✅ `label_config_comparison.csv` - Label configuration results
- ✅ `model_comparison.csv` - Model performance comparison

### Models
- ✅ `lightgbm_intraday.joblib` - Best model (AUC 0.521)
- ✅ `xgboost_intraday.json`
- ✅ `randomforest_intraday.joblib`
- ✅ `logistic_regression_intraday.joblib`

### Feature Importance
- ✅ `lightgbm_feature_importance.csv`
- ✅ `xgboost_feature_importance.csv`
- ✅ `randomforest_feature_importance.csv`
- ✅ `logistic_regression_feature_importance.csv`

---

## 9. Conclusion

### What Worked
✅ **Systematic first-principles research** - validated edges statistically before building
✅ **Mean reversion edge is REAL** (p < 0.0001) - not data mining
✅ **Session effects are REAL** (p = 0.001) - London/NY/Asia differ
✅ **Feature engineering captures edges** - models learn time, reversion, volatility
✅ **Infrastructure is solid** - modular, reproducible, documented code

### What Didn't Work (Yet)
❌ **Label design** - triple-barrier produces only 40-46% WR (need 52%)
❌ **Model performance** - AUC 0.52 (barely better than random)
❌ **Profitability targets** - not met (PF, WR, Sharpe all below threshold)

### Key Insight
**The edge exists, but we haven't found the right way to exploit it yet.**

This is NORMAL in quant research. Statistical edges don't automatically translate to profitable ML models. Iteration required.

### Estimated Work to Profitability
- **Optimistic:** 2-3 iterations on labeling + meta-learning (1-2 weeks)
- **Realistic:** 5-10 iterations, may need different approach (4-6 weeks)
- **Pessimistic:** Edge too small to exploit profitably at 15-min (consider 5-min or different approach)

### Recommendation
**Do NOT deploy current models.** They will lose money (too few trades, no edge).

**Next iteration:** Implement meta-labeling approach (see Section 7A.1 above). This is most promising path to WR ≥ 52%.

---

## 10. Lessons Learned

1. **Statistical significance ≠ Profitability** - p < 0.05 proves edge exists, but exploiting it is different problem
2. **Label design is critical** - bad labels → bad models, no matter how good features are
3. **Class imbalance matters** - 40/60 split makes models conservative
4. **AUC 0.52 is not "close enough"** - need 0.60+ for profitable trading
5. **Feature importance validates research** - models ARE learning the edges we found
6. **Iteration is necessary** - first attempt rarely succeeds in quant research

---

## Appendix A: Statistical Test Results

### Mean Reversion
- **Test:** Autocorrelation (Ljung-Box)
- **Result:** p < 0.0001 at lags 1, 5, 10, 20
- **Effect:** Lag-1 ACF = -0.0385
- **Conclusion:** Reject null hypothesis (not random walk). Mean reversion exists.

### Session Effects
- **Test:** ANOVA (one-way)
- **Result:** F = 5.32, p = 0.001
- **Effect:** After-hours +0.47 bps vs London +0.01 bps
- **Conclusion:** Reject null hypothesis. Sessions differ.

### Volatility Clustering
- **Test:** ARCH LM test
- **Result:** LM = 2744.31, p < 0.0001
- **Effect:** High vol predicts high vol
- **Conclusion:** Strong ARCH effects. Volatility is predictable.

---

## Appendix B: File Locations

```
/Users/omar/Desktop/ML/xauusd_signals/
├── research_intraday_strategy.py          # EDA script
├── feature_engineering_intraday.py        # Feature pipeline
├── labeling_intraday.py                   # Label generation
├── train_intraday_models.py               # Model training
│
├── research_results/
│   ├── RESEARCH_SUMMARY.md                # This file
│   ├── data_15min_2020_2024.parquet
│   ├── data_15min_2020_2024_features.parquet
│   ├── data_15min_2020_2024_labeled.parquet
│   ├── label_config_comparison.csv
│   ├── model_comparison.csv
│   │
│   └── intraday_models/
│       ├── lightgbm_intraday.joblib
│       ├── xgboost_intraday.json
│       ├── randomforest_intraday.joblib
│       ├── logistic_regression_intraday.joblib
│       ├── *_feature_importance.csv
│
└── Raw Data/
    ├── ohlcv_minute/
    └── quotes/
```

---

**End of Research Summary**

*Prepared by: Quant Research Team*
*Date: 2026-01-06*
*Status: RESEARCH PHASE COMPLETE - NOT READY FOR DEPLOYMENT*
