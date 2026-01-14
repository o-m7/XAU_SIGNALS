# Model 5: Statistical Mean Reversion (XAUUSD 15M)

## Overview

Pure statistical mean reversion model for XAUUSD on 15-minute timeframe. **NO technical analysis indicators, NO market narratives** - only mathematically defined features with testable hypotheses.

**Key Characteristics:**
- LONG only trading
- Maximum holding period: 15 bars (3.75 hours)
- All features have mathematical formulas
- Statistical validation required before backtesting
- Every filter has a statistical test

---

## Project Structure

```
src/models/model5/
├── __init__.py          # Module exports
├── config.py            # Configuration parameters
├── features.py          # Feature engineering (statistically grounded)
├── labels.py            # Target variable creation
├── signals.py           # Signal generation logic
├── backtest.py          # Walk-forward backtesting
├── validation.py        # Statistical validation tests
├── data_loader.py       # Load and resample data
└── train.py             # Model training pipeline
```

---

## Features

All features are mathematically defined:

### Price Statistics
- `returns_1`, `returns_2`, `returns_4`: Multi-period returns
- `zscore_20`: Rolling z-score of price
- `percentile_20`: Rolling percentile rank

### Volatility Measures
- `atr_14`: Average True Range (EMA)
- `parkinson_vol_20`: Parkinson volatility estimator
- `returns_std_20`: Rolling standard deviation of returns

### Mean Reversion Indicators
- `variance_ratio_2`, `variance_ratio_4`: Lo-MacKinlay variance ratios
- `autocorr_1`, `autocorr_2`: Rolling autocorrelations

### Candle Structure
- `body_ratio`: |C-O| / (H-L)
- `close_location`: (C-L) / (H-L)

### Distribution Statistics
- `returns_skew_20`: Skewness of returns
- `returns_kurtosis_20`: Excess kurtosis

### Quote Features (if available)
- `spread_mean_bps`: Average spread in basis points
- `spread_zscore`: Spread vs recent history
- `quote_imbalance`: Directional quote activity

### Intrabar Features (if available)
- `intrabar_rv`: Realized variance
- `intrabar_jump`: Jump component (RV - BV)
- `intrabar_reversals`: Sign change frequency

### Time Features
- `hour_sin`, `hour_cos`: Cyclical hour encoding
- `is_overlap`: London-NY overlap session

---

## Usage

### Basic Training Pipeline

```python
from src.models.model5.train import run_pipeline
from src.models.model5.config import Model5Config

# Run full pipeline
results = run_pipeline(
    data_15m_path="data/xauusd_15m.parquet",
    data_quotes_path="data/xauusd_quotes.parquet",  # Optional
    data_seconds_path="data/xauusd_1s.parquet",    # Optional
    config=Model5Config(),
    output_dir="models/model5"
)
```

### Command Line

```bash
# Basic usage
python -m src.models.model5.train data/xauusd_15m.parquet

# With quotes and second data
python -m src.models.model5.train \
    data/xauusd_15m.parquet \
    --quotes data/xauusd_quotes.parquet \
    --seconds data/xauusd_1s.parquet \
    --output models/model5
```

### Individual Components

```python
from src.models.model5 import (
    Model5Config,
    build_all_features,
    run_all_validations,
    Model5SignalEngine,
    run_backtest
)

# Load data
from src.models.model5.data_loader import load_all_data
df_15m, df_quotes, df_seconds = load_all_data(
    "data/xauusd_minute.parquet",
    "data/xauusd_quotes.parquet",
    "data/xauusd_1s.parquet"
)

# Build features
config = Model5Config()
df = build_all_features(df_15m, df_quotes, df_seconds, config)

# Validate
validation_results = run_all_validations(df, verbose=True)

# Generate signals
engine = Model5SignalEngine(config)
signal = engine.generate_signal(df.iloc[-1])

# Backtest
result = run_backtest(df, config, verbose=True)
```

---

## Validation Criteria

Before deploying, ALL must pass:

| Test | Requirement | How to Check |
|------|-------------|--------------|
| Variance Ratio | VR(2) < 1, p < 0.05 | `test_variance_ratio()` |
| Autocorrelation | ρ₁ < 0, p < 0.05 | `test_autocorrelation()` |
| Z-score Predictability | P(reversal \| z < -2) > 55%, p < 0.05 | `test_zscore_predictability()` |
| Backtest Win Rate | > 53% | Backtest |
| Backtest Profit Factor | > 1.1 | Backtest |
| Backtest T-stat | > 2.0 (p < 0.05) | Backtest |
| Sample Size | > 200 trades | Backtest |

---

## Signal Logic

**LONG Entry Conditions:**
1. `zscore_20 < -2.0` (oversold)
2. `variance_ratio_2 < 0.95` (mean reversion regime)
3. `spread_percentile < 80` (not illiquid)
4. `atr_percentile` between 20 and 90 (reasonable volatility)
5. `is_active == 1` (during trading hours)
6. R:R ratio >= 0.8

**Exit Conditions:**
- Target hit: Price reaches mean reversion target
- Stop loss: 1.5 ATR below entry
- Time stop: 15 bars maximum hold

---

## Configuration

Key parameters in `Model5Config`:

```python
config = Model5Config(
    zscore_entry_threshold=2.0,      # Z-score for entry
    max_variance_ratio=0.95,         # VR threshold
    max_spread_percentile=80,        # Spread filter
    stop_atr_multiple=1.5,           # Stop loss
    max_bars_in_trade=15,            # Time stop
    min_rr_ratio=0.8,                # Min reward:risk
    cooldown_bars=2,                 # Bars between trades
)
```

---

## Key Principles

1. **Every feature has a formula** - No hand-waving
2. **Every filter has a statistical test** - Validate before using
3. **No TA indicators** - RSI, MACD, Bollinger Bands are banned
4. **No market narratives** - No "liquidity", "smart money", "support/resistance"
5. **Statistical significance required** - p < 0.05 or don't trade
6. **LONG only** - Simpler model, one direction
7. **15M bars, max 15 bars hold** - As specified

---

## References

- Lo, A. W., & MacKinlay, A. C. (1988). Stock market prices do not follow random walks: Evidence from a simple specification test. *Review of Financial Studies*, 1(1), 41-66.
- Parkinson, M. (1980). The extreme value method for estimating the variance of the rate of return. *Journal of Business*, 53(1), 61-65.
- Garman, M. B., & Klass, M. J. (1980). On the estimation of security price volatilities from historical data. *Journal of Business*, 53(1), 67-78.

---

## Notes

- If validation tests fail, the strategy has no edge. Don't backtest garbage.
- All features are computed without lookahead bias.
- The model requires statistical evidence of mean reversion before trading.

