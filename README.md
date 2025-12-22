# XAUUSD Condition-Based Signal Engine

A Dr. Chen–style, condition-based signal generation system for XAUUSD (Gold) trading using Polygon.io historical data.

## Purpose

This system **does not predict price directly**. Instead, it predicts **market conditions** under which long/short/no-trade decisions for each horizon (5m, 15m, 30m) have **positive expectancy**, incorporating:

- Spread cost awareness
- Volatility-based Stop Loss / Take Profit levels
- Environment classification (+1 = long-favorable, 0 = no-trade, -1 = short-favorable)

## Data Requirements

### Source
Polygon.io historical data (seconds + minutes OHLCV + top-of-book quotes) from 2014-01-01 to 2025-12-07.

### Expected Files

1. **Second-level OHLCV** (`XAUUSD_second_YYYY.parquet`):
   - Columns: `timestamp` (UTC), `open`, `high`, `low`, `close`, `volume`

2. **Minute-level OHLCV** (`XAUUSD_minute_YYYY.parquet`):
   - Columns: `timestamp` (UTC), `open`, `high`, `low`, `close`, `volume`

3. **Top-of-book Quotes** (`XAUUSD_quotes_YYYY.parquet`):
   - Columns: `timestamp` (UTC), `bid_price`, `bid_size`, `ask_price`, `ask_size`

All timestamps are in UTC and represent the **end of the bar** for OHLCV data.

## Key Concepts

### Mid Price
```
mid = (bid_price + ask_price) / 2
```

### Spread & Cost
```
spread = ask_price - bid_price
spread_pct = spread / mid
```

### Volatility-Based SL/TP
For each horizon, we use volatility (σ) multipliers:
```
SL_ret = -k1 * σ
TP_ret = +k2 * σ
TP_eff = TP_ret + spread_pct  # Cost-adjusted take profit
```

### Environment Classification Labels
- **+1**: Long-favorable environment (TP reached before SL)
- **0**: No-trade environment (neither condition met clearly)
- **-1**: Short-favorable environment (inverse of long)

### Horizons
- **5 minutes**: k1=1.0, k2=1.5
- **15 minutes**: k1=1.0, k2=2.0
- **30 minutes**: k1=1.0, k2=2.5

## Project Structure

```
xauusd_signals/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py              # All configuration constants
│   ├── data_loader.py         # Data loading and alignment
│   ├── feature_engineering.py # Feature computation
│   ├── labeling.py            # Label generation
│   ├── model_training.py      # XGBoost model training
│   ├── signal_generator.py    # Signal generation
│   ├── backtest.py            # Backtesting simulation
│   ├── evaluation.py          # Metrics computation
│   └── utils/
│       ├── __init__.py
│       ├── time_utils.py
│       ├── plotting_utils.py
│       └── metrics.py
├── notebooks/
│   └── exploration.ipynb
└── models/                    # Trained model storage
```

## Installation

```bash
# Create virtual environment (Python 3.11 recommended)
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.data_loader import get_combined_dataset
from src.feature_engineering import build_feature_matrix
from src.labeling import generate_labels_for_all_horizons
from src.model_training import train_all_horizon_models

# Load and prepare data
df = get_combined_dataset(
    minute_path="Data/ohlcv_minute/XAUUSD_minute_2024.parquet",
    quotes_path="Data/quotes/XAUUSD_quotes_2024.parquet"
)

# Build features
df = build_feature_matrix(df)

# Generate labels
df = generate_labels_for_all_horizons(df)

# Train models
train_all_horizon_models(df, model_dir="models/")
```

## Signal Generation

```python
from src.signal_generator import generate_signals_for_latest_row
from src.config import VOL_PARAMS, FEATURE_COLUMNS

signals = generate_signals_for_latest_row(
    df=df,
    model_dir="models/",
    feature_cols=FEATURE_COLUMNS,
    vol_params=VOL_PARAMS
)
# Returns: {"timestamp": ..., "mid": ..., "signals": {"5m": {...}, "15m": {...}, "30m": {...}}}
```

## Important Notes

### Time Consistency (No Look-Ahead Bias)
- All features use only past/current data at time t
- Labels use future data only for training/validation
- Live signal generation never accesses future information

### Model Architecture
- XGBoost multiclass classifier (multi:softprob)
- Three independent models for each horizon
- Time-series train/test split (no shuffling)

## License

MIT License - See LICENSE file for details.

