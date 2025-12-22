#!/usr/bin/env python3
"""
Out-of-sample test on freshly downloaded Dec 8-22, 2025 data.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features_complete import build_complete_features

print("=" * 70)
print("  OUT-OF-SAMPLE TEST: December 8-22, 2025 (FRESHLY DOWNLOADED)")
print("=" * 70)

# Load model
model_path = Path(__file__).parent.parent / "models" / "y_tb_60_hgb_tuned.joblib"
artifact = joblib.load(model_path)
model = artifact["model"]
feature_cols = artifact["features"]
print(f"\nModel loaded: {len(feature_cols)} features")

# Load data
print("\nLoading data...")
bars = pd.read_parquet("/Users/omar/Desktop/ML/Data/ohlcv_minute/XAUUSD_minute_2025.parquet")
quotes = pd.read_parquet("/Users/omar/Desktop/ML/Data/quotes/XAUUSD_quotes_2025.parquet")

bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
quotes["timestamp"] = pd.to_datetime(quotes["timestamp"], utc=True)

print(f"  Bars: {len(bars):,} rows, {bars['timestamp'].min()} to {bars['timestamp'].max()}")

# Filter to Dec 8-22 only
bars_new = bars[bars["timestamp"] >= "2025-12-08"].copy()
print(f"  Dec 8-22 bars: {len(bars_new):,}")

# Need some lookback for features - get data from Dec 1
bars_with_lookback = bars[bars["timestamp"] >= "2025-12-01"].copy()
quotes_with_lookback = quotes[quotes["timestamp"] >= "2025-12-01"].copy()

print(f"  With lookback (Dec 1+): {len(bars_with_lookback):,} bars")

# Build features
print("\nBuilding features...")
bars_with_lookback = bars_with_lookback.set_index("timestamp").sort_index()
quotes_with_lookback = quotes_with_lookback.set_index("timestamp").sort_index()

df = build_complete_features(bars_with_lookback, quotes_with_lookback)
print(f"  Features built: {len(df):,} rows")

# Filter to only Dec 8-22 (after features built)
df = df[df.index >= "2025-12-08"]
print(f"  Dec 8-22 with features: {len(df):,} rows")

# Drop rows without labels
df = df.dropna(subset=["y_tb_60"])
df = df[df["y_tb_60"] != 0]  # Remove neutral
print(f"  After dropping NaN/neutral labels: {len(df):,} rows")

# Prepare X, y
X = df[feature_cols].values
y = df["y_tb_60"].values
close = df["close"].values

print(f"\n  Date range: {df.index.min()} to {df.index.max()}")
print(f"  Labels: +1={sum(y==1):,}, -1={sum(y==-1):,}")

# Generate predictions
proba = model.predict_proba(X)[:, 1]

# Thresholds
THRESH_LONG = 0.75
THRESH_SHORT = 0.20

signals = np.zeros(len(proba))
signals[proba >= THRESH_LONG] = 1
signals[proba <= THRESH_SHORT] = -1

print(f"  Signals: LONG={int(sum(signals==1)):,}, SHORT={int(sum(signals==-1)):,}, FLAT={int(sum(signals==0)):,}")

# Simulate
print("\n" + "=" * 70)
print("  SIMULATION RESULTS")
print("=" * 70)

equity = 25000.0
peak = equity
max_dd = 0
trades = []
RISK_PCT = 0.0025

for i in range(len(signals)):
    if signals[i] == 0:
        continue
    
    signal = int(signals[i])
    actual = int(y[i])
    
    # R-multiple: +1 if correct direction, -1 if wrong
    r_mult = 1.0 if signal == actual else -1.0
    
    pnl = equity * RISK_PCT * r_mult
    equity += pnl
    peak = max(peak, equity)
    dd = (peak - equity) / 25000 * 100
    max_dd = max(max_dd, dd)
    
    trades.append({
        "timestamp": df.index[i],
        "signal": signal,
        "actual": actual,
        "r_mult": r_mult,
        "pnl": pnl,
        "equity": equity,
    })

# Results
n_trades = len(trades)
wins = sum(1 for t in trades if t["r_mult"] > 0)
losses = n_trades - wins
win_rate = wins / n_trades * 100 if n_trades > 0 else 0
avg_r = np.mean([t["r_mult"] for t in trades]) if trades else 0
profit = equity - 25000
profit_pct = profit / 250

print(f"""
  Starting Balance:  $25,000.00
  Ending Equity:     ${equity:,.2f}
  Profit:            ${profit:+,.2f} ({profit_pct:.2f}%)
  Max Drawdown:      {max_dd:.2f}%

  Total Trades:      {n_trades}
  Wins:              {wins}
  Losses:            {losses}
  Win Rate:          {win_rate:.1f}%
  Average R:         {avg_r:+.4f}
""")

# Check status
if equity >= 26250:
    print("  ★★★ PASSED (5% target reached) ★★★")
elif equity <= 23500:
    print("  ✗✗✗ FAILED (6% drawdown breached) ✗✗✗")
else:
    print("  ◆◆◆ IN PROGRESS ◆◆◆")

print("\n" + "=" * 70)

# Show recent trades
if trades:
    print("\nRecent trades:")
    for t in trades[-10:]:
        direction = "LONG" if t["signal"] == 1 else "SHORT"
        result = "WIN" if t["r_mult"] > 0 else "LOSS"
        print(f"  {t['timestamp']} | {direction:5} | {result:4} | ${t['pnl']:+.2f} | Equity: ${t['equity']:,.2f}")

