#!/usr/bin/env python3
"""
Funded Account Backtest for Model 1

Tests Model 1 on a $50,000 account with:
- 1:30 leverage
- 1% risk per trade
- Dynamic ATR-based position sizing
- 10% profit target
- 4% max drawdown
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib

# Model path
MODEL_PATH = PROJECT_ROOT / "models" / "y_tb_60_2014_2023.joblib"
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"

# Test periods
TRAIN_START = "2020-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

# Signal thresholds
THRESHOLD_LONG = 0.60
THRESHOLD_SHORT = 0.30

# Account parameters
INITIAL_BALANCE = 50000.0
LEVERAGE = 10.0
FIXED_POSITION_SIZE = 0.25  # Fixed lot size per trade
PROFIT_TARGET_PCT = 0.10
MAX_DRAWDOWN_PCT = 0.04
MAX_BARS_IN_TRADE = 15  # 15 minutes

# Transaction costs (realistic)
SPREAD_COST = 0.0003  # ~3 pips spread for gold (0.03%)
SLIPPAGE_LONG = 0.0002  # 2 pips slippage on entry/exit
SLIPPAGE_SHORT = 0.0002  # 2 pips slippage on entry/exit


@dataclass
class FundedTrade:
    entry_time: object
    entry_price: float
    stop_price: float
    target_price: float
    position_size: float
    account_equity: float
    signal_type: str
    exit_time: object = None
    exit_price: float = 0
    exit_reason: str = ""
    bars_held: int = 0
    pnl: float = 0
    pnl_pct: float = 0


@dataclass
class FundedBacktestResult:
    initial_balance: float
    final_balance: float
    leverage: float
    risk_per_trade: float
    trades: List[FundedTrade]
    equity_curve: Optional[pd.Series] = None
    profit_target_hit: bool = False
    account_blown: bool = False
    
    @property
    def total_return_pct(self) -> float:
        if self.n_trades == 0:
            return 0.0
        return ((self.final_balance - self.initial_balance) / self.initial_balance) * 100
    
    @property
    def n_trades(self) -> int:
        return len(self.trades)
    
    @property
    def win_rate(self) -> float:
        if self.n_trades == 0:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / self.n_trades
    
    @property
    def profit_factor(self) -> float:
        if self.n_trades == 0:
            return float('inf')
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    @property
    def max_drawdown_pct(self) -> float:
        if self.equity_curve is None or len(self.equity_curve) == 0:
            return 0.0
        equity = self.equity_curve.values
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return abs(np.min(drawdown)) * 100
    
    @property
    def sharpe_ratio(self) -> float:
        if self.n_trades == 0 or self.equity_curve is None:
            return 0.0
        returns = self.equity_curve.pct_change().dropna()
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(252 * 24 * 60)
    
    @property
    def summary(self) -> str:
        status = "✅ PASSED" if self.profit_target_hit else ("❌ BLOWN" if self.account_blown else "⏸ INCOMPLETE")
        return f"""
============ FUNDED ACCOUNT BACKTEST (MODEL 1) ============
Status:              {status}
Initial Balance:     ${self.initial_balance:,.2f}
Final Balance:       ${self.final_balance:,.2f}
Total Return:        {self.total_return_pct:+.2f}%
Leverage:            {self.leverage}:1
Position Size:       0.25 lots (fixed)

Trades:              {self.n_trades:,}
Win Rate:            {self.win_rate:.1%}
Profit Factor:       {self.profit_factor:.2f}
Max Drawdown:        {self.max_drawdown_pct:.2f}%
Sharpe Ratio:        {self.sharpe_ratio:.3f}
==========================================================
"""


def calculate_position_size(
    account_equity: float,
    entry_price: float,
    stop_price: float,
    risk_per_trade: float,
    leverage: float
) -> float:
    """Calculate position size based on risk and ATR.
    
    Position size is in units (oz for gold). Leverage is applied when calculating margin,
    not when calculating position size.
    """
    risk_amount = account_equity * risk_per_trade
    price_risk = abs(entry_price - stop_price)
    
    if price_risk == 0:
        return 0
    
    # Position size in units (without leverage multiplication)
    position_size = risk_amount / price_risk
    
    return position_size


def run_funded_backtest(
    df: pd.DataFrame,
    model,
    feature_cols: List[str],
    initial_balance: float = 50000.0,
    leverage: float = 10.0,
    profit_target_pct: float = 0.10,
    max_drawdown_pct: float = 0.04,
    verbose: bool = True
) -> FundedBacktestResult:
    """Run funded account backtest for Model 1."""
    result = FundedBacktestResult(
        initial_balance=initial_balance,
        final_balance=initial_balance,
        leverage=leverage,
        risk_per_trade=0.0,  # Not used with fixed position size
        trades=[]
    )
    
    current_trade = None
    trade_entry_idx = 0
    account_equity = initial_balance
    peak_equity = initial_balance
    
    profit_target = initial_balance * (1 + profit_target_pct)
    max_drawdown_level = initial_balance * (1 - max_drawdown_pct)
    
    equity_values = []
    equity_times = []
    
    account_blown = False
    profit_target_hit = False
    
    # Batch predict all probabilities upfront (FAST!)
    if verbose:
        print("  [Pre-computing signals...]")
    available_features = [f for f in feature_cols if f in df.columns]
    if len(available_features) < len(feature_cols) * 0.8:
        if verbose:
            print(f"  ❌ Only {len(available_features)}/{len(feature_cols)} features available")
        return result
    
    X = df[available_features].values
    X = np.nan_to_num(X, nan=0.0)
    
    # Batch predict
    all_proba = model.predict_proba(X)
    classes = model.classes_
    if 1 in classes:
        up_idx = list(classes).index(1)
    else:
        up_idx = -1
    all_proba_up = all_proba[:, up_idx] if up_idx >= 0 else all_proba[:, -1]
    
    # Diagnostic counters
    signals_checked = 0
    signals_below_threshold = 0
    signals_above_threshold = 0
    signals_rejected_margin = 0
    signals_rejected_position_size = 0
    signals_rejected_price = 0
    signals_rejected_exception = 0
    proba_samples = []
    
    # Sample probabilities for diagnostics
    sample_indices = list(range(0, min(1000, len(df)), 10)) + list(range(1000, len(df), 100))
    proba_samples = [all_proba_up[i] for i in sample_indices if i < len(all_proba_up)]
    
    pbar = tqdm(total=len(df), desc="Backtesting", unit="bars", ncols=100, leave=True)
    
    # Loop through bars (skip last bar since we need i+1 for entry)
    for i in range(len(df) - 1):
        row = df.iloc[i]
        equity_values.append(account_equity)
        equity_times.append(row.name)
        
        if account_equity > peak_equity:
            peak_equity = account_equity
        
        # Check profit target
        if account_equity >= profit_target:
            if not profit_target_hit:
                profit_target_hit = True
                if verbose:
                    print(f"  ✅ Profit target hit at bar {i} ({row.name}): ${account_equity:,.2f}")
        
        # Check max drawdown
        if account_equity <= max_drawdown_level:
            if not account_blown:
                account_blown = True
                if verbose:
                    print(f"  ❌ Max drawdown breached at bar {i} ({row.name}): ${account_equity:,.2f}")
                if current_trade is not None:
                    current_trade.exit_time = row.name
                    current_trade.exit_price = row['close']
                    current_trade.exit_reason = "Account Blown"
                    price_change = (current_trade.exit_price - current_trade.entry_price) * (1 if current_trade.signal_type == "LONG" else -1)
                    current_trade.pnl = price_change * current_trade.position_size
                    current_trade.pnl_pct = (current_trade.pnl / current_trade.account_equity) * 100
                    account_equity += current_trade.pnl
                    result.trades.append(current_trade)
                    current_trade = None
                pbar.close()
                break
        
        if account_blown:
            continue
        
        # Update progress
        if i % 1000 == 0 or i == len(df) - 1:
            pbar.update(min(1000, len(df) - pbar.n))
            pbar.set_postfix({
                'Equity': f'${account_equity:,.0f}',
                'Trades': len(result.trades),
                'Status': 'BLOWN' if account_blown else ('TARGET' if profit_target_hit else 'ACTIVE')
            })
        
        if current_trade is None:
            # Use pre-computed probability
            try:
                signals_checked += 1
                proba_up = all_proba_up[i]
                
                # Determine signal
                if proba_up >= THRESHOLD_LONG:
                    signal_type = "LONG"
                    signals_above_threshold += 1
                elif proba_up <= THRESHOLD_SHORT:
                    signal_type = "SHORT"
                    signals_above_threshold += 1
                else:
                    signal_type = None
                    signals_below_threshold += 1
                
                if signal_type in ["LONG", "SHORT"]:
                    # FIX: Enter at NEXT bar's open (bar i+1), not current bar's close
                    # This avoids lookahead bias
                    if i + 1 >= len(df):
                        continue  # Can't enter if no next bar
                    
                    next_row = df.iloc[i + 1]
                    entry_price_raw = next_row.get('open', next_row.get('close', next_row.get('mid', 0)))
                    
                    if entry_price_raw <= 0 or np.isnan(entry_price_raw):
                        signals_rejected_price += 1
                        continue
                    
                    # Add spread and slippage costs
                    if signal_type == "LONG":
                        # Long: pay ask price (higher) + slippage
                        entry_price = entry_price_raw * (1 + SPREAD_COST + SLIPPAGE_LONG)
                    else:  # SHORT
                        # Short: sell at bid price (lower) - slippage
                        entry_price = entry_price_raw * (1 - SPREAD_COST - SLIPPAGE_SHORT)
                    
                    atr = row.get('atr_14', entry_price_raw * 0.002)  # Use raw price for ATR calc
                    if np.isnan(atr) or atr <= 0:
                        atr = entry_price_raw * 0.002
                    
                    stop_price = entry_price - (1.5 * atr) if signal_type == "LONG" else entry_price + (1.5 * atr)
                    target_price = entry_price + (1.0 * atr) if signal_type == "LONG" else entry_price - (1.0 * atr)
                    
                    # Fixed position size: 0.25 lots
                    # 1 lot = 100 oz for gold, so 0.25 lots = 25 oz
                    position_size = FIXED_POSITION_SIZE * 100  # Convert lots to oz
                    
                    # Margin = notional value / leverage
                    notional_value = position_size * entry_price
                    margin_required = notional_value / leverage
                    
                    # Check if we have enough margin (leave 5% buffer)
                    if margin_required > account_equity * 0.95:
                        signals_rejected_margin += 1
                        continue
                    
                    # Trade accepted! Entry happens at bar i+1
                    current_trade = FundedTrade(
                        entry_time=next_row.name,  # Bar i+1 timestamp
                        entry_price=entry_price,
                        stop_price=stop_price,
                        target_price=target_price,
                        position_size=position_size,
                        account_equity=account_equity,
                        signal_type=signal_type
                    )
                    trade_entry_idx = i + 1  # Entry at bar i+1
            except Exception as e:
                signals_rejected_exception += 1
                continue
        else:
            # Check for exit (only check if we've actually entered - i >= trade_entry_idx)
            if i < trade_entry_idx:
                continue  # Haven't entered yet (signal at i, entry at i+1)
            
            bars_held = i - trade_entry_idx
            
            # Stop hit (with slippage)
            if current_trade.signal_type == "LONG" and row['low'] <= current_trade.stop_price:
                current_trade.exit_time = row.name
                # Add slippage on exit
                current_trade.exit_price = current_trade.stop_price * (1 - SLIPPAGE_LONG)
                current_trade.exit_reason = "Stop"
                current_trade.bars_held = bars_held
            elif current_trade.signal_type == "SHORT" and row['high'] >= current_trade.stop_price:
                current_trade.exit_time = row.name
                # Add slippage on exit
                current_trade.exit_price = current_trade.stop_price * (1 + SLIPPAGE_SHORT)
                current_trade.exit_reason = "Stop"
                current_trade.bars_held = bars_held
            # Target hit (with slippage)
            elif current_trade.signal_type == "LONG" and row['high'] >= current_trade.target_price:
                current_trade.exit_time = row.name
                # Add slippage on exit (less favorable fill)
                current_trade.exit_price = current_trade.target_price * (1 - SLIPPAGE_LONG)
                current_trade.exit_reason = "Target"
                current_trade.bars_held = bars_held
            elif current_trade.signal_type == "SHORT" and row['low'] <= current_trade.target_price:
                current_trade.exit_time = row.name
                # Add slippage on exit (less favorable fill)
                current_trade.exit_price = current_trade.target_price * (1 + SLIPPAGE_SHORT)
                current_trade.exit_reason = "Target"
                current_trade.bars_held = bars_held
            # Time stop (with spread cost)
            elif bars_held >= MAX_BARS_IN_TRADE:
                current_trade.exit_time = row.name
                exit_price_raw = row['close']
                # Add spread and slippage on time exit
                if current_trade.signal_type == "LONG":
                    current_trade.exit_price = exit_price_raw * (1 - SPREAD_COST - SLIPPAGE_LONG)
                else:  # SHORT
                    current_trade.exit_price = exit_price_raw * (1 + SPREAD_COST + SLIPPAGE_SHORT)
                current_trade.exit_reason = "Time"
                current_trade.bars_held = bars_held
            
            # Exit occurred
            if current_trade.exit_time is not None:
                price_change = (current_trade.exit_price - current_trade.entry_price)
                if current_trade.signal_type == "SHORT":
                    price_change = -price_change
                
                current_trade.pnl = price_change * current_trade.position_size
                current_trade.pnl_pct = (current_trade.pnl / current_trade.account_equity) * 100
                account_equity += current_trade.pnl
                
                result.trades.append(current_trade)
                current_trade = None
    
    pbar.close()
    
    if equity_values:
        result.equity_curve = pd.Series(equity_values, index=equity_times)
    
    result.final_balance = account_equity
    result.profit_target_hit = profit_target_hit
    result.account_blown = account_blown
    
    if verbose:
        # Diagnostic info
        if signals_checked > 0:
            avg_proba = np.mean(proba_samples) if proba_samples else 0.0
            min_proba = np.min(proba_samples) if proba_samples else 0.0
            max_proba = np.max(proba_samples) if proba_samples else 0.0
            print(f"\n[Diagnostics]")
            print(f"  Signals checked: {signals_checked:,}")
            print(f"  Signals meeting threshold: {signals_above_threshold:,}")
            print(f"  Signals below threshold: {signals_below_threshold:,}")
            if signals_above_threshold > 0:
                print(f"\n  Why signals didn't become trades:")
                print(f"    Rejected - Invalid price: {signals_rejected_price:,}")
                print(f"    Rejected - Zero position size: {signals_rejected_position_size:,}")
                print(f"    Rejected - Margin insufficient: {signals_rejected_margin:,}")
                print(f"    Rejected - Exception: {signals_rejected_exception:,}")
                print(f"    Total rejected: {signals_rejected_price + signals_rejected_position_size + signals_rejected_margin + signals_rejected_exception:,}")
            if proba_samples:
                print(f"\n  Probability range: {min_proba:.3f} - {max_proba:.3f} (avg: {avg_proba:.3f})")
                print(f"  Thresholds: LONG >= {THRESHOLD_LONG:.2f}, SHORT <= {THRESHOLD_SHORT:.2f}")
        
        print(result.summary)
        if profit_target_hit:
            print(f"  ✅ Profit target (10%) reached!")
        if account_blown:
            print(f"  ❌ Account blown (max drawdown 4% breached)")
    
    return result


def main():
    print("=" * 80)
    print("MODEL 1: FUNDED ACCOUNT BACKTEST")
    print("=" * 80)
    print("\nAccount Settings:")
    print("  Initial Balance: $50,000")
    print("  Leverage: 1:30")
    print("  Risk per Trade: 1%")
    print("  Profit Target: 10% ($55,000)")
    print("  Max Drawdown: 4% ($48,000)")
    print("  Signal Thresholds: Long >= 0.60, Short <= 0.30")
    
    # Load model
    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return
    
    print(f"\n[1] Loading Model 1...")
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    feature_cols = artifact["features"]
    print(f"   ✓ Model loaded ({len(feature_cols)} features)")
    
    # Load data
    print(f"\n[2] Loading data...")
    df = pd.read_parquet(FEATURES_PATH)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    test_df = df[(df.index >= TEST_START) & (df.index <= TEST_END)].copy()
    
    print(f"   2024-2025: {len(test_df):,} rows (test period)")
    
    # Ensure we have y_tb_15 labels
    if 'y_tb_15' not in test_df.columns:
        print(f"\n[2.1] Generating y_tb_15 labels...")
        from src.features_complete import add_triple_barrier_labels
        test_df = add_triple_barrier_labels(test_df, h_max=15, tp_mult=1.0, sl_mult=1.0, horizons=[15])
        print(f"   ✓ Labels generated")
    
    # Backtest on test period (2024-2025)
    print("\n" + "=" * 80)
    print("[3] BACKTEST ON 2024-2025 DATA")
    print("=" * 80)
    
    test_result = run_funded_backtest(
        test_df,
        model,
        feature_cols,
        initial_balance=INITIAL_BALANCE,
        leverage=LEVERAGE,
        profit_target_pct=PROFIT_TARGET_PCT,
        max_drawdown_pct=MAX_DRAWDOWN_PCT,
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()

