"""
Funded Account Backtest for Model 5

Tests Model 5 on a $50,000 account with:
- 1:30 leverage
- 1% risk per trade
- Dynamic ATR-based position sizing
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model5.config import Model5Config
from src.models.model5.features import build_all_features
from src.models.model5.signals import Model5SignalEngine, Signal
from src.models.model5.backtest import Trade


@dataclass
class FundedTrade:
    """Trade with account-based position sizing."""
    entry_time: object
    entry_price: float
    stop_price: float
    target_price: float
    exit_time: object = None
    exit_price: float = 0
    position_size: float = 0  # Units (oz)
    pnl: float = 0  # Dollar PnL
    pnl_pct: float = 0  # Percentage of account
    bars_held: int = 0
    exit_reason: str = ""
    account_equity: float = 0  # Account equity at entry
    zscore_at_entry: float = 0


@dataclass
class FundedBacktestResult:
    """Results with account equity tracking."""
    trades: List[FundedTrade] = None
    equity_curve: pd.Series = None
    initial_balance: float = 50000.0
    leverage: float = 30.0
    risk_per_trade: float = 0.01  # 1%
    
    def __post_init__(self):
        if self.trades is None:
            self.trades = []
    
    @property
    def n_trades(self) -> int:
        return len(self.trades)
    
    @property
    def final_balance(self) -> float:
        if not self.trades:
            return self.initial_balance
        return self.trades[-1].account_equity + self.trades[-1].pnl
    
    @property
    def total_return_pct(self) -> float:
        return (self.final_balance - self.initial_balance) / self.initial_balance * 100
    
    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0
        return sum(1 for t in self.trades if t.pnl > 0) / len(self.trades)
    
    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    @property
    def max_drawdown_pct(self) -> float:
        if self.equity_curve is None or len(self.equity_curve) == 0:
            return 0
        peak = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - peak) / peak * 100
        return abs(drawdown.min())
    
    @property
    def sharpe_ratio(self) -> float:
        if self.equity_curve is None or len(self.equity_curve) < 2:
            return 0
        returns = self.equity_curve.pct_change().dropna()
        if len(returns) == 0 or returns.std() == 0:
            return 0
        # Annualized Sharpe (assuming 252 trading days, 24 hours/day, 4 bars/hour = 24,192 bars/year)
        return (returns.mean() / returns.std()) * np.sqrt(24192)
    
    def summary(self) -> str:
        return f"""
============ FUNDED ACCOUNT BACKTEST ============
Initial Balance:     ${self.initial_balance:,.2f}
Final Balance:       ${self.final_balance:,.2f}
Total Return:        {self.total_return_pct:+.2f}%
Leverage:            {self.leverage}:1
Risk per Trade:      {self.risk_per_trade*100:.1f}%

Trades:              {self.n_trades:,}
Win Rate:            {self.win_rate:.1%}
Profit Factor:       {self.profit_factor:.2f}
Max Drawdown:        {self.max_drawdown_pct:.2f}%
Sharpe Ratio:        {self.sharpe_ratio:.3f}
==================================================
"""


def calculate_position_size(
    account_equity: float,
    entry_price: float,
    stop_price: float,
    risk_per_trade: float,
    leverage: float
) -> float:
    """
    Calculate position size in units (oz) based on risk.
    
    Args:
        account_equity: Current account equity
        entry_price: Entry price
        stop_price: Stop loss price
        risk_per_trade: Risk as fraction of account (e.g., 0.01 for 1%)
        leverage: Leverage ratio (e.g., 30 for 1:30)
    
    Returns:
        Position size in units (oz)
    """
    # Risk amount in dollars
    risk_amount = account_equity * risk_per_trade
    
    # Risk per unit (price difference)
    risk_per_unit = abs(entry_price - stop_price)
    
    if risk_per_unit == 0:
        return 0
    
    # Base position size (without leverage)
    base_size = risk_amount / risk_per_unit
    
    # Apply leverage
    position_size = base_size * leverage
    
    return position_size


def run_funded_backtest(
    df: pd.DataFrame,
    config: Model5Config = None,
    initial_balance: float = 50000.0,
    leverage: float = 30.0,
    risk_per_trade: float = 0.01,
    profit_target_pct: float = 0.10,  # 10%
    max_drawdown_pct: float = 0.04,    # 4%
    verbose: bool = True
) -> FundedBacktestResult:
    """
    Run backtest with funded account simulation.
    
    Args:
        df: DataFrame with features and OHLCV
        config: Model5Config
        initial_balance: Starting account balance
        leverage: Leverage ratio (1:30 = 30.0)
        risk_per_trade: Risk per trade as fraction (0.01 = 1%)
        verbose: Print results
    
    Returns:
        FundedBacktestResult
    """
    config = config or Model5Config()
    engine = Model5SignalEngine(config)
    result = FundedBacktestResult(
        initial_balance=initial_balance,
        leverage=leverage,
        risk_per_trade=risk_per_trade
    )
    
    current_trade = None
    trade_entry_idx = 0
    account_equity = initial_balance
    peak_equity = initial_balance
    
    # Calculate targets
    profit_target = initial_balance * (1 + profit_target_pct)
    max_drawdown_level = initial_balance * (1 - max_drawdown_pct)
    
    # Track equity curve
    equity_values = []
    equity_times = []
    
    # Account status
    account_blown = False
    profit_target_hit = False
    
    # Progress bar
    pbar = tqdm(total=len(df), desc="Backtesting", unit="bars", ncols=100, leave=True)
    
    for i in range(len(df)):
        row = df.iloc[i]
        equity_values.append(account_equity)
        equity_times.append(row.name)
        
        # Update progress every 100 bars
        if i % 100 == 0 or i == len(df) - 1:
            pbar.update(min(100, len(df) - pbar.n))
            pbar.set_postfix({
                'Equity': f'${account_equity:,.0f}',
                'Trades': len(result.trades),
                'Status': 'BLOWN' if account_blown else ('TARGET' if profit_target_hit else 'ACTIVE')
            })
        
        # Update peak equity
        if account_equity > peak_equity:
            peak_equity = account_equity
        
        # Check profit target
        if account_equity >= profit_target:
            if not profit_target_hit:
                profit_target_hit = True
                if verbose:
                    print(f"  ✅ Profit target hit at bar {i} ({row.name}): ${account_equity:,.2f}")
            # Continue trading but track that target was hit
        
        # Check max drawdown
        current_dd_pct = (account_equity - peak_equity) / peak_equity
        if account_equity <= max_drawdown_level:
            if not account_blown:
                account_blown = True
                if verbose:
                    print(f"  ❌ Max drawdown breached at bar {i} ({row.name}): ${account_equity:,.2f}")
                # Exit current trade if any
                if current_trade is not None:
                    current_trade.exit_time = row.name
                    current_trade.exit_price = row['close']
                    current_trade.exit_reason = "Account Blown"
                    price_change = current_trade.exit_price - current_trade.entry_price
                    current_trade.pnl = price_change * current_trade.position_size
                    current_trade.pnl_pct = (current_trade.pnl / current_trade.account_equity) * 100
                    account_equity += current_trade.pnl
                    result.trades.append(current_trade)
                    current_trade = None
                pbar.close()
                break  # Stop trading
        
        # Don't enter new trades if account is blown
        if account_blown:
            continue
        
        if current_trade is None:
            # Check for entry
            signal = engine.generate_signal(row)
            
            if signal.signal == Signal.LONG:
                # Calculate position size based on current equity
                position_size = calculate_position_size(
                    account_equity,
                    signal.entry_price,
                    signal.stop_price,
                    risk_per_trade,
                    leverage
                )
                
                # Check margin requirement (simplified: assume 3.33% margin for 1:30)
                margin_required = (position_size * signal.entry_price) / leverage
                
                # Only enter if we have enough margin
                if margin_required <= account_equity * 0.95:  # Leave 5% buffer
                    current_trade = FundedTrade(
                        entry_time=row.name,
                        entry_price=signal.entry_price,
                        stop_price=signal.stop_price,
                        target_price=signal.target_price,
                        position_size=position_size,
                        account_equity=account_equity,
                        zscore_at_entry=signal.zscore,
                    )
                    trade_entry_idx = i
                # else: skip trade due to insufficient margin
        
        else:
            # Check for exit
            bars_held = i - trade_entry_idx
            
            # Stop hit
            if row['low'] <= current_trade.stop_price:
                current_trade.exit_time = row.name
                current_trade.exit_price = current_trade.stop_price
                current_trade.exit_reason = "Stop"
                current_trade.bars_held = bars_held
                
            # Target hit
            elif row['high'] >= current_trade.target_price:
                current_trade.exit_time = row.name
                current_trade.exit_price = current_trade.target_price
                current_trade.exit_reason = "Target"
                current_trade.bars_held = bars_held
            
            # Time stop
            elif bars_held >= config.max_bars_in_trade:
                current_trade.exit_time = row.name
                current_trade.exit_price = row['close']
                current_trade.exit_reason = "Time"
                current_trade.bars_held = bars_held
            
            # Exit occurred
            if current_trade.exit_time is not None:
                # Calculate PnL
                price_change = current_trade.exit_price - current_trade.entry_price
                current_trade.pnl = price_change * current_trade.position_size
                current_trade.pnl_pct = (current_trade.pnl / current_trade.account_equity) * 100
                
                # Update account equity
                account_equity += current_trade.pnl
                
                result.trades.append(current_trade)
                current_trade = None
    
    # Create equity curve
    if equity_values:
        result.equity_curve = pd.Series(equity_values, index=equity_times)
    
    # Set status flags
    result.profit_target_hit = profit_target_hit
    result.account_blown = account_blown
    
    if verbose:
        print(result.summary())
        if profit_target_hit:
            print(f"  ✅ Profit target (10%) reached!")
        if account_blown:
            print(f"  ❌ Account blown (max drawdown 4% breached)")
    
    return result


def main():
    print("=" * 80)
    print("MODEL 5: FUNDED ACCOUNT BACKTEST")
    print("=" * 80)
    print("\nAccount Settings:")
    print("  Initial Balance: $50,000")
    print("  Leverage: 1:30")
    print("  Risk per Trade: 1%")
    print("  Profit Target: 10% ($55,000)")
    print("  Max Drawdown: 4% ($48,000)")
    print("  Position Sizing: Dynamic ATR-based")
    
    # Configuration
    config = Model5Config()
    
    # Data paths
    features_file = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"
    
    if not features_file.exists():
        print(f"❌ Features file not found: {features_file}")
        return
    
    # Load data
    print("\n[1] Loading data...")
    df = pd.read_parquet(features_file)
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    df = df.sort_index()
    
    # Split: Train (2020-2023) and Test (2024-2025)
    train_df = df[(df.index >= "2020-01-01") & (df.index <= "2023-12-31")].copy()
    test_df = df[(df.index >= "2024-01-01") & (df.index <= "2025-12-31")].copy()
    
    print(f"   Train: {len(train_df):,} rows")
    print(f"   Test:  {len(test_df):,} rows")
    
    # Work with 1-minute bars directly (no resampling needed)
    # Features will be computed on 1-minute bars, but validation is over 15 bars
    print("\n[2] Building features on 1-minute bars...")
    print("   This may take a few minutes...")
    
    # Build features on 1-minute data with progress
    print("   Building training features...")
    train_features = build_all_features(train_df, None, None, config, show_progress=True)
    train_features = train_features.dropna(subset=['zscore_20', 'atr_14', 'variance_ratio_2'])
    print(f"   ✓ Train features: {len(train_features):,} 1-minute bars after cleanup")
    
    print("\n   Building test features...")
    test_features = build_all_features(test_df, None, None, config, show_progress=True)
    test_features = test_features.dropna(subset=['zscore_20', 'atr_14', 'variance_ratio_2'])
    print(f"   ✓ Test features:  {len(test_features):,} 1-minute bars after cleanup")
    print(f"   Note: Trades validated over next 15 bars (15 minutes)")
    
    # Backtest on training period
    print("\n" + "=" * 80)
    print("[3] TRAINING PERIOD BACKTEST (2020-2023)")
    print("=" * 80)
    
    train_result = run_funded_backtest(
        train_features,
        config,
        initial_balance=50000.0,
        leverage=30.0,
        risk_per_trade=0.01,
        profit_target_pct=0.10,
        max_drawdown_pct=0.04,
        verbose=True
    )
    
    # Backtest on test period
    print("\n" + "=" * 80)
    print("[4] OUT-OF-SAMPLE BACKTEST (2024-2025)")
    print("=" * 80)
    
    test_result = run_funded_backtest(
        test_features,
        config,
        initial_balance=50000.0,
        leverage=30.0,
        risk_per_trade=0.01,
        profit_target_pct=0.10,
        max_drawdown_pct=0.04,
        verbose=True
    )
    
    # Comparison
    print("\n" + "=" * 80)
    print("COMPARISON: TRAIN vs TEST")
    print("=" * 80)
    
    print(f"\n{'Metric':<30} {'Train (2020-2023)':<25} {'Test (2024-2025)':<25}")
    print("-" * 80)
    print(f"{'Trades':<30} {train_result.n_trades:<25,} {test_result.n_trades:<25,}")
    print(f"{'Win Rate':<30} {train_result.win_rate:<24.1%} {test_result.win_rate:<24.1%}")
    print(f"{'Profit Factor':<30} {train_result.profit_factor:<25.2f} {test_result.profit_factor:<25.2f}")
    print(f"{'Final Balance':<30} ${train_result.final_balance:<24,.2f} ${test_result.final_balance:<24,.2f}")
    print(f"{'Total Return':<30} {train_result.total_return_pct:<24.2f}% {test_result.total_return_pct:<24.2f}%")
    print(f"{'Max Drawdown':<30} {train_result.max_drawdown_pct:<24.2f}% {test_result.max_drawdown_pct:<24.2f}%")
    print(f"{'Sharpe Ratio':<30} {train_result.sharpe_ratio:<25.3f} {test_result.sharpe_ratio:<25.3f}")
    print(f"{'Profit Target Hit':<30} {'YES' if train_result.profit_target_hit else 'NO':<25} {'YES' if test_result.profit_target_hit else 'NO':<25}")
    print(f"{'Account Blown':<30} {'YES' if train_result.account_blown else 'NO':<25} {'YES' if test_result.account_blown else 'NO':<25}")
    
    # Trade statistics
    if train_result.trades:
        train_avg_pnl = np.mean([t.pnl for t in train_result.trades])
        train_avg_pnl_pct = np.mean([t.pnl_pct for t in train_result.trades])
        print(f"{'Avg PnL per Trade':<30} ${train_avg_pnl:<24.2f} ({train_avg_pnl_pct:+.2f}%)")
    
    if test_result.trades:
        test_avg_pnl = np.mean([t.pnl for t in test_result.trades])
        test_avg_pnl_pct = np.mean([t.pnl_pct for t in test_result.trades])
        print(f"{'Avg PnL per Trade':<30} ${test_avg_pnl:<24.2f} ({test_avg_pnl_pct:+.2f}%)")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    
    return {
        'train': train_result,
        'test': test_result,
    }


if __name__ == "__main__":
    results = main()

