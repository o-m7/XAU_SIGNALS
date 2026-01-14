#!/usr/bin/env python3
"""
Backtest Engine v2 - Proper Quant Finance Implementation

Fixes from v1:
- Entry timing: Skip entry bar, start checking bar+1
- ATR normalization: Handle both uppercase/lowercase
- Proper PnL calculation with tick value
- Dynamic slippage based on volatility
- Correct Sharpe/Drawdown calculations
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime


# =============================================================================
# CONSTANTS
# =============================================================================

# Gold contract specifications
TICK_SIZE = 0.01  # Minimum price movement
TICK_VALUE = 0.01  # Value per tick per oz
CONTRACT_SIZE = 100  # oz per lot

# Default costs (realistic prop firm levels - ~$2 total per trade)
DEFAULT_SPREAD_PIPS = 0.5  # 0.5 pips = $0.05 per oz
DEFAULT_SLIPPAGE_PIPS = 0.2  # 0.2 pip = $0.02 per oz
COMMISSION_PER_LOT = 2.0  # $2 per lot round trip


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Trade:
    """Represents a single trade with all details."""
    entry_time: datetime
    entry_price: float
    direction: str  # 'LONG' or 'SHORT'
    size_lots: float
    stop_price: float
    target_price: float
    
    exit_time: Optional[datetime] = None
    exit_price: float = 0.0
    exit_reason: str = ""  # 'TARGET', 'STOP', 'TIME', 'SIGNAL'
    bars_held: int = 0
    
    # Calculated after exit
    gross_pnl: float = 0.0
    costs: float = 0.0
    net_pnl: float = 0.0
    r_multiple: float = 0.0
    
    def calculate_pnl(self, spread_cost: float, slippage_cost: float):
        """Calculate PnL after trade is closed."""
        if self.exit_price == 0:
            return
        
        # Price change in direction of trade
        if self.direction == 'LONG':
            price_change = self.exit_price - self.entry_price
        else:
            price_change = self.entry_price - self.exit_price
        
        # Gross PnL (before costs)
        size_oz = self.size_lots * CONTRACT_SIZE
        self.gross_pnl = price_change * size_oz
        
        # Costs: spread + slippage (entry and exit) + commission
        self.costs = (spread_cost + slippage_cost * 2) * size_oz + COMMISSION_PER_LOT * self.size_lots
        
        # Net PnL
        self.net_pnl = self.gross_pnl - self.costs
        
        # R-multiple (risk-adjusted return)
        risk_per_oz = abs(self.entry_price - self.stop_price)
        if risk_per_oz > 0:
            self.r_multiple = price_change / risk_per_oz


@dataclass
class BacktestResult:
    """Complete backtest results with all metrics."""
    model_name: str
    initial_balance: float
    final_balance: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None
    
    # Calculated metrics
    total_return_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_r_multiple: float = 0.0
    
    def calculate_metrics(self):
        """Calculate all performance metrics."""
        if not self.trades:
            return
        
        # Basic stats
        self.total_return_pct = (self.final_balance - self.initial_balance) / self.initial_balance * 100
        
        # Win rate (exclude breakeven)
        wins = [t for t in self.trades if t.net_pnl > 0]
        losses = [t for t in self.trades if t.net_pnl < 0]
        total_decided = len(wins) + len(losses)
        self.win_rate = len(wins) / total_decided if total_decided > 0 else 0
        
        # Profit factor
        gross_profit = sum(t.net_pnl for t in wins)
        gross_loss = abs(sum(t.net_pnl for t in losses))
        self.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade
        self.avg_trade_pnl = sum(t.net_pnl for t in self.trades) / len(self.trades)
        self.avg_r_multiple = sum(t.r_multiple for t in self.trades) / len(self.trades)
        
        # Sharpe ratio (annualized)
        if self.equity_curve is not None and len(self.equity_curve) > 1:
            returns = self.equity_curve.pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                # Assuming minute bars, ~252 trading days * 1440 minutes
                annualization = np.sqrt(252 * 1440)
                self.sharpe_ratio = (returns.mean() / returns.std()) * annualization
        
        # Max drawdown
        if self.equity_curve is not None:
            rolling_max = self.equity_curve.cummax()
            drawdown = (self.equity_curve - rolling_max) / rolling_max
            self.max_drawdown_pct = abs(drawdown.min()) * 100


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_atr_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ATR column exists with lowercase name."""
    df = df.copy()
    
    # Check for uppercase version
    if 'ATR_14' in df.columns and 'atr_14' not in df.columns:
        df['atr_14'] = df['ATR_14']
    
    # Calculate if missing
    if 'atr_14' not in df.columns:
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
    
    # Floor ATR at 0.1% of price (never zero/negative)
    min_atr = df['close'] * 0.001
    df['atr_14'] = df['atr_14'].clip(lower=min_atr)
    
    return df


def calculate_dynamic_costs(atr: float, price: float) -> Tuple[float, float]:
    """
    Calculate spread and slippage based on volatility.
    Higher volatility = higher expected slippage.
    """
    # Base costs in price terms
    base_spread = DEFAULT_SPREAD_PIPS * 0.10  # $0.30 base
    base_slippage = DEFAULT_SLIPPAGE_PIPS * 0.10  # $0.10 base
    
    # Volatility multiplier (higher ATR = more slippage)
    vol_ratio = atr / (price * 0.002)  # Normalized to ~2% daily vol
    vol_multiplier = max(1.0, min(2.0, vol_ratio))  # Cap between 1x and 2x
    
    spread = base_spread * vol_multiplier
    slippage = base_slippage * vol_multiplier
    
    return spread, slippage


# =============================================================================
# MAIN BACKTEST ENGINE
# =============================================================================

def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    model_name: str = "Model",
    initial_balance: float = 50000.0,
    position_size_lots: float = 0.25,
    stop_atr_mult: float = 1.0,
    target_atr_mult: float = 1.0,
    max_bars_in_trade: int = 30,
    use_time_exit: bool = False,  # NEW: Use time-based exit only
    verbose: bool = True
) -> BacktestResult:
    """
    Run backtest with proper quant finance implementation.
    
    Args:
        df: DataFrame with OHLCV + atr_14 columns
        signals: Series with 1 (LONG), -1 (SHORT), 0 (no trade)
        model_name: Name for reporting
        initial_balance: Starting account balance
        position_size_lots: Fixed lot size per trade
        stop_atr_mult: Stop distance in ATR multiples
        target_atr_mult: Target distance in ATR multiples
        max_bars_in_trade: Maximum holding period
        verbose: Print progress
    
    Returns:
        BacktestResult with all metrics
    """
    # Normalize data
    df = normalize_atr_column(df)
    
    # Initialize
    result = BacktestResult(
        model_name=model_name,
        initial_balance=initial_balance,
        final_balance=initial_balance
    )
    
    account_equity = initial_balance
    current_trade: Optional[Trade] = None
    trade_entry_bar: int = -1
    
    equity_values = []
    equity_times = []
    
    if verbose:
        print(f"\n[Backtesting {model_name}]")
        print(f"  Period: {df.index[0]} to {df.index[-1]}")
        print(f"  Bars: {len(df):,}")
    
    # Main loop
    for i in range(len(df) - 1):
        row = df.iloc[i]
        
        # Record equity
        equity_values.append(account_equity)
        equity_times.append(row.name)
        
        # Get signal for this bar
        signal = signals.iloc[i] if i < len(signals) else 0
        
        # === TRADE MANAGEMENT (if in trade) ===
        if current_trade is not None:
            # IMPORTANT: Skip the entry bar for stop/target checks
            # We entered at bar i, so we can only exit starting at bar i+1
            if i <= trade_entry_bar:
                continue
            
            bars_held = i - trade_entry_bar
            
            # Determine exit
            exit_price = 0.0
            exit_reason = ""
            
            if use_time_exit:
                # TIME-BASED EXIT ONLY (no stops/targets)
                if bars_held >= max_bars_in_trade:
                    exit_price = row['close']
                    exit_reason = "TIME"
            else:
                # Check stop/target using current bar's high/low
                hit_stop = False
                hit_target = False
                
                if current_trade.direction == 'LONG':
                    if row['low'] <= current_trade.stop_price:
                        hit_stop = True
                    elif row['high'] >= current_trade.target_price:
                        hit_target = True
                else:  # SHORT
                    if row['high'] >= current_trade.stop_price:
                        hit_stop = True
                    elif row['low'] <= current_trade.target_price:
                        hit_target = True
                
                if hit_target:
                    exit_price = current_trade.target_price
                    exit_reason = "TARGET"
                elif hit_stop:
                    exit_price = current_trade.stop_price
                    exit_reason = "STOP"
                elif bars_held >= max_bars_in_trade:
                    exit_price = row['close']
                    exit_reason = "TIME"
            
            # Close trade if exit triggered
            if exit_price > 0:
                current_trade.exit_time = row.name
                current_trade.exit_price = exit_price
                current_trade.exit_reason = exit_reason
                current_trade.bars_held = bars_held
                
                # Calculate PnL with costs
                atr = row.get('atr_14', row['close'] * 0.002)
                spread, slippage = calculate_dynamic_costs(atr, row['close'])
                current_trade.calculate_pnl(spread, slippage)
                
                # Update account
                account_equity += current_trade.net_pnl
                result.trades.append(current_trade)
                current_trade = None
                trade_entry_bar = -1
        
        # === SIGNAL HANDLING (if no trade) ===
        if current_trade is None and signal != 0:
            # Entry at CURRENT bar's close (matches training labels)
            entry_price_raw = row['close']
            
            if entry_price_raw <= 0 or np.isnan(entry_price_raw):
                continue
            
            # Get ATR for stop/target calculation
            atr = row.get('atr_14', entry_price_raw * 0.002)
            if np.isnan(atr) or atr <= 0:
                atr = entry_price_raw * 0.002
            
            # Calculate costs
            spread, slippage = calculate_dynamic_costs(atr, entry_price_raw)
            
            # Adjust entry price for costs
            if signal == 1:  # LONG
                direction = 'LONG'
                entry_price = entry_price_raw + spread / 2 + slippage
                stop_price = entry_price - (stop_atr_mult * atr)
                target_price = entry_price + (target_atr_mult * atr)
            else:  # SHORT
                direction = 'SHORT'
                entry_price = entry_price_raw - spread / 2 - slippage
                stop_price = entry_price + (stop_atr_mult * atr)
                target_price = entry_price - (target_atr_mult * atr)
            
            # Create trade
            current_trade = Trade(
                entry_time=row.name,
                entry_price=entry_price,
                direction=direction,
                size_lots=position_size_lots,
                stop_price=stop_price,
                target_price=target_price
            )
            trade_entry_bar = i  # Entry on signal bar
    
    # Close any open trade at end
    if current_trade is not None:
        last_row = df.iloc[-1]
        current_trade.exit_time = last_row.name
        current_trade.exit_price = last_row['close']
        current_trade.exit_reason = "END"
        current_trade.bars_held = len(df) - 1 - trade_entry_bar
        
        atr = last_row.get('atr_14', last_row['close'] * 0.002)
        spread, slippage = calculate_dynamic_costs(atr, last_row['close'])
        current_trade.calculate_pnl(spread, slippage)
        
        account_equity += current_trade.net_pnl
        result.trades.append(current_trade)
    
    # Finalize result
    result.final_balance = account_equity
    if equity_values:
        result.equity_curve = pd.Series(equity_values, index=equity_times)
    result.calculate_metrics()
    
    if verbose:
        print(f"  Trades: {len(result.trades)}")
        print(f"  Win Rate: {result.win_rate:.1%}")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown_pct:.2f}%")
        print(f"  Total Return: {result.total_return_pct:+.2f}%")
    
    return result


def print_detailed_results(result: BacktestResult):
    """Print comprehensive backtest results."""
    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS: {result.model_name}")
    print("=" * 60)
    
    print(f"\nAccount Performance:")
    print(f"  Initial Balance: ${result.initial_balance:,.2f}")
    print(f"  Final Balance:   ${result.final_balance:,.2f}")
    print(f"  Total Return:    {result.total_return_pct:+.2f}%")
    print(f"  Max Drawdown:    {result.max_drawdown_pct:.2f}%")
    
    print(f"\nTrade Statistics:")
    print(f"  Total Trades:    {len(result.trades)}")
    
    longs = [t for t in result.trades if t.direction == 'LONG']
    shorts = [t for t in result.trades if t.direction == 'SHORT']
    print(f"  Long Trades:     {len(longs)}")
    print(f"  Short Trades:    {len(shorts)}")
    
    wins = [t for t in result.trades if t.net_pnl > 0]
    losses = [t for t in result.trades if t.net_pnl < 0]
    print(f"  Winning Trades:  {len(wins)}")
    print(f"  Losing Trades:   {len(losses)}")
    print(f"  Win Rate:        {result.win_rate:.1%}")
    
    print(f"\nRisk Metrics:")
    print(f"  Profit Factor:   {result.profit_factor:.2f}")
    print(f"  Sharpe Ratio:    {result.sharpe_ratio:.2f}")
    print(f"  Avg Trade PnL:   ${result.avg_trade_pnl:.2f}")
    print(f"  Avg R-Multiple:  {result.avg_r_multiple:+.2f}R")
    
    # Exit reason breakdown
    exit_reasons = {}
    for t in result.trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
    
    print(f"\nExit Reasons:")
    for reason, count in sorted(exit_reasons.items()):
        print(f"  {reason}: {count} ({count/len(result.trades)*100:.1f}%)")
    
    # Prop challenge check
    print(f"\nProp Challenge Criteria:")
    print(f"  Win Rate > 60%:    {'PASS' if result.win_rate > 0.60 else 'FAIL'}")
    print(f"  Max DD < 4%:       {'PASS' if result.max_drawdown_pct < 4.0 else 'FAIL'}")
    print(f"  Profit > 6%:       {'PASS' if result.total_return_pct > 6.0 else 'FAIL'}")


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Backtest Engine v2 - Test")
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    dates = pd.date_range('2024-01-01', periods=n, freq='1min')
    
    close = 2000 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        'open': close + np.random.randn(n) * 0.1,
        'high': close + np.abs(np.random.randn(n)) * 0.5,
        'low': close - np.abs(np.random.randn(n)) * 0.5,
        'close': close,
        'volume': np.random.randint(100, 1000, n)
    }, index=dates)
    
    # Random signals for testing
    signals = pd.Series(np.random.choice([-1, 0, 0, 0, 1], n), index=dates)
    
    # Run backtest
    result = run_backtest(df, signals, model_name="Test Model")
    print_detailed_results(result)

