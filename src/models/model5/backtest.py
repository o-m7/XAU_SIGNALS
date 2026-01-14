"""
backtest.py - Walk-Forward Backtesting
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass, field
from scipy import stats

from .config import Model5Config
from .signals import Model5SignalEngine, Signal


@dataclass
class Trade:
    entry_time: object
    entry_price: float
    stop_price: float
    target_price: float
    exit_time: object = None
    exit_price: float = 0
    pnl: float = 0
    pnl_r: float = 0
    bars_held: int = 0
    exit_reason: str = ""
    zscore_at_entry: float = 0


@dataclass
class BacktestResult:
    trades: List[Trade] = field(default_factory=list)
    
    @property
    def n_trades(self) -> int:
        return len(self.trades)
    
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
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)
    
    @property
    def avg_pnl(self) -> float:
        return np.mean([t.pnl for t in self.trades]) if self.trades else 0
    
    @property
    def sharpe(self) -> float:
        if not self.trades or len(self.trades) < 2:
            return 0
        pnls = [t.pnl for t in self.trades]
        return np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
    
    @property
    def max_drawdown(self) -> float:
        if not self.trades:
            return 0
        cum = np.cumsum([t.pnl for t in self.trades])
        peak = np.maximum.accumulate(cum)
        return np.max(peak - cum)
    
    @property
    def t_statistic(self) -> float:
        if not self.trades or len(self.trades) < 2:
            return 0
        pnls = [t.pnl for t in self.trades]
        return np.mean(pnls) / (np.std(pnls) / np.sqrt(len(pnls))) if np.std(pnls) > 0 else 0
    
    @property
    def p_value(self) -> float:
        if len(self.trades) < 2:
            return 1.0
        return 1 - stats.t.cdf(self.t_statistic, df=len(self.trades)-1)
    
    def summary(self) -> str:
        return f"""
============ BACKTEST RESULTS ============
Trades:          {self.n_trades}
Win Rate:        {self.win_rate:.1%}
Profit Factor:   {self.profit_factor:.2f}
Total PnL:       ${self.total_pnl:.2f}
Avg PnL:         ${self.avg_pnl:.2f}
Sharpe/Trade:    {self.sharpe:.3f}
Max Drawdown:    ${self.max_drawdown:.2f}
T-statistic:     {self.t_statistic:.2f}
P-value:         {self.p_value:.4f}
Significant:     {"YES" if self.p_value < 0.05 else "NO"}
==========================================
"""


def run_backtest(
    df: pd.DataFrame,
    config: Model5Config = None,
    verbose: bool = True
) -> BacktestResult:
    """
    Run backtest on prepared data.
    """
    config = config or Model5Config()
    engine = Model5SignalEngine(config)
    result = BacktestResult()
    
    current_trade = None
    trade_entry_idx = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        if current_trade is None:
            # Check for entry
            signal = engine.generate_signal(row)
            
            if signal.signal == Signal.LONG:
                current_trade = Trade(
                    entry_time=row.name,
                    entry_price=signal.entry_price,
                    stop_price=signal.stop_price,
                    target_price=signal.target_price,
                    zscore_at_entry=signal.zscore,
                )
                trade_entry_idx = i
        
        else:
            # Check for exit
            bars_held = i - trade_entry_idx
            
            # Stop hit
            if row['low'] <= current_trade.stop_price:
                current_trade.exit_time = row.name
                current_trade.exit_price = current_trade.stop_price
                current_trade.exit_reason = "Stop"
                current_trade.pnl = current_trade.exit_price - current_trade.entry_price
                current_trade.bars_held = bars_held
                
            # Target hit
            elif row['high'] >= current_trade.target_price:
                current_trade.exit_time = row.name
                current_trade.exit_price = current_trade.target_price
                current_trade.exit_reason = "Target"
                current_trade.pnl = current_trade.exit_price - current_trade.entry_price
                current_trade.bars_held = bars_held
            
            # Time stop
            elif bars_held >= config.max_bars_in_trade:
                current_trade.exit_time = row.name
                current_trade.exit_price = row['close']
                current_trade.exit_reason = "Time"
                current_trade.pnl = current_trade.exit_price - current_trade.entry_price
                current_trade.bars_held = bars_held
            
            # Exit occurred
            if current_trade.exit_time is not None:
                risk = current_trade.entry_price - current_trade.stop_price
                current_trade.pnl_r = current_trade.pnl / risk if risk > 0 else 0
                result.trades.append(current_trade)
                current_trade = None
    
    if verbose:
        print(result.summary())
    
    return result

