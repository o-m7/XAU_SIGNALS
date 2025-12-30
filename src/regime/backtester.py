#!/usr/bin/env python3
"""
Backtesting Framework with Full Trade Management.

Tracks:
- Entry/Exit timing
- P&L per trade
- Drawdown tracking
- Session analysis
- Regime performance
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from .strategy import RegimeAwareStrategy


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    entry_price: float
    direction: int  # 1=long, -1=short
    position_size: float
    stop_loss: float
    take_profit: float
    regime: str
    ml_prob: float
    
    exit_time: pd.Timestamp = None
    exit_price: float = None
    exit_reason: str = None
    pnl: float = 0.0
    pnl_r: float = 0.0
    mae: float = 0.0  # Max Adverse Excursion
    mfe: float = 0.0  # Max Favorable Excursion


class Backtester:
    """
    Backtesting engine with realistic execution.
    """
    
    def __init__(
        self,
        strategy: RegimeAwareStrategy,
        spread_cost: float = 0.50,  # $0.50 spread per trade
        max_holding_bars: int = 60   # Max 60 minutes per trade
    ):
        self.strategy = strategy
        self.spread_cost = spread_cost
        self.max_holding_bars = max_holding_bars
        
        self.trades: List[Trade] = []
        self.active_trade: Optional[Trade] = None
        self.equity_curve = []
    
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtest on historical data.
        """
        print(f"\n{'='*70}")
        print("BACKTESTING")
        print(f"{'='*70}")
        print(f"\nData: {len(df):,} bars from {df.index[0]} to {df.index[-1]}")
        
        # Add regime detection
        df = self.strategy.regime_detector.detect_regime(df)
        
        # Track equity
        initial_balance = self.strategy.risk_manager.account_balance
        current_equity = initial_balance
        
        # Iterate through bars
        for i in range(len(df)):
            current_bar = df.iloc[i]
            timestamp = df.index[i]
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'regime': current_bar.get('regime', 'UNKNOWN')
            })
            
            # Check if new day (reset daily PnL)
            if i > 0:
                prev_date = df.index[i-1].date()
                curr_date = timestamp.date()
                new_day = (curr_date != prev_date)
                if new_day:
                    self.strategy.risk_manager.update_pnl(0, new_day=True)
            
            # Manage active trade
            if self.active_trade is not None:
                self._check_exit(current_bar, i, timestamp)
                
                # Update equity from active trade P&L
                if self.active_trade is not None:
                    unrealized_pnl = self._calculate_unrealized_pnl(
                        self.active_trade, current_bar['close']
                    )
                    current_equity = initial_balance + self.strategy.risk_manager.total_pnl + unrealized_pnl
                else:
                    # Trade closed
                    current_equity = initial_balance + self.strategy.risk_manager.total_pnl
            
            # Generate new signal if no active trade
            if self.active_trade is None:
                regime_info = {
                    'regime': current_bar.get('regime', 'RANGING'),
                    'regime_strength': current_bar.get('regime_strength', 0.5)
                }
                
                signal, prob, metadata = self.strategy.generate_signal(
                    current_bar, regime_info
                )
                
                if signal != 0:
                    # Enter trade
                    trade_params = self.strategy.calculate_trade_params(
                        signal, current_bar, regime_info['regime']
                    )
                    
                    self.active_trade = Trade(
                        entry_time=timestamp,
                        entry_price=current_bar['close'],
                        direction=signal,
                        position_size=trade_params['position_size'],
                        stop_loss=trade_params['stop'],
                        take_profit=trade_params['target'],
                        regime=regime_info['regime'],
                        ml_prob=prob
                    )
        
        # Close any remaining active trade
        if self.active_trade is not None:
            self._force_close(df.iloc[-1], df.index[-1])
        
        # Create results DataFrame
        results_df = self._create_results_df()
        
        return results_df
    
    def _check_exit(self, bar: pd.Series, bar_idx: int, timestamp: pd.Timestamp):
        """Check if active trade should exit."""
        if self.active_trade is None:
            return
        
        high = bar['high']
        low = bar['low']
        close = bar['close']
        
        # Update MAE/MFE
        if self.active_trade.direction == 1:  # LONG
            # MAE: how far below entry did it go?
            self.active_trade.mae = min(self.active_trade.mae, low - self.active_trade.entry_price)
            # MFE: how far above entry did it go?
            self.active_trade.mfe = max(self.active_trade.mfe, high - self.active_trade.entry_price)
            
            # Check stop loss
            if low <= self.active_trade.stop_loss:
                self._close_trade(bar, self.active_trade.stop_loss, "stop_loss", timestamp)
                return
            
            # Check take profit
            if high >= self.active_trade.take_profit:
                self._close_trade(bar, self.active_trade.take_profit, "take_profit", timestamp)
                return
        
        else:  # SHORT
            # MAE: how far above entry did it go?
            self.active_trade.mae = max(self.active_trade.mae, high - self.active_trade.entry_price)
            # MFE: how far below entry did it go?
            self.active_trade.mfe = min(self.active_trade.mfe, low - self.active_trade.entry_price)
            
            # Check stop loss
            if high >= self.active_trade.stop_loss:
                self._close_trade(bar, self.active_trade.stop_loss, "stop_loss", timestamp)
                return
            
            # Check take profit
            if low <= self.active_trade.take_profit:
                self._close_trade(bar, self.active_trade.take_profit, "take_profit", timestamp)
                return
        
        # Check max holding period
        bars_held = (timestamp - self.active_trade.entry_time).total_seconds() / 60  # minutes
        if bars_held >= self.max_holding_bars:
            self._close_trade(bar, close, "max_hold", timestamp)
    
    def _close_trade(self, bar: pd.Series, exit_price: float, reason: str, timestamp: pd.Timestamp):
        """Close active trade."""
        trade = self.active_trade
        trade.exit_time = timestamp
        trade.exit_price = exit_price
        trade.exit_reason = reason
        
        # Calculate P&L (including spread)
        if trade.direction == 1:  # LONG
            gross_pnl = (exit_price - trade.entry_price) * trade.position_size
        else:  # SHORT
            gross_pnl = (trade.entry_price - exit_price) * trade.position_size
        
        net_pnl = gross_pnl - (self.spread_cost * trade.position_size)
        trade.pnl = net_pnl
        
        # Calculate R (multiples of risk)
        risk = abs(trade.entry_price - trade.stop_loss) * trade.position_size
        trade.pnl_r = net_pnl / risk if risk > 0 else 0
        
        # Update risk manager
        self.strategy.risk_manager.update_pnl(net_pnl)
        
        # Store completed trade
        self.trades.append(trade)
        self.active_trade = None
    
    def _force_close(self, bar: pd.Series, timestamp: pd.Timestamp):
        """Force close active trade at end of backtest."""
        if self.active_trade is not None:
            self._close_trade(bar, bar['close'], "end_of_data", timestamp)
    
    def _calculate_unrealized_pnl(self, trade: Trade, current_price: float) -> float:
        """Calculate unrealized P&L for active trade."""
        if trade.direction == 1:
            return (current_price - trade.entry_price) * trade.position_size
        else:
            return (trade.entry_price - current_price) * trade.position_size
    
    def _create_results_df(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = [asdict(t) for t in self.trades]
        df = pd.DataFrame(trades_data)
        
        return df
    
    def print_stats(self):
        """Print backtest statistics."""
        if not self.trades:
            print("\nNo trades executed.")
            return
        
        df = self._create_results_df()
        
        # Overall stats
        total_trades = len(df)
        wins = (df['pnl'] > 0).sum()
        losses = (df['pnl'] < 0).sum()
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        total_pnl = df['pnl'].sum()
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
        avg_loss = df[df['pnl'] < 0]['pnl'].mean() if losses > 0 else 0
        
        profit_factor = abs(df[df['pnl'] > 0]['pnl'].sum() / df[df['pnl'] < 0]['pnl'].sum()) if losses > 0 else np.inf
        
        # R-multiples
        avg_r = df['pnl_r'].mean()
        
        # Drawdown
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['dd'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_dd = equity_df['dd'].min()
        
        # Print
        print(f"\n{'='*70}")
        print("BACKTEST RESULTS")
        print(f"{'='*70}")
        
        print(f"\n  Total Trades:    {total_trades}")
        print(f"  Wins:            {wins} ({win_rate*100:.1f}%)")
        print(f"  Losses:          {losses}")
        
        print(f"\n  Total P&L:       ${total_pnl:,.2f}")
        print(f"  Avg Win:         ${avg_win:,.2f}")
        print(f"  Avg Loss:        ${avg_loss:,.2f}")
        print(f"  Profit Factor:   {profit_factor:.2f}")
        
        print(f"\n  Avg R-multiple:  {avg_r:.2f}R")
        print(f"  Max Drawdown:    {max_dd*100:.2f}%")
        
        # Per-regime stats
        if 'regime' in df.columns:
            print(f"\n  REGIME BREAKDOWN:")
            for regime in df['regime'].unique():
                regime_df = df[df['regime'] == regime]
                regime_wr = (regime_df['pnl'] > 0).sum() / len(regime_df)
                regime_pnl_pos = regime_df[regime_df['pnl'] > 0]['pnl'].sum()
                regime_pnl_neg = regime_df[regime_df['pnl'] < 0]['pnl'].sum()
                regime_pf = abs(regime_pnl_pos / regime_pnl_neg) if regime_pnl_neg != 0 else np.inf
                
                print(f"    {regime:12s}: {len(regime_df):3d} trades, WR={regime_wr*100:.1f}%, PF={regime_pf:.2f}")

