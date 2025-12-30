#!/usr/bin/env python3
"""
Model 4 Backtesting Script

Simulates Model 4 signal generation with realistic execution assumptions:
- Spread: 1.5 bps
- Slippage: 0.5 bps
- Stop loss: 1.5 ATR
- Take profit: 2.0 ATR

Outputs:
- Win rate, Profit Factor, Sharpe per trade
- Max drawdown
- Signal distribution (LONG vs SHORT)
- Performance by session (London/NY/Overlap)
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model4 import (
    Model4Config,
    build_model4_features,
    get_model4_feature_columns,
    add_trend_aligned_labels,
    Model4SignalEngine,
    Signal,
)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    sl_price: float
    tp_price: float
    atr: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None  # "TP", "SL", "TIMEOUT"
    session: str = ""


class Model4Backtester:
    """
    Backtester for Model 4 strategy.
    """

    def __init__(
        self,
        config: Model4Config = None,
        spread_bps: float = 1.5,
        slippage_bps: float = 0.5,
        sl_atr_mult: float = 1.5,
        tp_atr_mult: float = 2.0,
        max_holding_bars: int = 48,  # 48 bars @ 5T = 4 hours max hold
        risk_per_trade: float = 0.01,
        initial_capital: float = 25_000.0,
    ):
        self.config = config or Model4Config()
        self.spread_bps = spread_bps
        self.slippage_bps = slippage_bps
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.max_holding_bars = max_holding_bars
        self.risk_per_trade = risk_per_trade
        self.initial_capital = initial_capital

        # State
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.current_trade: Optional[Trade] = None
        self.bars_in_trade: int = 0

    def _get_execution_cost_bps(self) -> float:
        """Get total execution cost in basis points."""
        return self.spread_bps + self.slippage_bps

    def _get_session(self, hour: int) -> str:
        """Classify session based on hour (UTC)."""
        if 13 <= hour < 16:
            return "Overlap"
        elif 7 <= hour < 16:
            return "London"
        elif 13 <= hour < 21:
            return "NY"
        else:
            return "Asia"

    def run_backtest(
        self,
        df: pd.DataFrame,
        model_path: str = None,
    ) -> Dict:
        """
        Run backtest on prepared data.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with features, labels, and OHLC data
        model_path : str
            Path to trained model (optional - if None, uses trend + threshold logic)

        Returns:
        --------
        Dict with backtest results
        """
        print("Running Model 4 backtest...")

        # Initialize signal engine if model provided
        signal_engine = None
        if model_path and Path(model_path).exists():
            signal_engine = Model4SignalEngine(model_path, self.config)
            print(f"Using trained model: {model_path}")
        else:
            print("No model provided - using trend filter + threshold logic")

        # Initialize state
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.current_trade = None
        self.bars_in_trade = 0
        last_signal_idx = -self.config.cooldown_bars - 1

        feature_cols = get_model4_feature_columns()

        # Iterate through bars
        for i in range(len(df)):
            row = df.iloc[i]
            timestamp = df.index[i]

            # Check if in trade
            if self.current_trade:
                self._update_trade(row, timestamp, i)
                if self.current_trade is None:  # Trade closed
                    self.equity_curve.append(
                        self.equity_curve[-1] + self.trades[-1].pnl
                    )
                continue

            # Skip if in cooldown
            if i - last_signal_idx < self.config.cooldown_bars:
                continue

            # Check tradeable conditions
            adx = row.get('adx', 0)
            if adx < self.config.min_adx:
                continue

            hour = row.get('hour', timestamp.hour)
            is_london = self.config.london_start <= hour < self.config.london_end
            is_ny = self.config.ny_start <= hour < self.config.ny_end
            if not (is_london or is_ny):
                continue

            spread_pct = row.get('spread_pct', 0.0001)
            if spread_pct > self.config.max_spread_pct:
                continue

            # Generate signal
            if signal_engine:
                # Use trained model
                try:
                    features = row[feature_cols]
                    signal_result = signal_engine.generate_signal(features, timestamp)
                    if signal_result.signal == Signal.NONE:
                        continue
                    signal = signal_result.signal.value
                except Exception as e:
                    continue
            else:
                # Use simple trend + threshold logic
                trend = row.get('trend', 0)
                if trend == 0:
                    continue
                signal = "LONG" if trend == 1 else "SHORT"

            # Open trade
            entry_price = row['close']
            atr = row.get('atr_14', row.get('atr', 1.0))

            # Apply execution cost
            exec_cost = entry_price * (self._get_execution_cost_bps() / 10000)
            if signal == "LONG":
                entry_price += exec_cost  # Buy at slightly higher price
                sl_price = entry_price - self.sl_atr_mult * atr
                tp_price = entry_price + self.tp_atr_mult * atr
            else:
                entry_price -= exec_cost  # Sell at slightly lower price
                sl_price = entry_price + self.sl_atr_mult * atr
                tp_price = entry_price - self.tp_atr_mult * atr

            self.current_trade = Trade(
                entry_time=timestamp,
                direction=signal,
                entry_price=entry_price,
                sl_price=sl_price,
                tp_price=tp_price,
                atr=atr,
                session=self._get_session(hour),
            )
            self.bars_in_trade = 0
            last_signal_idx = i

        # Close any remaining open trade
        if self.current_trade:
            self._close_trade(df.iloc[-1]['close'], df.index[-1], "TIMEOUT")
            self.equity_curve.append(
                self.equity_curve[-1] + self.trades[-1].pnl
            )

        # Compute results
        return self._compute_results(df)

    def _update_trade(self, row, timestamp, bar_idx):
        """Update current trade - check for SL/TP/timeout."""
        self.bars_in_trade += 1

        high = row['high']
        low = row['low']
        close = row['close']

        if self.current_trade.direction == "LONG":
            # Check stop loss first
            if low <= self.current_trade.sl_price:
                self._close_trade(self.current_trade.sl_price, timestamp, "SL")
                return
            # Check take profit
            if high >= self.current_trade.tp_price:
                self._close_trade(self.current_trade.tp_price, timestamp, "TP")
                return
        else:  # SHORT
            # Check stop loss first
            if high >= self.current_trade.sl_price:
                self._close_trade(self.current_trade.sl_price, timestamp, "SL")
                return
            # Check take profit
            if low <= self.current_trade.tp_price:
                self._close_trade(self.current_trade.tp_price, timestamp, "TP")
                return

        # Check timeout
        if self.bars_in_trade >= self.max_holding_bars:
            self._close_trade(close, timestamp, "TIMEOUT")

    def _close_trade(self, exit_price: float, exit_time, exit_reason: str):
        """Close current trade and record result."""
        trade = self.current_trade

        # Apply exit execution cost
        exec_cost = exit_price * (self._get_execution_cost_bps() / 10000)
        if trade.direction == "LONG":
            exit_price -= exec_cost  # Sell at slightly lower price
            pnl = exit_price - trade.entry_price
        else:
            exit_price += exec_cost  # Buy back at slightly higher price
            pnl = trade.entry_price - exit_price

        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.pnl = pnl
        trade.exit_reason = exit_reason

        self.trades.append(trade)
        self.current_trade = None
        self.bars_in_trade = 0

    def _compute_results(self, df: pd.DataFrame) -> Dict:
        """Compute backtest metrics."""
        if not self.trades:
            return {
                'n_trades': 0,
                'error': 'No trades generated'
            }

        # Basic metrics
        n_trades = len(self.trades)
        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate = len(wins) / n_trades if n_trades > 0 else 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.0001
        profit_factor = gross_profit / gross_loss

        # Sharpe per trade
        pnl_array = np.array(pnls)
        sharpe_per_trade = pnl_array.mean() / (pnl_array.std() + 1e-8)

        # Drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (running_max - equity_array) / running_max
        max_drawdown = drawdown.max()

        # Signal distribution
        long_trades = [t for t in self.trades if t.direction == "LONG"]
        short_trades = [t for t in self.trades if t.direction == "SHORT"]
        long_ratio = len(long_trades) / n_trades

        # Performance by session
        session_results = {}
        for session in ["London", "NY", "Overlap", "Asia"]:
            session_trades = [t for t in self.trades if t.session == session]
            if session_trades:
                session_pnls = [t.pnl for t in session_trades]
                session_wins = [p for p in session_pnls if p > 0]
                session_results[session] = {
                    'n_trades': len(session_trades),
                    'win_rate': len(session_wins) / len(session_trades),
                    'total_pnl': sum(session_pnls),
                    'avg_pnl': np.mean(session_pnls),
                }

        # Exit reason breakdown
        exit_reasons = {}
        for reason in ["TP", "SL", "TIMEOUT"]:
            reason_trades = [t for t in self.trades if t.exit_reason == reason]
            if reason_trades:
                exit_reasons[reason] = {
                    'count': len(reason_trades),
                    'pct': len(reason_trades) / n_trades,
                }

        # Trades per day
        if len(df) > 0:
            trading_days = (df.index[-1] - df.index[0]).days
            trades_per_day = n_trades / max(trading_days, 1)
        else:
            trades_per_day = 0

        # Final equity
        final_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital

        return {
            'n_trades': n_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_per_trade': sharpe_per_trade,
            'max_drawdown': max_drawdown,
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls),
            'total_return': total_return,
            'final_equity': final_equity,
            'trades_per_day': trades_per_day,
            'long_ratio': long_ratio,
            'short_ratio': 1 - long_ratio,
            'session_results': session_results,
            'exit_reasons': exit_reasons,
            'equity_curve': self.equity_curve,
        }


def print_results(results: Dict):
    """Pretty print backtest results."""
    print("\n" + "="*60)
    print("MODEL 4 BACKTEST RESULTS")
    print("="*60)

    if 'error' in results:
        print(f"Error: {results['error']}")
        return

    print(f"\n📊 Overall Performance:")
    print(f"  Total Trades: {results['n_trades']}")
    print(f"  Win Rate: {results['win_rate']*100:.1f}%")
    print(f"  Profit Factor: {results['profit_factor']:.2f}")
    print(f"  Sharpe/Trade: {results['sharpe_per_trade']:.3f}")
    print(f"  Max Drawdown: {results['max_drawdown']*100:.2f}%")

    print(f"\n💰 Returns:")
    print(f"  Total PnL: ${results['total_pnl']:.2f}")
    print(f"  Avg PnL/Trade: ${results['avg_pnl']:.2f}")
    print(f"  Total Return: {results['total_return']*100:.2f}%")
    print(f"  Final Equity: ${results['final_equity']:,.2f}")

    print(f"\n📈 Signal Distribution:")
    print(f"  LONG: {results['long_ratio']*100:.1f}%")
    print(f"  SHORT: {results['short_ratio']*100:.1f}%")
    print(f"  Trades/Day: {results['trades_per_day']:.1f}")

    print(f"\n🕐 Performance by Session:")
    for session, data in results.get('session_results', {}).items():
        print(f"  {session}: {data['n_trades']} trades, "
              f"WR={data['win_rate']*100:.1f}%, "
              f"Avg=${data['avg_pnl']:.2f}")

    print(f"\n🎯 Exit Reasons:")
    for reason, data in results.get('exit_reasons', {}).items():
        print(f"  {reason}: {data['count']} ({data['pct']*100:.1f}%)")

    # Validation checks
    print(f"\n✅ Validation Checks:")
    checks = {
        'Profit Factor > 1.2': results['profit_factor'] > 1.2,
        'Win Rate > 50%': results['win_rate'] > 0.50,
        'Max DD < 6%': results['max_drawdown'] < 0.06,
        'Trades/Day >= 5': results['trades_per_day'] >= 5,
        'Signal Balance (35-65%)': 0.35 < results['long_ratio'] < 0.65,
    }
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check}")


def main():
    parser = argparse.ArgumentParser(description="Backtest Model 4 strategy")

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/ohlcv_minute/XAUUSD_minute_2024.parquet",
        help="Path to 1-minute OHLCV data"
    )
    parser.add_argument(
        "--quotes_path",
        type=str,
        default=None,
        help="Path to quotes data (optional)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to trained Model 4 (optional)"
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="5T",
        help="Target timeframe (default: 5T)"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_path}...")
    df_1t = pd.read_parquet(args.data_path)

    # Ensure datetime index
    if not isinstance(df_1t.index, pd.DatetimeIndex):
        if 'timestamp' in df_1t.columns:
            df_1t['timestamp'] = pd.to_datetime(df_1t['timestamp'], utc=True)
            df_1t = df_1t.set_index('timestamp')

    # Load quotes if available
    df_quotes = None
    if args.quotes_path and Path(args.quotes_path).exists():
        print(f"Loading quotes from {args.quotes_path}...")
        df_quotes = pd.read_parquet(args.quotes_path)
        if not isinstance(df_quotes.index, pd.DatetimeIndex):
            if 'timestamp' in df_quotes.columns:
                df_quotes['timestamp'] = pd.to_datetime(df_quotes['timestamp'], utc=True)
                df_quotes = df_quotes.set_index('timestamp')

    print(f"Loaded {len(df_1t):,} 1-minute bars")

    # Build features
    print("Building Model 4 features...")
    config = Model4Config(base_timeframe=args.timeframe)
    df = build_model4_features(df_1t, df_quotes, args.timeframe)

    # Add labels for analysis
    df = add_trend_aligned_labels(
        df,
        horizon=config.horizon_bars,
        threshold_atr_mult=config.threshold_atr_mult
    )

    print(f"Built {len(df):,} bars with features")

    # Validate labels
    label_dist = df['y_good_entry'].value_counts(normalize=True)
    print(f"\nLabel Distribution:")
    print(f"  Good entry (1): {label_dist.get(1, 0)*100:.1f}%")
    print(f"  Bad entry (0): {label_dist.get(0, 0)*100:.1f}%")

    # Run backtest
    backtester = Model4Backtester(config)
    results = backtester.run_backtest(df, args.model_path)

    # Print results
    print_results(results)


if __name__ == "__main__":
    main()
