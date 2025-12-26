"""
Realistic Backtest for Model #2 Regime Classifier

Tests actual trading performance using regime-aware strategies:
- Mean reversion strategy for MEAN_REVERTING regime
- Breakout strategy for BREAKOUT regime
- Avoid/reduce size in HIGH_VOL and LOW_LIQUIDITY regimes
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model2_regime.regime_labeling import REGIME_NAMES

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
MODELS_DIR = PROJECT_ROOT / "models" / "model2_regime"

# Regime IDs
REGIME_TRENDING = 0
REGIME_MEAN_REVERTING = 1
REGIME_BREAKOUT = 2
REGIME_HIGH_VOL = 3
REGIME_LOW_LIQUIDITY = 4


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    entry_price: float
    direction: str  # 'LONG' or 'SHORT'
    tp_price: float
    sl_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_dollars: float = 0.0
    pnl_r: float = 0.0  # In R multiples
    regime: Optional[str] = None


def load_regime_predictions():
    """Load regime predictions from OOS validation."""
    pred_path = MODELS_DIR / "regime_predictions_dec2025.parquet"
    
    if not pred_path.exists():
        logger.error(f"Predictions not found: {pred_path}")
        logger.info("Run src/model2_regime/validate_oos.py first")
        return None
    
    df = pd.read_parquet(pred_path)
    logger.info(f"Loaded {len(df):,} bars with regime predictions")
    
    return df


def mean_reversion_signal(row: pd.Series) -> Optional[str]:
    """
    Mean reversion strategy logic.
    
    Enter when price is stretched from VWAP in low-vol conditions.
    """
    if 'vwap_zscore' not in row.index or 'vol_percentile' not in row.index:
        return None
    
    # Require low-ish vol
    if row['vol_percentile'] > 0.6:
        return None
    
    # VWAP mean reversion
    if row['vwap_zscore'] > 1.5:  # Price far above VWAP
        return 'SHORT'
    elif row['vwap_zscore'] < -1.5:  # Price far below VWAP
        return 'LONG'
    
    return None


def breakout_signal(row: pd.Series, prev_rows: pd.DataFrame) -> Optional[str]:
    """
    Breakout strategy logic.
    
    Enter when volatility expands after consolidation.
    """
    if len(prev_rows) < 10:
        return None
    
    # Check if we were in consolidation (tight BB)
    if 'bb_squeeze' not in row.index or 'vol_expanding' not in row.index:
        return None
    
    prev_squeeze = prev_rows['bb_squeeze'].iloc[-10:].mean()
    
    # Was in consolidation, now expanding
    if prev_squeeze < 0.7 and row['vol_expanding'] == 1:
        # Direction based on VWAP
        if 'vwap_distance' in row.index:
            if row['vwap_distance'] > 0.001:
                return 'LONG'
            elif row['vwap_distance'] < -0.001:
                return 'SHORT'
    
    return None


def calculate_tp_sl(
    entry_price: float,
    direction: str,
    atr: float,
    risk_reward: float = 1.5
) -> tuple:
    """
    Calculate TP/SL based on ATR.
    
    Args:
        entry_price: Entry price
        direction: 'LONG' or 'SHORT'
        atr: ATR value
        risk_reward: TP/SL ratio
        
    Returns:
        (tp_price, sl_price)
    """
    # Use ATR for distance
    sl_distance = atr * 1.5
    tp_distance = sl_distance * risk_reward
    
    if direction == 'LONG':
        sl_price = entry_price - sl_distance
        tp_price = entry_price + tp_distance
    else:  # SHORT
        sl_price = entry_price + sl_distance
        tp_price = entry_price - tp_distance
    
    return round(tp_price, 2), round(sl_price, 2)


def simulate_trade_exit(
    trade: Trade,
    df: pd.DataFrame,
    max_bars: int = 60  # 5 hours on 5-min bars
) -> Trade:
    """
    Simulate trade exit using triple-barrier logic.
    
    Args:
        trade: Trade to simulate
        df: DataFrame with price data
        max_bars: Maximum bars to hold trade
        
    Returns:
        Trade with exit filled
    """
    entry_idx = df.index.get_loc(trade.entry_time)
    
    for i in range(1, min(max_bars + 1, len(df) - entry_idx)):
        current_idx = entry_idx + i
        current_bar = df.iloc[current_idx]
        
        high = current_bar['high']
        low = current_bar['low']
        close = current_bar['close']
        
        # Check TP/SL hit
        if trade.direction == 'LONG':
            if high >= trade.tp_price:
                trade.exit_time = df.index[current_idx]
                trade.exit_price = trade.tp_price
                trade.exit_reason = 'TP'
                trade.pnl_dollars = trade.tp_price - trade.entry_price
                trade.pnl_r = 1.5  # 1.5R
                return trade
            elif low <= trade.sl_price:
                trade.exit_time = df.index[current_idx]
                trade.exit_price = trade.sl_price
                trade.exit_reason = 'SL'
                trade.pnl_dollars = trade.sl_price - trade.entry_price
                trade.pnl_r = -1.0  # -1R
                return trade
        else:  # SHORT
            if low <= trade.tp_price:
                trade.exit_time = df.index[current_idx]
                trade.exit_price = trade.tp_price
                trade.exit_reason = 'TP'
                trade.pnl_dollars = trade.entry_price - trade.tp_price
                trade.pnl_r = 1.5  # 1.5R
                return trade
            elif high >= trade.sl_price:
                trade.exit_time = df.index[current_idx]
                trade.exit_price = trade.sl_price
                trade.exit_reason = 'SL'
                trade.pnl_dollars = trade.entry_price - trade.sl_price
                trade.pnl_r = -1.0  # -1R
                return trade
    
    # Max duration reached
    exit_idx = min(entry_idx + max_bars, len(df) - 1)
    trade.exit_time = df.index[exit_idx]
    trade.exit_price = df.iloc[exit_idx]['close']
    trade.exit_reason = 'MAX_TIME'
    
    if trade.direction == 'LONG':
        trade.pnl_dollars = trade.exit_price - trade.entry_price
    else:
        trade.pnl_dollars = trade.entry_price - trade.exit_price
    
    # Calculate R based on SL distance
    sl_distance = abs(trade.entry_price - trade.sl_price)
    trade.pnl_r = trade.pnl_dollars / sl_distance if sl_distance > 0 else 0
    
    return trade


def run_regime_backtest(
    df: pd.DataFrame,
    start_balance: float = 25000,
    risk_pct: float = 0.01
):
    """
    Run realistic backtest using regime-based strategies.
    
    Args:
        df: DataFrame with regime predictions and features
        start_balance: Starting account balance
        risk_pct: Risk per trade (% of balance)
    """
    logger.info("\n" + "="*80)
    logger.info("REGIME-BASED BACKTEST (December 2025)")
    logger.info("="*80)
    logger.info(f"Start balance: ${start_balance:,.0f}")
    logger.info(f"Risk per trade: {risk_pct*100:.2f}%")
    
    trades = []
    balance = start_balance
    peak_balance = start_balance
    max_dd = 0.0
    
    active_trade = None
    
    for i in range(20, len(df)):  # Start after lookback period
        current_time = df.index[i]
        current_bar = df.iloc[i]
        regime_id = current_bar['regime_pred']
        regime_name = REGIME_NAMES.get(regime_id, 'UNKNOWN')
        
        # Skip if we have an active trade
        if active_trade is not None:
            # Check if trade should be closed
            if active_trade.exit_time is None:
                # Simulate trade
                active_trade = simulate_trade_exit(active_trade, df.iloc[:i+1])
                
                # If trade closed, update balance
                if active_trade.exit_time is not None:
                    balance += active_trade.pnl_dollars
                    peak_balance = max(peak_balance, balance)
                    dd = (peak_balance - balance) / peak_balance
                    max_dd = max(max_dd, dd)
                    
                    trades.append(active_trade)
                    active_trade = None
            continue
        
        # Generate signal based on regime
        signal = None
        
        if regime_id == REGIME_MEAN_REVERTING:
            signal = mean_reversion_signal(current_bar)
        elif regime_id == REGIME_BREAKOUT:
            signal = breakout_signal(current_bar, df.iloc[max(0, i-20):i])
        elif regime_id == REGIME_HIGH_VOL:
            # Avoid trading in high vol
            continue
        elif regime_id == REGIME_LOW_LIQUIDITY:
            # Avoid trading in low liquidity
            continue
        
        # Enter trade if signal
        if signal in ['LONG', 'SHORT']:
            # Calculate TP/SL
            atr = current_bar.get('atr', current_bar['close'] * 0.005)  # Fallback
            tp_price, sl_price = calculate_tp_sl(
                current_bar['close'],
                signal,
                atr,
                risk_reward=1.5
            )
            
            # Create trade
            trade = Trade(
                entry_time=current_time,
                entry_price=current_bar['close'],
                direction=signal,
                tp_price=tp_price,
                sl_price=sl_price,
                regime=regime_name
            )
            
            active_trade = trade
    
    # Close any remaining active trade
    if active_trade is not None and active_trade.exit_time is None:
        active_trade.exit_time = df.index[-1]
        active_trade.exit_price = df.iloc[-1]['close']
        active_trade.exit_reason = 'END_OF_DATA'
        
        if active_trade.direction == 'LONG':
            active_trade.pnl_dollars = active_trade.exit_price - active_trade.entry_price
        else:
            active_trade.pnl_dollars = active_trade.entry_price - active_trade.exit_price
        
        sl_distance = abs(active_trade.entry_price - active_trade.sl_price)
        active_trade.pnl_r = active_trade.pnl_dollars / sl_distance if sl_distance > 0 else 0
        
        balance += active_trade.pnl_dollars
        trades.append(active_trade)
    
    # Analyze results
    if len(trades) == 0:
        logger.warning("No trades generated!")
        return
    
    trades_df = pd.DataFrame([{
        'entry_time': t.entry_time,
        'direction': t.direction,
        'entry_price': t.entry_price,
        'exit_time': t.exit_time,
        'exit_price': t.exit_price,
        'exit_reason': t.exit_reason,
        'pnl_dollars': t.pnl_dollars,
        'pnl_r': t.pnl_r,
        'regime': t.regime
    } for t in trades])
    
    # Overall metrics
    total_trades = len(trades)
    wins = len([t for t in trades if t.pnl_r > 0])
    losses = len([t for t in trades if t.pnl_r < 0])
    breakeven = total_trades - wins - losses
    win_rate = wins / total_trades if total_trades > 0 else 0
    
    total_pnl = sum(t.pnl_dollars for t in trades)
    avg_win = np.mean([t.pnl_dollars for t in trades if t.pnl_dollars > 0]) if wins > 0 else 0
    avg_loss = np.mean([t.pnl_dollars for t in trades if t.pnl_dollars < 0]) if losses > 0 else 0
    
    cum_r = sum(t.pnl_r for t in trades)
    avg_r = cum_r / total_trades if total_trades > 0 else 0
    
    final_balance = balance
    total_return_pct = (final_balance - start_balance) / start_balance * 100
    
    logger.info("\n" + "="*80)
    logger.info("BACKTEST RESULTS")
    logger.info("="*80)
    logger.info(f"\nTotal trades: {total_trades}")
    logger.info(f"Wins: {wins} | Losses: {losses} | Breakeven: {breakeven}")
    logger.info(f"Win rate: {win_rate*100:.1f}%")
    logger.info(f"\nP&L: ${total_pnl:,.2f}")
    logger.info(f"Avg win: ${avg_win:.2f} | Avg loss: ${avg_loss:.2f}")
    logger.info(f"Win/Loss ratio: {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0 else "N/A")
    logger.info(f"\nCumulative R: {cum_r:.2f}R")
    logger.info(f"Average R per trade: {avg_r:.2f}R")
    logger.info(f"\nFinal balance: ${final_balance:,.2f}")
    logger.info(f"Total return: {total_return_pct:+.1f}%")
    logger.info(f"Max drawdown: {max_dd*100:.1f}%")
    
    # Performance by regime
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE BY REGIME")
    logger.info("="*80)
    
    for regime_name in trades_df['regime'].unique():
        regime_trades = [t for t in trades if t.regime == regime_name]
        if len(regime_trades) == 0:
            continue
        
        regime_wins = len([t for t in regime_trades if t.pnl_r > 0])
        regime_total = len(regime_trades)
        regime_win_rate = regime_wins / regime_total if regime_total > 0 else 0
        regime_cum_r = sum(t.pnl_r for t in regime_trades)
        regime_pnl = sum(t.pnl_dollars for t in regime_trades)
        
        logger.info(f"\n{regime_name}:")
        logger.info(f"  Trades: {regime_total}")
        logger.info(f"  Win rate: {regime_win_rate*100:.1f}%")
        logger.info(f"  Cum R: {regime_cum_r:.2f}R")
        logger.info(f"  P&L: ${regime_pnl:,.2f}")
    
    # Performance by direction
    logger.info("\n" + "="*80)
    logger.info("PERFORMANCE BY DIRECTION")
    logger.info("="*80)
    
    for direction in ['LONG', 'SHORT']:
        dir_trades = [t for t in trades if t.direction == direction]
        if len(dir_trades) == 0:
            continue
        
        dir_wins = len([t for t in dir_trades if t.pnl_r > 0])
        dir_total = len(dir_trades)
        dir_win_rate = dir_wins / dir_total if dir_total > 0 else 0
        dir_cum_r = sum(t.pnl_r for t in dir_trades)
        dir_pnl = sum(t.pnl_dollars for t in dir_trades)
        
        logger.info(f"\n{direction}:")
        logger.info(f"  Trades: {dir_total}")
        logger.info(f"  Win rate: {dir_win_rate*100:.1f}%")
        logger.info(f"  Cum R: {dir_cum_r:.2f}R")
        logger.info(f"  P&L: ${dir_pnl:,.2f}")
    
    # Save trades
    output_path = MODELS_DIR / "backtest_trades_dec2025.csv"
    trades_df.to_csv(output_path, index=False)
    logger.info(f"\n\nSaved trades to: {output_path}")
    
    return trades_df


def main():
    logger.info("="*80)
    logger.info("MODEL #2 REGIME-BASED REALISTIC BACKTEST")
    logger.info("="*80)
    
    # Load regime predictions
    df = load_regime_predictions()
    if df is None:
        return
    
    # Run backtest
    trades_df = run_regime_backtest(df, start_balance=25000, risk_pct=0.01)
    
    logger.info("\n" + "="*80)
    logger.info("BACKTEST COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()

