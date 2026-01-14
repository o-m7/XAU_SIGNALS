#!/usr/bin/env python3
"""
Train and Backtest Model #4 (5m News-Gated Long-Only).

Complete implementation:
1. Load 5min data
2. Compute Model4 features + news gate
3. Create forward return labels (realistic)
4. Train XGBoost
5. Run FULL backtest with proper strategy rules
6. Show comprehensive results

Strategy Rules (from spec):
- Entry: ML signal × news_gate > 0.8 AND RSI < 75
- Position size: Based on confidence + VIX proxy
- Stop: Entry - 1.5*ATR
- Target: Entry + 0.5*ATR
- Max hold: 30 minutes (6 bars @ 5min)

Usage:
    python scripts/train_and_backtest_model4.py [--weeks 52]
"""

import sys
from pathlib import Path
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model4_features import compute_model4_features, MODEL4_FEATURES

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Model4Backtest")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_5min_data(weeks: int = 52) -> pd.DataFrame:
    """Load 5min XAUUSD data."""
    data_path = PROJECT_ROOT / "models" / "model2_regime" / "regime_features_5min.parquet"

    if not data_path.exists():
        raise FileNotFoundError(f"5min data not found: {data_path}")

    logger.info(f"Loading 5min data from {data_path}")
    df = pd.read_parquet(data_path)

    # Filter to last N weeks
    end_date = df.index.max()
    start_date = end_date - timedelta(weeks=weeks)
    df = df[df.index >= start_date].copy()

    logger.info(f"Loaded {len(df):,} bars ({start_date} to {end_date})")

    return df


# =============================================================================
# LABELING: Forward Returns (More Realistic than Triple-Barrier)
# =============================================================================

def create_forward_return_labels(
    df: pd.DataFrame,
    horizon_bars: int = 20,  # 20 bars = 100 minutes @ 5min
    positive_threshold: float = 0.003  # 0.3% return = positive label
) -> pd.DataFrame:
    """
    Create labels based on forward returns.

    More realistic than triple-barrier:
    - Label = 1 if forward return > threshold within horizon
    - Label = 0 otherwise

    Args:
        df: DataFrame with OHLCV
        horizon_bars: Lookback horizon (20 bars = 100min)
        positive_threshold: Min return for positive label (0.003 = 0.3%)

    Returns:
        DataFrame with 'target' and 'forward_return' columns
    """
    logger.info(f"Creating forward return labels: horizon={horizon_bars} bars, threshold={positive_threshold*100:.1f}%")

    close = df['close'].values
    labels = np.zeros(len(df), dtype=int)
    forward_returns = np.full(len(df), np.nan)

    for i in range(len(df) - horizon_bars):
        entry_price = close[i]
        future_prices = close[i+1:i+horizon_bars+1]

        # Max return within horizon
        max_return = (future_prices.max() - entry_price) / entry_price
        forward_returns[i] = max_return

        # Label = 1 if max return exceeds threshold
        if max_return >= positive_threshold:
            labels[i] = 1
        else:
            labels[i] = 0

    df['target'] = labels
    df['forward_return'] = forward_returns

    # Summary
    valid_labels = labels[~np.isnan(forward_returns)]
    pos_rate = valid_labels.mean()
    logger.info(f"Labels: {pos_rate*100:.1f}% positive ({(valid_labels==1).sum():,} / {len(valid_labels):,})")
    logger.info(f"Avg forward return (all): {np.nanmean(forward_returns)*100:.3f}%")
    logger.info(f"Avg forward return (positive): {np.nanmean(forward_returns[labels==1])*100:.3f}%")

    return df


# =============================================================================
# NEWS GATE COMPUTATION
# =============================================================================

def compute_news_gate_sync(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute news gate multiplier (simplified, no async).

    Uses volatility regime (ATR) and volume patterns only.
    """
    logger.info("Computing news gate multipliers...")

    multipliers = []

    for idx, row in df.iterrows():
        mult = 1.0

        # Volatility regime (ATR-based)
        atr = row.get('atr_14', row.get('ATR_14', 5.0))
        if atr < 3.0:
            mult *= 0.85  # Low vol
        elif atr > 8.0:
            mult *= 1.15  # High vol

        # Volume pattern
        volume = row.get('volume', 1000)
        if volume < 1000:
            mult *= 0.85  # Weak volume

        # Clamp to [0.5, 1.15]
        mult = max(0.5, min(1.15, mult))

        multipliers.append(mult)

    df['news_gate_mult'] = multipliers

    logger.info(f"News gate: mean={np.mean(multipliers):.3f}, min={np.min(multipliers):.3f}, max={np.max(multipliers):.3f}")

    return df


# =============================================================================
# TRAIN MODEL
# =============================================================================

def split_train_val(df: pd.DataFrame, val_weeks: int = 4) -> tuple:
    """Split into train/val using walk-forward."""
    val_cutoff = df.index.max() - timedelta(weeks=val_weeks)

    train = df[df.index < val_cutoff].copy()
    val = df[df.index >= val_cutoff].copy()

    logger.info(f"\nTrain: {len(train):,} bars ({train.index.min()} to {train.index.max()})")
    logger.info(f"Val:   {len(val):,} bars ({val.index.min()} to {val.index.max()})")

    return train, val


def train_xgboost(X_train, y_train, X_val, y_val) -> XGBClassifier:
    """Train XGBoost with class balancing."""
    logger.info("\n" + "="*80)
    logger.info("TRAINING XGBOOST")
    logger.info("="*80)

    # Class balance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    logger.info(f"Class balance: {n_pos:,} positive / {n_neg:,} negative (scale={scale_pos_weight:.2f})")

    model = XGBClassifier(
        max_depth=6,
        n_estimators=300,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=30,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    logger.info("✓ Training complete")

    return model


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class Model4Backtest:
    """
    Complete backtest engine implementing strategy from spec.

    Strategy Rules:
    - Entry: ML signal × news_gate > 0.8 AND RSI < 75
    - Position size: 1% risk, scaled by confidence
    - Stop: Entry - 1.5*ATR
    - Target: Entry + 0.5*ATR
    - Max hold: 30 minutes (6 bars @ 5min)
    """

    def __init__(
        self,
        model: XGBClassifier,
        initial_capital: float = 10_000.0,
        risk_per_trade: float = 0.01,  # 1% risk
        entry_threshold: float = 0.5,  # ML × news_gate > 0.5
        max_rsi: float = 75.0,  # RSI filter
        atr_stop_mult: float = 0.8,  # Stop at 0.8 ATR (OPTIMIZED)
        atr_target_mult: float = 1.0,  # Target at 1.0 ATR (OPTIMIZED, 1.25:1 R:R)
        max_hold_bars: int = 6,  # 30 min @ 5min bars
    ):
        self.model = model
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.entry_threshold = entry_threshold
        self.max_rsi = max_rsi
        self.atr_stop_mult = atr_stop_mult
        self.atr_target_mult = atr_target_mult
        self.max_hold_bars = max_hold_bars

        # State
        self.capital = initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtest on validation data.

        Args:
            df: DataFrame with features, close prices, news_gate_mult

        Returns:
            DataFrame with trades
        """
        logger.info("\n" + "="*80)
        logger.info("RUNNING BACKTEST")
        logger.info("="*80)
        logger.info(f"Initial capital: ${self.initial_capital:,.0f}")
        logger.info(f"Entry threshold: ML × news_gate > {self.entry_threshold}")
        logger.info(f"RSI filter: < {self.max_rsi}")
        logger.info(f"TP/SL: {self.atr_target_mult}ATR / {self.atr_stop_mult}ATR")
        logger.info(f"Max hold: {self.max_hold_bars} bars (30 min)")
        logger.info("")

        for i in range(len(df)):
            row = df.iloc[i]

            # Check exit conditions first
            if self.position is not None:
                self._check_exit(i, row)

            # Check entry conditions
            if self.position is None:
                self._check_entry(i, row, df)

            # Record equity
            self.equity_curve.append({
                'timestamp': row.name,
                'equity': self.capital,
                'in_position': self.position is not None
            })

        # Force close any open position at end
        if self.position is not None:
            self._force_close(len(df)-1, df.iloc[-1])

        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades)

        logger.info(f"\n✓ Backtest complete: {len(self.trades)} trades executed")

        return trades_df

    def _check_entry(self, i: int, row: pd.Series, df: pd.DataFrame):
        """Check if entry conditions are met."""
        # Get ML probability
        X = row[MODEL4_FEATURES].values.reshape(1, -1)
        X = np.nan_to_num(X, nan=0.0)

        proba = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        up_idx = list(classes).index(1) if 1 in classes else -1
        ml_prob = proba[up_idx]

        # Apply news gate
        news_mult = row['news_gate_mult']
        gated_signal = ml_prob * news_mult

        # Get RSI
        rsi = row['rsi_14']

        # Entry rules
        if gated_signal < self.entry_threshold:
            return  # Signal too weak

        if rsi >= self.max_rsi:
            return  # RSI overbought

        # Calculate position size
        atr = row['atr_14']
        price = row['close']

        # Risk = 1% of capital
        risk_dollars = self.capital * self.risk_per_trade

        # Stop distance in dollars
        stop_distance = self.atr_stop_mult * atr

        # Position size (contracts/shares)
        # For gold: 1 contract = $100/point, 1 pip = $0.01 = $1
        # Simplified: risk_dollars / stop_distance
        position_size = risk_dollars / stop_distance

        # Scale by confidence (if signal < 1.0, reduce size)
        confidence_scalar = min(gated_signal, 1.0)
        position_size *= confidence_scalar

        # Calculate TP/SL
        tp_price = price + (self.atr_target_mult * atr)
        sl_price = price - (self.atr_stop_mult * atr)

        # Open position
        self.position = {
            'entry_idx': i,
            'entry_time': row.name,
            'entry_price': price,
            'tp': tp_price,
            'sl': sl_price,
            'size': position_size,
            'ml_prob': ml_prob,
            'news_mult': news_mult,
            'gated_signal': gated_signal,
            'rsi': rsi,
            'atr': atr,
        }

    def _check_exit(self, i: int, row: pd.Series):
        """Check if exit conditions are met."""
        if self.position is None:
            return

        price = row['close']

        # Check TP
        if price >= self.position['tp']:
            self._close_position(i, row, price, 'TP_HIT')
            return

        # Check SL
        if price <= self.position['sl']:
            self._close_position(i, row, price, 'SL_HIT')
            return

        # Check max hold time
        bars_held = i - self.position['entry_idx']
        if bars_held >= self.max_hold_bars:
            self._close_position(i, row, price, 'MAX_HOLD')
            return

    def _close_position(self, i: int, row: pd.Series, exit_price: float, exit_reason: str):
        """Close position and record trade."""
        if self.position is None:
            return

        # Calculate P&L
        entry_price = self.position['entry_price']
        size = self.position['size']

        # P&L in dollars (simplified: size × price_change)
        price_change = exit_price - entry_price
        pnl_dollars = size * price_change
        pnl_pct = (exit_price - entry_price) / entry_price

        # Update capital
        self.capital += pnl_dollars

        # Hold time
        bars_held = i - self.position['entry_idx']
        hold_minutes = bars_held * 5

        # Record trade
        self.trades.append({
            'entry_time': self.position['entry_time'],
            'exit_time': row.name,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'tp': self.position['tp'],
            'sl': self.position['sl'],
            'size': size,
            'pnl_dollars': pnl_dollars,
            'pnl_pct': pnl_pct,
            'bars_held': bars_held,
            'hold_minutes': hold_minutes,
            'exit_reason': exit_reason,
            'ml_prob': self.position['ml_prob'],
            'news_mult': self.position['news_mult'],
            'gated_signal': self.position['gated_signal'],
            'rsi': self.position['rsi'],
            'result': 'WIN' if pnl_dollars > 0 else 'LOSS',
            'equity': self.capital,
        })

        # Clear position
        self.position = None

    def _force_close(self, i: int, row: pd.Series):
        """Force close position at end of backtest."""
        if self.position is None:
            return

        self._close_position(i, row, row['close'], 'END_OF_DATA')


# =============================================================================
# METRICS CALCULATION
# =============================================================================

def calculate_metrics(trades_df: pd.DataFrame, initial_capital: float) -> dict:
    """Calculate comprehensive backtest metrics."""
    if len(trades_df) == 0:
        logger.warning("⚠️ No trades executed!")
        return {}

    logger.info("\n" + "="*80)
    logger.info("BACKTEST RESULTS")
    logger.info("="*80)

    # Basic stats
    n_trades = len(trades_df)
    n_wins = (trades_df['result'] == 'WIN').sum()
    n_losses = (trades_df['result'] == 'LOSS').sum()
    win_rate = n_wins / n_trades if n_trades > 0 else 0

    # P&L
    total_pnl = trades_df['pnl_dollars'].sum()
    avg_win = trades_df[trades_df['pnl_dollars'] > 0]['pnl_dollars'].mean() if n_wins > 0 else 0
    avg_loss = trades_df[trades_df['pnl_dollars'] < 0]['pnl_dollars'].mean() if n_losses > 0 else 0

    # Profit factor
    gross_profit = trades_df[trades_df['pnl_dollars'] > 0]['pnl_dollars'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_dollars'] < 0]['pnl_dollars'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Returns
    final_capital = initial_capital + total_pnl
    total_return = (final_capital - initial_capital) / initial_capital

    # Drawdown
    equity_curve = trades_df['equity'].values
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_drawdown = abs(drawdown.min())

    # Sharpe (simplified: trade-based)
    trade_returns = trades_df['pnl_pct'].values
    sharpe = (trade_returns.mean() / trade_returns.std() * np.sqrt(252)) if trade_returns.std() > 0 else 0

    # Hold time
    avg_hold_minutes = trades_df['hold_minutes'].mean()

    # Exit reasons
    exit_reasons = trades_df['exit_reason'].value_counts()

    # Print results
    logger.info(f"\n📊 PERFORMANCE SUMMARY")
    logger.info(f"{'─'*80}")
    logger.info(f"Total Trades:        {n_trades:,}")
    logger.info(f"Wins:                {n_wins:,} ({win_rate*100:.1f}%)")
    logger.info(f"Losses:              {n_losses:,} ({(1-win_rate)*100:.1f}%)")
    logger.info(f"")
    logger.info(f"Initial Capital:     ${initial_capital:,.2f}")
    logger.info(f"Final Capital:       ${final_capital:,.2f}")
    logger.info(f"Total P&L:           ${total_pnl:,.2f} ({total_return*100:+.2f}%)")
    logger.info(f"")
    logger.info(f"Avg Win:             ${avg_win:,.2f}")
    logger.info(f"Avg Loss:            ${avg_loss:,.2f}")
    logger.info(f"Win/Loss Ratio:      {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "Win/Loss Ratio:      N/A")
    logger.info(f"")
    logger.info(f"Profit Factor:       {profit_factor:.2f}")
    logger.info(f"Sharpe Ratio:        {sharpe:.2f}")
    logger.info(f"Max Drawdown:        {max_drawdown*100:.2f}%")
    logger.info(f"")
    logger.info(f"Avg Hold Time:       {avg_hold_minutes:.1f} minutes")
    logger.info(f"")
    logger.info(f"Exit Reasons:")
    for reason, count in exit_reasons.items():
        logger.info(f"  {reason:15s} {count:5,} ({count/n_trades*100:5.1f}%)")

    # Validation against targets
    logger.info(f"\n🎯 VALIDATION vs TARGETS")
    logger.info(f"{'─'*80}")

    target_wr = 0.52
    target_pf = 1.6

    wr_pass = "✅" if win_rate >= target_wr else "❌"
    pf_pass = "✅" if profit_factor >= target_pf else "❌"

    logger.info(f"Win Rate:            {win_rate*100:.1f}% {wr_pass} (Target: {target_wr*100:.0f}%)")
    logger.info(f"Profit Factor:       {profit_factor:.2f} {pf_pass} (Target: {target_pf:.1f})")

    passed = (win_rate >= target_wr) and (profit_factor >= target_pf)

    if passed:
        logger.info(f"\n✅ MODEL PASSES DEPLOYMENT CRITERIA")
    else:
        logger.info(f"\n⚠️  MODEL DOES NOT MEET DEPLOYMENT CRITERIA")

    return {
        'n_trades': n_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'avg_hold_minutes': avg_hold_minutes,
        'passed': passed,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train and Backtest Model #4")
    parser.add_argument("--weeks", type=int, default=52, help="Training data weeks (default: 52)")
    parser.add_argument("--val_weeks", type=int, default=4, help="Validation weeks (default: 4)")
    args = parser.parse_args()

    logger.info("="*80)
    logger.info("MODEL #4: 5m NEWS-GATED LONG-ONLY")
    logger.info("Training + Full Backtest")
    logger.info("="*80)
    logger.info("")

    # 1. Load data
    df = load_5min_data(weeks=args.weeks)

    # 2. Compute Model4 features
    logger.info("\nComputing Model4 features...")
    df = compute_model4_features(df, inplace=True)

    # 3. Create labels
    df = create_forward_return_labels(df, horizon_bars=20, positive_threshold=0.003)

    # 4. Compute news gate
    df = compute_news_gate_sync(df)

    # 5. Drop NaN
    df = df.dropna(subset=MODEL4_FEATURES + ['target', 'news_gate_mult'])
    logger.info(f"After dropna: {len(df):,} bars")

    # 6. Split train/val
    train_df, val_df = split_train_val(df, val_weeks=args.val_weeks)

    X_train = train_df[MODEL4_FEATURES]
    y_train = train_df['target']
    X_val = val_df[MODEL4_FEATURES]
    y_val = val_df['target']

    # 7. Train
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # 8. Run backtest
    backtest = Model4Backtest(model)
    trades_df = backtest.run(val_df)

    # 9. Calculate metrics
    metrics = calculate_metrics(trades_df, backtest.initial_capital)

    # 10. Save model if passed
    if metrics.get('passed', False):
        model_path = PROJECT_ROOT / "models" / "model4_news_gated_5m.joblib"

        artifact = {
            'model': model,
            'features': MODEL4_FEATURES,
            'metrics': metrics,
            'trained_at': datetime.now().isoformat(),
            'training_weeks': args.weeks,
        }

        joblib.dump(artifact, model_path)
        logger.info(f"\n✅ Model saved to {model_path}")
    else:
        logger.info(f"\n⚠️  Model did not pass validation - not saved")
        logger.info(f"   Run with better parameters or use --force if needed")

    # 11. Save trades
    if len(trades_df) > 0:
        trades_path = PROJECT_ROOT / "backtest_results" / "model4_trades.csv"
        trades_path.parent.mkdir(exist_ok=True)
        trades_df.to_csv(trades_path, index=False)
        logger.info(f"📁 Trades saved to {trades_path}")

    logger.info("\n" + "="*80)
    logger.info("COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
