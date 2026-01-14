#!/usr/bin/env python3
"""
Unified Funded Account Backtest for All Models (1, 3, 5, 6)

2-Step Prop Challenge Parameters:
- Initial Balance: $50,000
- Max Drawdown: 4% (account breached if equity drops to $48,000)
- Profit Target: 6% (target reached at $53,000)
- Win Rate Target: >60%
- Test Period: 2024-01-01 to 2025-12-31
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from tqdm import tqdm
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# FEATURE BUILDING (must match training scripts)
# ============================================================================

def build_all_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build all features needed by all models including regime detection."""
    df = df.copy()
    
    # Ensure ATR exists and is positive
    # Check for both lowercase and uppercase versions
    if 'ATR_14' in df.columns and 'atr_14' not in df.columns:
        df['atr_14'] = df['ATR_14']  # Normalize to lowercase
    
    if 'atr_14' not in df.columns:
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
    
    # Floor ATR at 0.1% of price
    min_atr = df['close'] * 0.001
    df['atr_14'] = df['atr_14'].clip(lower=min_atr)
    
    # CMF (Model 3) - Multiple timeframes
    if all(c in df.columns for c in ['high', 'low', 'close', 'volume']):
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
        mfv = mfm * df['volume']
        if 'cmf_20' not in df.columns:
            df['cmf_20'] = mfv.rolling(window=20).sum() / (df['volume'].rolling(window=20).sum() + 1e-10)
        if 'cmf_10' not in df.columns:
            df['cmf_10'] = mfv.rolling(window=10).sum() / (df['volume'].rolling(window=10).sum() + 1e-10)
        if 'cmf' not in df.columns:
            df['cmf'] = df['cmf_20']
        if 'cmf_momentum' not in df.columns:
            df['cmf_momentum'] = df['cmf_20'] - df['cmf_20'].shift(5)
        if 'cmf_crossover' not in df.columns:
            df['cmf_crossover'] = (df['cmf_20'] > 0).astype(int) - (df['cmf_20'].shift(1) > 0).astype(int)
    
    # MACD (Model 3) - Full features
    if 'close' in df.columns:
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        if 'macd' not in df.columns:
            df['macd'] = ema12 - ema26
        if 'macd_signal' not in df.columns:
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        if 'macd_hist' not in df.columns:
            df['macd_hist'] = df['macd'] - df['macd_signal']
        if 'macd_diff' not in df.columns:
            df['macd_diff'] = df['macd_hist']
        if 'macd_crossover' not in df.columns:
            df['macd_crossover'] = (df['macd'] > df['macd_signal']).astype(int) - (df['macd'].shift(1) > df['macd_signal'].shift(1)).astype(int)
        if 'macd_hist_momentum' not in df.columns:
            df['macd_hist_momentum'] = df['macd_hist'] - df['macd_hist'].shift(3)
    
    # Efficiency Ratio (Model 5)
    if 'er_30' not in df.columns and 'close' in df.columns:
        change = abs(df['close'] - df['close'].shift(30))
        volatility = df['close'].diff().abs().rolling(30).sum()
        df['er_30'] = change / (volatility + 1e-10)
    
    # Bollinger Bands (Model 5) - Multiple periods
    if 'close' in df.columns:
        for period in [20, 50]:
            ma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            if f'bb_upper_{period}' not in df.columns:
                df[f'bb_upper_{period}'] = ma + 2 * std
            if f'bb_lower_{period}' not in df.columns:
                df[f'bb_lower_{period}'] = ma - 2 * std
            if f'bb_position_{period}' not in df.columns:
                df[f'bb_position_{period}'] = (df['close'] - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)
            if f'bb_width_{period}' not in df.columns:
                df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / (ma + 1e-10)
        # Legacy names
        if 'bb_upper' not in df.columns:
            df['bb_upper'] = df['bb_upper_20']
        if 'bb_lower' not in df.columns:
            df['bb_lower'] = df['bb_lower_20']
        if 'bb_position' not in df.columns:
            df['bb_position'] = df['bb_position_20']
    
    # RSI (Model 5)
    if 'close' in df.columns:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        if 'rsi_14' not in df.columns:
            df['rsi_14'] = 100 - (100 / (1 + rs))
        if 'rsi_oversold' not in df.columns:
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        if 'rsi_overbought' not in df.columns:
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
    
    # Price Z-score (Model 5) - Multiple periods
    if 'close' in df.columns:
        for period in [20, 50]:
            ma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            if f'zscore_{period}' not in df.columns:
                df[f'zscore_{period}'] = (df['close'] - ma) / (std + 1e-10)
        if 'price_zscore' not in df.columns:
            df['price_zscore'] = df['zscore_50']
    
    # Volume features (Model 6) - Full orderflow
    if 'volume' in df.columns:
        if 'volume_ma20' not in df.columns:
            df['volume_ma20'] = df['volume'].rolling(20).mean()
        if 'volume_ma5' not in df.columns:
            df['volume_ma5'] = df['volume'].rolling(5).mean()
        if 'volume_ma' not in df.columns:
            df['volume_ma'] = df['volume_ma20']
        if 'volume_ratio' not in df.columns:
            df['volume_ratio'] = df['volume'] / (df['volume_ma20'] + 1e-10)
        if 'volume_spike' not in df.columns:
            df['volume_spike'] = (df['volume'] > df['volume_ma20'] * 2).astype(int)
        if 'volume_trend' not in df.columns:
            df['volume_trend'] = df['volume_ma5'] / (df['volume_ma20'] + 1e-10)
        # Accumulation/Distribution
        if 'ad_line' not in df.columns:
            clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'] + 1e-10)
            df['ad_line'] = (clv * df['volume']).cumsum()
        if 'ad_momentum' not in df.columns:
            df['ad_momentum'] = df['ad_line'].diff(5)
    
    # Micro momentum (Model 6) - Multiple periods
    if 'close' in df.columns:
        for period in [3, 5, 10]:
            if f'momentum_{period}' not in df.columns:
                df[f'momentum_{period}'] = df['close'].pct_change(period) * 100
            if f'momentum_{period}_smooth' not in df.columns:
                df[f'momentum_{period}_smooth'] = df[f'momentum_{period}'].rolling(3).mean()
        if 'micro_momentum' not in df.columns:
            df['micro_momentum'] = df['momentum_5']
        if 'micro_momentum_smooth' not in df.columns:
            df['micro_momentum_smooth'] = df['momentum_5_smooth']
        # Velocity and acceleration
        if 'price_velocity' not in df.columns:
            df['price_velocity'] = df['close'].diff()
        if 'price_acceleration' not in df.columns:
            df['price_acceleration'] = df['price_velocity'].diff()
        # Micro trend
        if 'micro_trend' not in df.columns:
            df['micro_trend'] = df['close'].rolling(10).apply(
                lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=False
            )
    
    # Mid price dynamics (Model 6)
    if 'mid' not in df.columns and 'close' in df.columns:
        df['mid'] = df['close']
    if 'mid_velocity' not in df.columns:
        df['mid_velocity'] = df['mid'].diff()
    if 'mid_acceleration' not in df.columns:
        df['mid_acceleration'] = df['mid_velocity'].diff()
    
    # Range dynamics (Model 6)
    if all(c in df.columns for c in ['high', 'low', 'close', 'open']):
        if 'hl_range' not in df.columns:
            df['hl_range'] = df['high'] - df['low']
        if 'hl_range_ma' not in df.columns:
            df['hl_range_ma'] = df['hl_range'].rolling(20).mean()
        if 'hl_range_ratio' not in df.columns:
            df['hl_range_ratio'] = df['hl_range'] / (df['hl_range_ma'] + 1e-10)
        if 'upper_wick' not in df.columns:
            df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
        if 'lower_wick' not in df.columns:
            df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
        if 'wick_ratio' not in df.columns:
            df['wick_ratio'] = (df['upper_wick'] - df['lower_wick']) / (df['hl_range'] + 1e-10)
    
    # Regime detection features (ADX-based) - always calculate DI+ and DI-
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    atr_rolling = pd.Series(tr, index=df.index).rolling(14).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).mean() / (atr_rolling + 1e-10)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).mean() / (atr_rolling + 1e-10)
    
    if 'adx' not in df.columns:
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(14).mean()
    
    # Always add di_plus and di_minus
    if 'di_plus' not in df.columns:
        df['di_plus'] = plus_di
    if 'di_minus' not in df.columns:
        df['di_minus'] = minus_di
    
    # Regime boolean features (training script names)
    sma50 = close.rolling(50).mean()
    if 'regime_trending_up' not in df.columns:
        df['regime_trending_up'] = ((df['adx'] >= 25) & (close > sma50) & (df['di_plus'] > df['di_minus'])).astype(int)
    if 'regime_trending_down' not in df.columns:
        df['regime_trending_down'] = ((df['adx'] >= 25) & (close < sma50) & (df['di_minus'] > df['di_plus'])).astype(int)
    if 'regime_ranging' not in df.columns:
        df['regime_ranging'] = (df['adx'] < 20).astype(int)
    if 'regime_volatile' not in df.columns:
        # Volatility: ATR in top 20%
        atr_for_vol = df['atr_14']
        atr_pct_rank = atr_for_vol.rolling(100).apply(lambda x: 1 if len(x.dropna()) > 0 and x.iloc[-1] >= np.percentile(x.dropna(), 80) else 0, raw=False)
        df['regime_volatile'] = atr_pct_rank.fillna(0).astype(int)
    
    return df


# ============================================================================
# CONFIGURATION
# ============================================================================

FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "xauusd_features_2020_2025.parquet"

# Model paths - using BACKUP model that actually works (61% win rate verified)
MODELS_CONFIG = {
    "Model Backup (Verified 61%)": {
        "path": PROJECT_ROOT / "models" / "backups" / "y_tb_60_hgb_tuned_20251230_175110.joblib",
        "threshold_long": 0.60,   # Long if >60% up
        "threshold_short": 0.30,  # Short if <30% up
    },
}

# Test period
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

# Funded Account Parameters (2-Step Challenge)
INITIAL_BALANCE = 50000.0
LEVERAGE = 10.0
FIXED_POSITION_SIZE = 0.25  # 0.25 lots per trade
PROFIT_TARGET_PCT = 0.06    # 6% profit target
MAX_DRAWDOWN_PCT = 0.04     # 4% max drawdown
WIN_RATE_TARGET = 0.60      # 60% win rate target
MAX_BARS_IN_TRADE = 15      # 15 minutes max hold

# Transaction costs (realistic)
SPREAD_COST = 0.0003        # ~3 pips spread for gold
SLIPPAGE_LONG = 0.0002      # 2 pips slippage
SLIPPAGE_SHORT = 0.0002


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FundedTrade:
    entry_time: object
    entry_price: float
    stop_price: float
    target_price: float
    position_size: float
    account_equity: float
    signal_type: str
    r_multiple: float = 0.0
    exit_time: object = None
    exit_price: float = 0
    exit_reason: str = ""
    bars_held: int = 0
    pnl: float = 0
    pnl_pct: float = 0


@dataclass
class FundedBacktestResult:
    model_name: str
    initial_balance: float
    final_balance: float
    leverage: float
    trades: List[FundedTrade] = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None
    profit_target_hit: bool = False
    account_blown: bool = False
    
    @property
    def n_trades(self) -> int:
        return len(self.trades)
    
    @property
    def n_longs(self) -> int:
        return sum(1 for t in self.trades if t.signal_type == "LONG")
    
    @property
    def n_shorts(self) -> int:
        return sum(1 for t in self.trades if t.signal_type == "SHORT")
    
    @property
    def n_wins(self) -> int:
        return sum(1 for t in self.trades if t.pnl > 0)
    
    @property
    def n_losses(self) -> int:
        return sum(1 for t in self.trades if t.pnl <= 0)
    
    @property
    def win_rate(self) -> float:
        if self.n_trades == 0:
            return 0.0
        return self.n_wins / self.n_trades
    
    @property
    def profit_factor(self) -> float:
        if self.n_trades == 0:
            return 0.0
        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    @property
    def total_return_pct(self) -> float:
        return ((self.final_balance - self.initial_balance) / self.initial_balance) * 100
    
    @property
    def max_drawdown_pct(self) -> float:
        if self.equity_curve is None or len(self.equity_curve) == 0:
            return 0.0
        equity = self.equity_curve.values
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        return abs(np.min(drawdown)) * 100
    
    @property
    def cumulative_r(self) -> float:
        return sum(t.r_multiple for t in self.trades)
    
    @property
    def avg_r_per_trade(self) -> float:
        if self.n_trades == 0:
            return 0.0
        return self.cumulative_r / self.n_trades
    
    @property
    def passes_challenge(self) -> Dict[str, bool]:
        return {
            "max_drawdown": self.max_drawdown_pct < MAX_DRAWDOWN_PCT * 100,
            "profit_target": self.total_return_pct >= PROFIT_TARGET_PCT * 100,
            "win_rate": self.win_rate >= WIN_RATE_TARGET,
        }
    
    @property
    def all_criteria_passed(self) -> bool:
        return all(self.passes_challenge.values())


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def run_funded_backtest(
    df: pd.DataFrame,
    model,
    feature_cols: List[str],
    model_name: str,
    threshold_long: float,
    threshold_short: float,
    verbose: bool = True
) -> FundedBacktestResult:
    """Run funded account backtest for a model."""
    
    result = FundedBacktestResult(
        model_name=model_name,
        initial_balance=INITIAL_BALANCE,
        final_balance=INITIAL_BALANCE,
        leverage=LEVERAGE,
        trades=[]
    )
    
    current_trade = None
    trade_entry_idx = 0
    account_equity = INITIAL_BALANCE
    peak_equity = INITIAL_BALANCE
    
    profit_target = INITIAL_BALANCE * (1 + PROFIT_TARGET_PCT)
    max_drawdown_level = INITIAL_BALANCE * (1 - MAX_DRAWDOWN_PCT)
    
    equity_values = []
    equity_times = []
    
    account_blown = False
    profit_target_hit = False
    
    # Get available features
    available_features = [f for f in feature_cols if f in df.columns]
    missing_features = [f for f in feature_cols if f not in df.columns]
    
    if len(missing_features) > 0 and verbose:
        print(f"  Missing {len(missing_features)} features: {missing_features[:10]}...")
    
    if len(available_features) < len(feature_cols):
        # Model requires exact features - can't use partial
        if verbose:
            print(f"  Warning: Only {len(available_features)}/{len(feature_cols)} features available")
        if len(available_features) != len(feature_cols):
            # Need all features
            return result
    
    # Batch predict - use feature_cols order
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    
    all_proba = model.predict_proba(X)
    classes = model.classes_
    up_idx = list(classes).index(1) if 1 in classes else -1
    all_proba_up = all_proba[:, up_idx] if up_idx >= 0 else all_proba[:, -1]
    
    if verbose:
        pbar = tqdm(total=len(df), desc=f"Backtesting {model_name[:20]}", unit="bars", ncols=100, leave=True)
    
    for i in range(len(df) - 1):
        row = df.iloc[i]
        equity_values.append(account_equity)
        equity_times.append(row.name)
        
        if account_equity > peak_equity:
            peak_equity = account_equity
        
        # Check profit target
        if account_equity >= profit_target and not profit_target_hit:
            profit_target_hit = True
        
        # Check max drawdown
        if account_equity <= max_drawdown_level and not account_blown:
            account_blown = True
            if current_trade is not None:
                current_trade.exit_time = row.name
                current_trade.exit_price = row['close']
                current_trade.exit_reason = "Account Blown"
                price_change = (current_trade.exit_price - current_trade.entry_price)
                if current_trade.signal_type == "SHORT":
                    price_change = -price_change
                current_trade.pnl = price_change * current_trade.position_size
                current_trade.pnl_pct = (current_trade.pnl / current_trade.account_equity) * 100
                
                # Calculate R-multiple
                risk_per_unit = abs(current_trade.entry_price - current_trade.stop_price)
                if risk_per_unit > 0:
                    current_trade.r_multiple = price_change / risk_per_unit
                
                account_equity += current_trade.pnl
                result.trades.append(current_trade)
                current_trade = None
            if verbose:
                pbar.close()
            break
        
        if account_blown:
            continue
        
        if verbose and i % 5000 == 0:
            pbar.update(min(5000, len(df) - pbar.n))
            pbar.set_postfix({
                'Equity': f'${account_equity:,.0f}',
                'Trades': len(result.trades),
            })
        
        if current_trade is None:
            proba_up = all_proba_up[i]
            
            # Determine signal - CORRECT (model predicts UP correctly)
            if proba_up >= threshold_long:
                signal_type = "LONG"
            elif proba_up <= threshold_short:
                signal_type = "SHORT"
            else:
                signal_type = None
            
            if signal_type in ["LONG", "SHORT"]:
                if i + 1 >= len(df):
                    continue
                
                next_row = df.iloc[i + 1]
                entry_price_raw = next_row.get('open', next_row.get('close', next_row.get('mid', 0)))
                
                if entry_price_raw <= 0 or np.isnan(entry_price_raw):
                    continue
                
                # Add costs
                if signal_type == "LONG":
                    entry_price = entry_price_raw * (1 + SPREAD_COST + SLIPPAGE_LONG)
                else:
                    entry_price = entry_price_raw * (1 - SPREAD_COST - SLIPPAGE_SHORT)
                
                # FIX: Ensure positive ATR with proper floor
                atr = row.get('atr_14', entry_price_raw * 0.002)
                if np.isnan(atr) or atr <= 0:
                    atr = entry_price_raw * 0.002
                atr = max(atr, entry_price_raw * 0.0005)  # Minimum 0.05% ATR
                
                # 1:1 Risk/Reward ratio
                stop_price = entry_price - (1.0 * atr) if signal_type == "LONG" else entry_price + (1.0 * atr)
                target_price = entry_price + (1.0 * atr) if signal_type == "LONG" else entry_price - (1.0 * atr)
                
                position_size = FIXED_POSITION_SIZE * 100  # lots to oz
                notional_value = position_size * entry_price
                margin_required = notional_value / LEVERAGE
                
                if margin_required > account_equity * 0.95:
                    continue
                
                current_trade = FundedTrade(
                    entry_time=next_row.name,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    target_price=target_price,
                    position_size=position_size,
                    account_equity=account_equity,
                    signal_type=signal_type
                )
                trade_entry_idx = i + 1
        else:
            if i < trade_entry_idx:
                continue
            
            bars_held = i - trade_entry_idx
            
            # Check exits
            if current_trade.signal_type == "LONG" and row['low'] <= current_trade.stop_price:
                current_trade.exit_time = row.name
                current_trade.exit_price = current_trade.stop_price * (1 - SLIPPAGE_LONG)
                current_trade.exit_reason = "Stop"
                current_trade.bars_held = bars_held
            elif current_trade.signal_type == "SHORT" and row['high'] >= current_trade.stop_price:
                current_trade.exit_time = row.name
                current_trade.exit_price = current_trade.stop_price * (1 + SLIPPAGE_SHORT)
                current_trade.exit_reason = "Stop"
                current_trade.bars_held = bars_held
            elif current_trade.signal_type == "LONG" and row['high'] >= current_trade.target_price:
                current_trade.exit_time = row.name
                current_trade.exit_price = current_trade.target_price * (1 - SLIPPAGE_LONG)
                current_trade.exit_reason = "Target"
                current_trade.bars_held = bars_held
            elif current_trade.signal_type == "SHORT" and row['low'] <= current_trade.target_price:
                current_trade.exit_time = row.name
                current_trade.exit_price = current_trade.target_price * (1 + SLIPPAGE_SHORT)
                current_trade.exit_reason = "Target"
                current_trade.bars_held = bars_held
            elif bars_held >= MAX_BARS_IN_TRADE:
                current_trade.exit_time = row.name
                exit_price_raw = row['close']
                if current_trade.signal_type == "LONG":
                    current_trade.exit_price = exit_price_raw * (1 - SPREAD_COST - SLIPPAGE_LONG)
                else:
                    current_trade.exit_price = exit_price_raw * (1 + SPREAD_COST + SLIPPAGE_SHORT)
                current_trade.exit_reason = "Time"
                current_trade.bars_held = bars_held
            
            if current_trade.exit_time is not None:
                price_change = (current_trade.exit_price - current_trade.entry_price)
                if current_trade.signal_type == "SHORT":
                    price_change = -price_change
                
                current_trade.pnl = price_change * current_trade.position_size
                current_trade.pnl_pct = (current_trade.pnl / current_trade.account_equity) * 100
                
                # Calculate R-multiple
                risk_per_unit = abs(current_trade.entry_price - current_trade.stop_price)
                if risk_per_unit > 0:
                    current_trade.r_multiple = price_change / risk_per_unit
                
                account_equity += current_trade.pnl
                result.trades.append(current_trade)
                current_trade = None
    
    if verbose:
        pbar.close()
    
    if equity_values:
        result.equity_curve = pd.Series(equity_values, index=equity_times)
    
    result.final_balance = account_equity
    result.profit_target_hit = profit_target_hit
    result.account_blown = account_blown
    
    return result


# ============================================================================
# REPORTING
# ============================================================================

def print_model_result(result: FundedBacktestResult):
    """Print detailed results for a single model."""
    
    status_emoji = "PASSED" if result.all_criteria_passed else "FAILED"
    if result.account_blown:
        status_emoji = "BLOWN"
    elif result.profit_target_hit:
        status_emoji = "TARGET HIT"
    
    print(f"\n{result.model_name}:")
    print(f"  Trades: {result.n_trades:,} | Longs: {result.n_longs:,} | Shorts: {result.n_shorts:,}")
    print(f"  Wins: {result.n_wins:,} | Losses: {result.n_losses:,}")
    print(f"  Win Rate: {result.win_rate:.1%} | Profit Factor: {result.profit_factor:.2f}")
    print(f"  Initial: ${result.initial_balance:,.2f} | Final: ${result.final_balance:,.2f}")
    print(f"  Total Return: {result.total_return_pct:+.2f}%")
    print(f"  Max Drawdown: {result.max_drawdown_pct:.2f}%")
    print(f"  Cumulative R: {result.cumulative_r:+.2f}R | Avg R/Trade: {result.avg_r_per_trade:+.3f}R")
    
    # Criteria check
    criteria = result.passes_challenge
    print(f"\n  Challenge Criteria:")
    print(f"    Max DD < 4%:      {'PASS' if criteria['max_drawdown'] else 'FAIL'} ({result.max_drawdown_pct:.2f}%)")
    print(f"    Profit >= 6%:     {'PASS' if criteria['profit_target'] else 'FAIL'} ({result.total_return_pct:.2f}%)")
    print(f"    Win Rate > 60%:   {'PASS' if criteria['win_rate'] else 'FAIL'} ({result.win_rate:.1%})")
    print(f"  Status: {status_emoji}")


def print_summary(results: List[FundedBacktestResult]):
    """Print summary comparison of all models."""
    
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    # Header
    print(f"\n{'Model':<25} {'Trades':>8} {'Win%':>8} {'Return':>10} {'Max DD':>8} {'R-Total':>10} {'Status':>10}")
    print("-" * 80)
    
    for r in results:
        status = "PASS" if r.all_criteria_passed else "FAIL"
        if r.account_blown:
            status = "BLOWN"
        print(f"{r.model_name:<25} {r.n_trades:>8,} {r.win_rate:>7.1%} {r.total_return_pct:>+9.2f}% {r.max_drawdown_pct:>7.2f}% {r.cumulative_r:>+9.2f}R {status:>10}")
    
    # Best model
    print("\n" + "-" * 80)
    passing_models = [r for r in results if r.all_criteria_passed]
    if passing_models:
        best = max(passing_models, key=lambda x: x.total_return_pct)
        print(f"RECOMMENDED: {best.model_name} (Return: {best.total_return_pct:+.2f}%, Win Rate: {best.win_rate:.1%})")
    else:
        # If none pass all criteria, show the one with highest win rate
        best = max(results, key=lambda x: x.win_rate) if results else None
        if best:
            print(f"BEST PERFORMING (did not pass all criteria): {best.model_name}")
            print(f"  Win Rate: {best.win_rate:.1%}, Return: {best.total_return_pct:+.2f}%, Max DD: {best.max_drawdown_pct:.2f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("FUNDED ACCOUNT BACKTEST - ALL MODELS")
    print("=" * 80)
    print(f"\nTest Period: {TEST_START} to {TEST_END}")
    print(f"\n2-Step Prop Challenge Parameters:")
    print(f"  Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"  Profit Target:   {PROFIT_TARGET_PCT*100:.0f}% (${INITIAL_BALANCE * (1 + PROFIT_TARGET_PCT):,.2f})")
    print(f"  Max Drawdown:    {MAX_DRAWDOWN_PCT*100:.0f}% (${INITIAL_BALANCE * (1 - MAX_DRAWDOWN_PCT):,.2f})")
    print(f"  Win Rate Target: >{WIN_RATE_TARGET*100:.0f}%")
    print(f"  Leverage:        1:{int(LEVERAGE)}")
    print(f"  Position Size:   {FIXED_POSITION_SIZE} lots")
    
    # Load test data
    print(f"\n[1] Loading test data...")
    if not FEATURES_PATH.exists():
        print(f"ERROR: Features file not found: {FEATURES_PATH}")
        return
    
    df = pd.read_parquet(FEATURES_PATH)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    
    test_df = df[(df.index >= TEST_START) & (df.index <= TEST_END)].copy()
    print(f"  Loaded {len(test_df):,} rows for test period")
    print(f"  Date range: {test_df.index.min().date()} to {test_df.index.max().date()}")
    
    # Build all features needed by models
    print(f"\n[2] Building additional features for all models...")
    test_df = build_all_model_features(test_df)
    print(f"  Features after building: {len(test_df.columns)} columns")
    
    # Run backtests
    print(f"\n[3] Running backtests...")
    results = []
    
    for model_name, config in MODELS_CONFIG.items():
        model_path = config["path"]
        
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")
        
        if not model_path.exists():
            print(f"  SKIPPED: Model file not found: {model_path}")
            continue
        
        try:
            artifact = joblib.load(model_path)
            model = artifact["model"]
            feature_cols = artifact["features"]
            print(f"  Loaded model ({len(feature_cols)} features)")
            
            result = run_funded_backtest(
                test_df,
                model,
                feature_cols,
                model_name=model_name,
                threshold_long=config["threshold_long"],
                threshold_short=config["threshold_short"],
                verbose=True
            )
            
            print_model_result(result)
            results.append(result)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if results:
        print_summary(results)
    else:
        print("\nNo models were successfully tested!")
    
    print("\n" + "=" * 80)
    print("BACKTEST COMPLETE")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

