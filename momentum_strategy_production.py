#!/usr/bin/env python3
"""
Production-Ready Time-Series Momentum Trading System for XAUUSD

Strategy: Risk-adjusted momentum using multiple timeframes
Timeframe: Daily
Period: 2014-2025 (train: 2014-2023, validation: 2024-2025)

Author: Claude
Date: 2026-01-11
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Data
    'data_path': 'Raw Data/ohlcv_minute/XAUUSD_minute_all.parquet',
    'train_start': '2014-01-01',
    'train_end': '2023-12-31',
    'val_start': '2024-01-01',
    'val_end': '2025-01-31',

    # Momentum parameters
    'lookback_windows': [60, 126, 252],  # Trading days
    'momentum_weights': [0.3, 0.3, 0.4],  # Weights for each window
    'volatility_window': 20,  # Days for realized vol
    'regime_window': 252,  # Days for vol regime detection

    # Signal generation
    'signal_threshold': 0.0,  # Composite score threshold for long
    'vol_regime_threshold': 1.5,  # Max vol vs average

    # Risk management
    'stop_loss_pct': -0.02,  # -2% daily stop
    'take_profit_pct': 0.03,  # +3% daily take profit
    'max_exposure': 1.0,  # 100% max position

    # Transaction costs
    'transaction_cost': 0.0005,  # 0.05% per trade

    # Output
    'output_dir': Path('momentum_strategy_results'),
    'save_plots': True,
    'save_trades': True,
}

# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    """Load and prepare XAUUSD data for momentum strategy."""

    def __init__(self, config: Dict):
        self.config = config

    def load_and_resample(self) -> pd.DataFrame:
        """
        Load minute data and resample to daily.

        Returns:
            DataFrame with daily OHLCV indexed by date
        """
        print("="*80)
        print("LOADING DATA")
        print("="*80)

        # Load minute data
        print(f"Loading minute data from {self.config['data_path']}...")
        df = pd.read_parquet(self.config['data_path'])

        # Set timestamp as index
        df = df.set_index('timestamp')

        print(f"Loaded {len(df):,} minute bars")
        print(f"Date range: {df.index.min()} to {df.index.max()}")

        # Resample to daily
        print("\nResampling to daily frequency...")
        daily = df.resample('1D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'vwap': 'mean',
            'trades': 'sum'
        })

        # Drop days with no trading (weekends, holidays)
        daily = daily.dropna(subset=['close'])

        # Calculate returns
        daily['returns'] = daily['close'].pct_change()

        print(f"Resampled to {len(daily):,} trading days")
        print(f"Date range: {daily.index.min().date()} to {daily.index.max().date()}")
        print(f"\nSample data:")
        print(daily[['open', 'high', 'low', 'close', 'volume', 'returns']].head(10))

        return daily


# =============================================================================
# FEATURE ENGINEER
# =============================================================================

class FeatureEngineer:
    """Engineer momentum and risk features."""

    def __init__(self, config: Dict):
        self.config = config

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features needed for momentum strategy.

        Args:
            df: DataFrame with OHLCV and returns

        Returns:
            DataFrame with additional feature columns
        """
        print("\n" + "="*80)
        print("ENGINEERING FEATURES")
        print("="*80)

        df = df.copy()

        # 1. Rolling returns at multiple windows
        print("\n1. Calculating rolling returns...")
        for window in self.config['lookback_windows']:
            col_name = f'return_{window}d'
            df[col_name] = df['close'].pct_change(window)
            print(f"   {col_name}: {window}-day return")

        # 2. Realized volatility
        print("\n2. Calculating realized volatility...")
        vol_window = self.config['volatility_window']
        df['realized_vol'] = df['returns'].rolling(vol_window).std() * np.sqrt(252)
        print(f"   realized_vol: {vol_window}-day rolling std (annualized)")

        # 3. Risk-adjusted momentum (return / volatility)
        print("\n3. Calculating risk-adjusted momentum...")
        for window in self.config['lookback_windows']:
            return_col = f'return_{window}d'
            ra_col = f'risk_adj_momentum_{window}d'
            # Add small constant to avoid division by zero
            df[ra_col] = df[return_col] / (df['realized_vol'] + 1e-6)
            print(f"   {ra_col}: {return_col} / realized_vol")

        # 4. Volatility regime indicator
        print("\n4. Calculating volatility regime...")
        regime_window = self.config['regime_window']
        df['vol_avg_252d'] = df['realized_vol'].rolling(regime_window).mean()
        df['vol_regime'] = df['realized_vol'] / (df['vol_avg_252d'] + 1e-6)
        print(f"   vol_regime: current_vol / {regime_window}-day average")

        # 5. Drawdown from rolling high
        print("\n5. Calculating drawdown...")
        df['rolling_high_252d'] = df['close'].rolling(252).max()
        df['drawdown'] = (df['close'] - df['rolling_high_252d']) / df['rolling_high_252d']
        print(f"   drawdown: distance from 252-day high")

        # Summary
        print(f"\n✓ Feature engineering complete")
        print(f"  Total features: {len([c for c in df.columns if c not in ['open','high','low','close','volume','vwap','trades','returns']])}")

        return df


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

class SignalGenerator:
    """Generate trading signals based on momentum."""

    def __init__(self, config: Dict):
        self.config = config

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals.

        Args:
            df: DataFrame with features

        Returns:
            DataFrame with 'signal' and 'composite_score' columns
        """
        print("\n" + "="*80)
        print("GENERATING SIGNALS")
        print("="*80)

        df = df.copy()

        # Composite score = weighted average of risk-adjusted returns
        print("\n1. Calculating composite momentum score...")
        weights = self.config['momentum_weights']
        windows = self.config['lookback_windows']

        df['composite_score'] = 0.0
        for weight, window in zip(weights, windows):
            col = f'risk_adj_momentum_{window}d'
            df['composite_score'] += weight * df[col]
            print(f"   + {weight} × {col}")

        # Generate signals
        print("\n2. Generating signals...")
        threshold = self.config['signal_threshold']
        vol_threshold = self.config['vol_regime_threshold']

        # Long when:
        # - Composite score > threshold
        # - Volatility regime < threshold (not too volatile)
        df['signal'] = 0
        df.loc[
            (df['composite_score'] > threshold) &
            (df['vol_regime'] < vol_threshold),
            'signal'
        ] = 1

        # Position sizing: inverse volatility weighting
        print("\n3. Calculating position sizes...")
        max_exposure = self.config['max_exposure']

        # Target volatility = 20% annualized (adjust position to normalize vol)
        target_vol = 0.20
        df['position_size'] = (target_vol / (df['realized_vol'] + 1e-6)) * df['signal']

        # Cap at max exposure
        df['position_size'] = df['position_size'].clip(upper=max_exposure)

        # Summary
        n_long = (df['signal'] == 1).sum()
        n_flat = (df['signal'] == 0).sum()

        print(f"\n✓ Signal generation complete")
        print(f"  Long signals: {n_long:,} ({n_long/len(df)*100:.1f}%)")
        print(f"  Flat signals: {n_flat:,} ({n_flat/len(df)*100:.1f}%)")
        print(f"  Avg position size (when long): {df[df['signal']==1]['position_size'].mean():.2%}")

        return df


# =============================================================================
# BACKTESTER
# =============================================================================

class Backtester:
    """Vectorized backtest engine with proper execution logic."""

    def __init__(self, config: Dict):
        self.config = config

    def run_backtest(self, df: pd.DataFrame, period_name: str = "Full") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run vectorized backtest.

        Args:
            df: DataFrame with signals and position sizes
            period_name: Name for this backtest period

        Returns:
            (results_df, trades_df)
        """
        print("\n" + "="*80)
        print(f"RUNNING BACKTEST - {period_name.upper()} PERIOD")
        print("="*80)

        df = df.copy()

        # Execution: Calculate positions at close, execute next day at open
        df['position'] = df['position_size'].shift(1).fillna(0)

        # Strategy returns (before costs)
        # Return = position × next_day_return
        df['next_open'] = df['open'].shift(-1)
        df['next_close'] = df['close'].shift(-1)
        df['next_return'] = (df['next_close'] - df['next_open']) / df['next_open']

        df['strategy_return'] = df['position'] * df['next_return']

        # Apply stop loss and take profit
        df['strategy_return'] = df['strategy_return'].clip(
            lower=self.config['stop_loss_pct'],
            upper=self.config['take_profit_pct']
        )

        # Transaction costs
        df['position_change'] = df['position'].diff().abs()
        df['costs'] = df['position_change'] * self.config['transaction_cost']

        # Net returns
        df['net_return'] = df['strategy_return'] - df['costs']

        # Equity curve (handle NaN properly)
        df['equity'] = (1 + df['net_return'].fillna(0)).cumprod()
        df['buy_hold_equity'] = (1 + df['returns'].fillna(0)).cumprod()

        # Drawdown
        df['running_max'] = df['equity'].cummax()
        df['drawdown_pct'] = (df['equity'] - df['running_max']) / df['running_max']

        # Extract trades
        trades_df = self._extract_trades(df)

        print(f"\n✓ Backtest complete for {period_name} period")
        print(f"  Total days: {len(df):,}")
        print(f"  Trading days: {(df['position'] > 0).sum():,}")
        print(f"  Number of trades: {len(trades_df):,}")

        return df, trades_df

    def _extract_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract individual trades from backtest results."""
        trades = []

        in_trade = False
        entry_date = None
        entry_price = None

        for date, row in df.iterrows():
            if not in_trade and row['position'] > 0:
                # Enter trade
                in_trade = True
                entry_date = date
                entry_price = row['next_open']

            elif in_trade and row['position'] == 0:
                # Exit trade
                exit_date = date
                exit_price = row['next_open']

                if entry_price and exit_price:
                    pnl = (exit_price - entry_price) / entry_price
                    holding_days = (exit_date - entry_date).days

                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'return': pnl,
                        'holding_days': holding_days,
                    })

                in_trade = False
                entry_date = None
                entry_price = None

        return pd.DataFrame(trades)


# =============================================================================
# PERFORMANCE ANALYZER
# =============================================================================

class PerformanceAnalyzer:
    """Calculate and display performance metrics."""

    @staticmethod
    def calculate_metrics(df: pd.DataFrame, trades_df: pd.DataFrame, period_name: str) -> Dict:
        """Calculate comprehensive performance metrics."""

        # Filter to valid returns
        returns = df['net_return'].dropna()

        if len(returns) == 0:
            return {}

        # Basic metrics
        total_return = df['equity'].iloc[-1] - 1
        n_days = len(returns)
        n_years = n_days / 252
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Risk metrics
        volatility = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() * 252) / (volatility + 1e-6)

        # Drawdown
        max_dd = df['drawdown_pct'].min()
        avg_dd = df[df['drawdown_pct'] < 0]['drawdown_pct'].mean()

        # Trade metrics
        if len(trades_df) > 0:
            wins = trades_df[trades_df['return'] > 0]
            losses = trades_df[trades_df['return'] <= 0]

            win_rate = len(wins) / len(trades_df)
            avg_win = wins['return'].mean() if len(wins) > 0 else 0
            avg_loss = losses['return'].mean() if len(losses) > 0 else 0

            gross_profit = wins['return'].sum() if len(wins) > 0 else 0
            gross_loss = abs(losses['return'].sum()) if len(losses) > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

            avg_holding = trades_df['holding_days'].mean()
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_holding = 0

        # Buy and hold comparison
        bh_return = df['buy_hold_equity'].iloc[-1] - 1
        bh_cagr = (1 + bh_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        metrics = {
            'period': period_name,
            'total_return': total_return,
            'cagr': cagr,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'avg_drawdown': avg_dd,
            'num_trades': len(trades_df),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_holding_days': avg_holding,
            'buy_hold_return': bh_return,
            'buy_hold_cagr': bh_cagr,
        }

        return metrics

    @staticmethod
    def print_metrics(metrics: Dict):
        """Print performance metrics in a formatted table."""

        print("\n" + "="*80)
        print(f"PERFORMANCE METRICS - {metrics['period'].upper()}")
        print("="*80)

        print(f"\n📊 RETURNS")
        print(f"{'─'*80}")
        print(f"Total Return:        {metrics['total_return']*100:>8.2f}%")
        print(f"CAGR:                {metrics['cagr']*100:>8.2f}%")
        print(f"Volatility (ann.):   {metrics['volatility']*100:>8.2f}%")
        print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>8.2f}")

        print(f"\n📉 RISK")
        print(f"{'─'*80}")
        print(f"Max Drawdown:        {metrics['max_drawdown']*100:>8.2f}%")
        print(f"Avg Drawdown:        {metrics['avg_drawdown']*100:>8.2f}%")

        print(f"\n💼 TRADES")
        print(f"{'─'*80}")
        print(f"Number of Trades:    {metrics['num_trades']:>8,}")
        print(f"Win Rate:            {metrics['win_rate']*100:>8.1f}%")
        print(f"Avg Win:             {metrics['avg_win']*100:>8.2f}%")
        print(f"Avg Loss:            {metrics['avg_loss']*100:>8.2f}%")
        print(f"Profit Factor:       {metrics['profit_factor']:>8.2f}")
        print(f"Avg Holding (days):  {metrics['avg_holding_days']:>8.1f}")

        print(f"\n📈 BENCHMARK (Buy & Hold)")
        print(f"{'─'*80}")
        print(f"B&H Return:          {metrics['buy_hold_return']*100:>8.2f}%")
        print(f"B&H CAGR:            {metrics['buy_hold_cagr']*100:>8.2f}%")
        print(f"Outperformance:      {(metrics['total_return']-metrics['buy_hold_return'])*100:>8.2f}%")

        # Validation checks
        print(f"\n🎯 VALIDATION vs TARGETS")
        print(f"{'─'*80}")

        sharpe_pass = "✅" if metrics['sharpe_ratio'] > 1.0 else "❌"
        dd_pass = "✅" if metrics['max_drawdown'] > -0.15 else "❌"
        pf_pass = "✅" if metrics['profit_factor'] > 1.5 else "❌"

        print(f"Sharpe > 1.0:        {metrics['sharpe_ratio']:>8.2f} {sharpe_pass}")
        print(f"Max DD < 15%:        {metrics['max_drawdown']*100:>8.2f}% {dd_pass}")
        print(f"Profit Factor > 1.5: {metrics['profit_factor']:>8.2f} {pf_pass}")

        all_pass = all([
            metrics['sharpe_ratio'] > 1.0,
            metrics['max_drawdown'] > -0.15,
            metrics['profit_factor'] > 1.5
        ])

        if all_pass:
            print(f"\n✅ ALL TARGETS MET")
        else:
            print(f"\n⚠️  SOME TARGETS NOT MET")


# =============================================================================
# VISUALIZATION
# =============================================================================

class Visualizer:
    """Create performance visualizations."""

    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = config['output_dir']
        self.output_dir.mkdir(exist_ok=True)

    def plot_equity_curves(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Plot equity curves for train and validation."""

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Training period
        ax = axes[0]
        ax.plot(train_df.index, train_df['equity'], label='Strategy', linewidth=2)
        ax.plot(train_df.index, train_df['buy_hold_equity'], label='Buy & Hold', linewidth=2, alpha=0.7)
        ax.set_title('Training Period (2014-2023): Equity Curve', fontsize=14, fontweight='bold')
        ax.set_ylabel('Equity (Initial = 1.0)')
        ax.legend()
        ax.grid(alpha=0.3)

        # Validation period
        ax = axes[1]
        ax.plot(val_df.index, val_df['equity'], label='Strategy', linewidth=2)
        ax.plot(val_df.index, val_df['buy_hold_equity'], label='Buy & Hold', linewidth=2, alpha=0.7)
        ax.set_title('Validation Period (2024-2025): Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Equity (Initial = 1.0)')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if self.config['save_plots']:
            plt.savefig(self.output_dir / 'equity_curves.png', dpi=300, bbox_inches='tight')
            print(f"✓ Saved equity_curves.png")

        plt.close()

    def plot_drawdown(self, df: pd.DataFrame, period_name: str):
        """Plot drawdown chart."""

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.fill_between(df.index, 0, df['drawdown_pct']*100, alpha=0.3, color='red')
        ax.plot(df.index, df['drawdown_pct']*100, color='red', linewidth=1.5)
        ax.set_title(f'{period_name} Period: Drawdown from Peak', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.axhline(y=-15, color='orange', linestyle='--', label='15% Target', linewidth=2)
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if self.config['save_plots']:
            filename = f'drawdown_{period_name.lower().replace(" ", "_")}.png'
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved {filename}")

        plt.close()

    def plot_monthly_returns(self, df: pd.DataFrame, period_name: str):
        """Plot monthly returns heatmap."""

        # Calculate monthly returns
        df = df.copy()
        df['month'] = df.index.to_period('M')
        monthly = df.groupby('month')['net_return'].sum()

        # Pivot to heatmap format
        monthly_df = monthly.reset_index()
        monthly_df['year'] = monthly_df['month'].dt.year
        monthly_df['month_name'] = monthly_df['month'].dt.month

        pivot = monthly_df.pivot(index='year', columns='month_name', values='net_return')
        pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.5)))

        sns.heatmap(pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                    cbar_kws={'label': 'Monthly Return (%)'}, linewidths=0.5)

        ax.set_title(f'{period_name} Period: Monthly Returns (%)',
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Year')
        ax.set_xlabel('Month')

        plt.tight_layout()

        if self.config['save_plots']:
            filename = f'monthly_returns_{period_name.lower().replace(" ", "_")}.png'
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
            print(f"✓ Saved {filename}")

        plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""

    print("\n" + "="*80)
    print("MOMENTUM STRATEGY - PRODUCTION RUN")
    print("="*80)
    print(f"Strategy: Risk-adjusted multi-timeframe momentum")
    print(f"Asset: XAUUSD")
    print(f"Frequency: Daily")
    print(f"Train: {CONFIG['train_start']} to {CONFIG['train_end']}")
    print(f"Validation: {CONFIG['val_start']} to {CONFIG['val_end']}")
    print("="*80)

    # 1. Load data
    loader = DataLoader(CONFIG)
    df = loader.load_and_resample()

    # 2. Engineer features
    engineer = FeatureEngineer(CONFIG)
    df = engineer.engineer_features(df)

    # 3. Validation checks
    print("\n" + "="*80)
    print("DATA VALIDATION")
    print("="*80)

    feature_cols = [c for c in df.columns if 'risk_adj' in c or c in ['realized_vol', 'vol_regime']]
    nan_counts = df[feature_cols].isna().sum()

    if nan_counts.any():
        print("⚠️  NaN values found in features:")
        print(nan_counts[nan_counts > 0])
    else:
        print("✅ No NaN values in trading features")

    # Drop initial NaN rows (from rolling calculations)
    df_clean = df.dropna(subset=feature_cols)
    print(f"✅ Dropped {len(df) - len(df_clean)} rows with NaN")
    print(f"✅ Clean data: {len(df_clean):,} rows")

    # 4. Generate signals
    generator = SignalGenerator(CONFIG)
    df_clean = generator.generate_signals(df_clean)

    # 5. Split into train/validation
    train_df = df_clean[CONFIG['train_start']:CONFIG['train_end']].copy()
    val_df = df_clean[CONFIG['val_start']:CONFIG['val_end']].copy()

    print(f"\n✅ Train period: {len(train_df):,} days ({train_df.index.min().date()} to {train_df.index.max().date()})")
    print(f"✅ Validation period: {len(val_df):,} days ({val_df.index.min().date()} to {val_df.index.max().date()})")

    # 6. Run backtests
    backtester = Backtester(CONFIG)

    train_results, train_trades = backtester.run_backtest(train_df, "Training")
    val_results, val_trades = backtester.run_backtest(val_df, "Validation")

    # 7. Calculate metrics
    analyzer = PerformanceAnalyzer()

    train_metrics = analyzer.calculate_metrics(train_results, train_trades, "Training")
    val_metrics = analyzer.calculate_metrics(val_results, val_trades, "Validation")

    analyzer.print_metrics(train_metrics)
    analyzer.print_metrics(val_metrics)

    # 8. Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    viz = Visualizer(CONFIG)
    viz.plot_equity_curves(train_results, val_results)
    viz.plot_drawdown(train_results, "Training")
    viz.plot_drawdown(val_results, "Validation")
    viz.plot_monthly_returns(train_results, "Training")
    viz.plot_monthly_returns(val_results, "Validation")

    # 9. Save results
    if CONFIG['save_trades']:
        train_trades.to_csv(CONFIG['output_dir'] / 'trades_training.csv', index=False)
        val_trades.to_csv(CONFIG['output_dir'] / 'trades_validation.csv', index=False)
        print(f"✓ Saved trade CSVs")

    # Save summary
    summary = pd.DataFrame([train_metrics, val_metrics])
    summary.to_csv(CONFIG['output_dir'] / 'performance_summary.csv', index=False)
    print(f"✓ Saved performance_summary.csv")

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Results saved to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()
