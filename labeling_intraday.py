#!/usr/bin/env python3
"""
Labeling Methodology for Intraday Mean Reversion Strategy

Based on validated mean reversion edge:
- Lag-1 ACF = -0.0385 (p < 0.0001)
- P(Up | Down) = 51.92% > P(Up | Up) = 48.24%

Label Design:
- Triple-barrier method (profit target, stop loss, time limit)
- Optimize barrier ratios for R-multiple > 1.2 and WR ≥ 52%
- Test multiple configurations
- Meta-labels to filter weak signals (optional)

Author: Quant Research Team
Date: 2026-01-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


def get_daily_vol(returns, span=20):
    """
    Calculate dynamic volatility for barrier sizing.

    Uses exponentially weighted moving average for responsiveness.
    """
    return returns.ewm(span=span).std()


def apply_triple_barrier(
    df,
    pt_mult=2.0,
    sl_mult=1.0,
    max_hold_bars=20,
    vol_span=20
):
    """
    Apply triple-barrier labeling method.

    Barriers:
    1. Profit target: +pt_mult * volatility
    2. Stop loss: -sl_mult * volatility
    3. Time limit: max_hold_bars

    Args:
        df: DataFrame with OHLCV and returns
        pt_mult: Profit target multiplier (in volatility units)
        sl_mult: Stop loss multiplier (in volatility units)
        max_hold_bars: Maximum holding period
        vol_span: Lookback for volatility calculation

    Returns:
        DataFrame with labels: 1 (long), -1 (short), 0 (neutral)
    """
    print(f"\nApplying Triple Barrier Labels:")
    print(f"  Profit Target: {pt_mult}x vol")
    print(f"  Stop Loss:     {sl_mult}x vol")
    print(f"  Max Hold:      {max_hold_bars} bars")

    df = df.copy()

    # Calculate dynamic volatility
    daily_vol = get_daily_vol(df['returns'], span=vol_span)

    # Initialize output columns
    df['barrier_hit'] = 0  # Which barrier: 1=profit, -1=stop, 0=time
    df['hold_bars'] = 0
    df['forward_return'] = 0.0
    df['label'] = 0

    # For each bar, look forward and find which barrier hits first
    prices = df['close'].values
    returns = df['returns'].values
    vol = daily_vol.values

    for i in range(len(df) - max_hold_bars):
        entry_price = prices[i]
        entry_vol = vol[i]

        if np.isnan(entry_vol) or entry_vol == 0:
            continue

        # Define barriers
        profit_target = entry_price * (1 + pt_mult * entry_vol)
        stop_loss = entry_price * (1 - sl_mult * entry_vol)

        # Look forward
        hit_barrier = False
        for j in range(1, max_hold_bars + 1):
            if i + j >= len(prices):
                break

            future_price = prices[i + j]
            forward_ret = (future_price - entry_price) / entry_price

            # Check profit target
            if future_price >= profit_target:
                df.iloc[i, df.columns.get_loc('barrier_hit')] = 1
                df.iloc[i, df.columns.get_loc('hold_bars')] = j
                df.iloc[i, df.columns.get_loc('forward_return')] = forward_ret
                df.iloc[i, df.columns.get_loc('label')] = 1  # Profitable long
                hit_barrier = True
                break

            # Check stop loss
            if future_price <= stop_loss:
                df.iloc[i, df.columns.get_loc('barrier_hit')] = -1
                df.iloc[i, df.columns.get_loc('hold_bars')] = j
                df.iloc[i, df.columns.get_loc('forward_return')] = forward_ret
                df.iloc[i, df.columns.get_loc('label')] = -1  # Stopped out
                hit_barrier = True
                break

        # If no barrier hit, use end-of-period return
        if not hit_barrier:
            j = max_hold_bars
            if i + j < len(prices):
                future_price = prices[i + j]
                forward_ret = (future_price - entry_price) / entry_price
                df.iloc[i, df.columns.get_loc('barrier_hit')] = 0
                df.iloc[i, df.columns.get_loc('hold_bars')] = j
                df.iloc[i, df.columns.get_loc('forward_return')] = forward_ret

                # Label based on direction
                if forward_ret > 0:
                    df.iloc[i, df.columns.get_loc('label')] = 1
                elif forward_ret < 0:
                    df.iloc[i, df.columns.get_loc('label')] = -1

    # Statistics
    total = len(df)
    profit_hit = (df['barrier_hit'] == 1).sum()
    stop_hit = (df['barrier_hit'] == -1).sum()
    time_hit = (df['barrier_hit'] == 0).sum()

    print(f"\n  Barrier Hit Statistics:")
    print(f"    Profit Target: {profit_hit:6,} ({profit_hit/total*100:5.2f}%)")
    print(f"    Stop Loss:     {stop_hit:6,} ({stop_hit/total*100:5.2f}%)")
    print(f"    Time Limit:    {time_hit:6,} ({time_hit/total*100:5.2f}%)")

    # Label distribution
    longs = (df['label'] == 1).sum()
    shorts = (df['label'] == -1).sum()
    neutral = (df['label'] == 0).sum()

    print(f"\n  Label Distribution:")
    print(f"    Long (1):    {longs:6,} ({longs/total*100:5.2f}%)")
    print(f"    Short (-1):  {shorts:6,} ({shorts/total*100:5.2f}%)")
    print(f"    Neutral (0): {neutral:6,} ({neutral/total*100:5.2f}%)")

    # Win rate (of labeled trades)
    labeled = df[df['label'] != 0]
    if len(labeled) > 0:
        wins = (labeled['forward_return'] > 0).sum()
        win_rate = wins / len(labeled)
        print(f"\n  Win Rate (labeled trades): {win_rate*100:.2f}%")

        # R-multiple
        avg_win = labeled[labeled['forward_return'] > 0]['forward_return'].mean()
        avg_loss = labeled[labeled['forward_return'] < 0]['forward_return'].mean()
        if avg_loss != 0:
            r_multiple = abs(avg_win / avg_loss)
            print(f"  R-multiple (Avg Win / Avg Loss): {r_multiple:.2f}")

    return df


def create_mean_reversion_labels(df, config='default'):
    """
    Create labels optimized for mean reversion strategy.

    Configurations:
    - 'default': Balanced win rate (~50-55%), R-multiple > 1.2
    - 'conservative': Higher win rate (>55%), lower R-multiple
    - 'aggressive': Lower win rate (~48%), higher R-multiple

    Args:
        df: DataFrame with features
        config: Label configuration

    Returns:
        DataFrame with labels
    """
    configs = {
        'default': {
            'pt_mult': 1.5,
            'sl_mult': 1.0,
            'max_hold_bars': 12,  # ~3 hours at 15-min
            'vol_span': 20
        },
        'conservative': {
            'pt_mult': 1.2,
            'sl_mult': 0.8,
            'max_hold_bars': 8,
            'vol_span': 20
        },
        'aggressive': {
            'pt_mult': 2.0,
            'sl_mult': 1.0,
            'max_hold_bars': 20,
            'vol_span': 20
        },
        'tight': {
            'pt_mult': 1.0,
            'sl_mult': 0.5,
            'max_hold_bars': 6,
            'vol_span': 10
        }
    }

    params = configs.get(config, configs['default'])

    print(f"\nCreating labels with config: {config}")
    print(f"  Parameters: {params}")

    df_labeled = apply_triple_barrier(df, **params)

    return df_labeled


def test_multiple_label_configs(df):
    """
    Test multiple labeling configurations to find optimal.

    Returns best config based on:
    - Win Rate ≥ 52%
    - R-multiple > 1.2
    - Label balance (not too skewed)
    """
    print("=" * 80)
    print("TESTING MULTIPLE LABEL CONFIGURATIONS")
    print("=" * 80)

    configs = ['default', 'conservative', 'aggressive', 'tight']
    results = {}

    for config in configs:
        print(f"\n{'-' * 80}")
        print(f"Configuration: {config.upper()}")
        print(f"{'-' * 80}")

        df_test = df.copy()
        df_labeled = create_mean_reversion_labels(df_test, config=config)

        # Calculate metrics
        labeled = df_labeled[df_labeled['label'] != 0]

        if len(labeled) > 0:
            wins = (labeled['forward_return'] > 0).sum()
            win_rate = wins / len(labeled)

            avg_win = labeled[labeled['forward_return'] > 0]['forward_return'].mean()
            avg_loss = labeled[labeled['forward_return'] < 0]['forward_return'].mean()

            if avg_loss != 0 and not np.isnan(avg_loss):
                r_multiple = abs(avg_win / avg_loss)
            else:
                r_multiple = 0

            # Sharpe proxy (mean / std of forward returns)
            sharpe = labeled['forward_return'].mean() / labeled['forward_return'].std() if labeled['forward_return'].std() > 0 else 0

            # Label balance
            longs = (df_labeled['label'] == 1).sum()
            shorts = (df_labeled['label'] == -1).sum()
            balance = min(longs, shorts) / max(longs, shorts) if max(longs, shorts) > 0 else 0

            results[config] = {
                'win_rate': win_rate,
                'r_multiple': r_multiple,
                'sharpe': sharpe,
                'balance': balance,
                'n_trades': len(labeled),
                'avg_hold': labeled['hold_bars'].mean()
            }

    # Print summary
    print("\n" + "=" * 80)
    print("CONFIGURATION COMPARISON")
    print("=" * 80)

    df_results = pd.DataFrame(results).T
    df_results = df_results[['win_rate', 'r_multiple', 'sharpe', 'n_trades', 'avg_hold', 'balance']]

    print(df_results.to_string())

    # Recommend best
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Score: win_rate * r_multiple * (1 + sharpe)
    df_results['score'] = df_results['win_rate'] * df_results['r_multiple'] * (1 + df_results['sharpe'].clip(lower=0))

    best_config = df_results['score'].idxmax()

    print(f"\nBest Configuration: {best_config.upper()}")
    print(f"  Win Rate:   {df_results.loc[best_config, 'win_rate']*100:.2f}%")
    print(f"  R-multiple: {df_results.loc[best_config, 'r_multiple']:.2f}")
    print(f"  Sharpe:     {df_results.loc[best_config, 'sharpe']:.4f}")
    print(f"  Trades:     {df_results.loc[best_config, 'n_trades']:.0f}")
    print(f"  Avg Hold:   {df_results.loc[best_config, 'avg_hold']:.1f} bars")

    # Check if meets targets
    meets_wr = df_results.loc[best_config, 'win_rate'] >= 0.52
    meets_r = df_results.loc[best_config, 'r_multiple'] > 1.2

    print(f"\n  Meets Win Rate ≥ 52%: {'✅ YES' if meets_wr else '❌ NO'}")
    print(f"  Meets R-multiple > 1.2: {'✅ YES' if meets_r else '❌ NO'}")

    return best_config, df_results


if __name__ == "__main__":
    from pathlib import Path

    RESULTS_DIR = Path("/Users/omar/Desktop/ML/xauusd_signals/research_results")

    # Load featured data
    df = pd.read_parquet(RESULTS_DIR / "data_15min_2020_2024_features.parquet")

    print("=" * 80)
    print("LABEL DESIGN & OPTIMIZATION")
    print("=" * 80)
    print(f"\nInput data: {df.shape}")
    print()

    # Test multiple configurations
    best_config, config_results = test_multiple_label_configs(df)

    # Apply best configuration
    print("\n" + "=" * 80)
    print("APPLYING BEST CONFIGURATION")
    print("=" * 80)

    df_final = create_mean_reversion_labels(df, config=best_config)

    # Save
    output_path = RESULTS_DIR / "data_15min_2020_2024_labeled.parquet"
    df_final.to_parquet(output_path)

    print(f"\n\nSaved labeled dataset to: {output_path}")
    print(f"Shape: {df_final.shape}")
    print()

    # Save config results
    config_results.to_csv(RESULTS_DIR / "label_config_comparison.csv")
    print(f"Saved config comparison to: {RESULTS_DIR / 'label_config_comparison.csv'}")
    print()
