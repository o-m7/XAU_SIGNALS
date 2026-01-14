#!/usr/bin/env python3
"""
Analyze the impact of the 200-day MA regime filter on momentum strategy.
Compare before/after results and visualize why it didn't help.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Results from original run (before regime filter)
original_results = {
    'Training': {
        'total_return': 0.2484,
        'sharpe': 0.28,
        'max_dd': -0.1849,
        'profit_factor': 1.32,
        'win_rate': 0.415,
        'trades': 53
    },
    'Validation': {
        'total_return': 0.4046,
        'sharpe': 2.12,
        'max_dd': -0.0710,
        'profit_factor': 11.34,
        'win_rate': 0.75,
        'trades': 4
    }
}

# Results with regime filter (after adding 200-day MA)
regime_filter_results = {
    'Training': {
        'total_return': 0.2402,
        'sharpe': 0.28,
        'max_dd': -0.2274,
        'profit_factor': 1.35,
        'win_rate': 0.377,
        'trades': 53
    },
    'Validation': {
        'total_return': 0.4046,
        'sharpe': 2.12,
        'max_dd': -0.0710,
        'profit_factor': 11.34,
        'win_rate': 0.75,
        'trades': 4
    }
}

# Create comparison visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Regime Filter Impact: 200-Day MA Filter vs Original Strategy',
             fontsize=16, fontweight='bold')

# Metrics to compare
metrics = [
    ('total_return', 'Total Return', True),
    ('sharpe', 'Sharpe Ratio', True),
    ('max_dd', 'Max Drawdown', False),
    ('profit_factor', 'Profit Factor', True),
    ('win_rate', 'Win Rate', True),
    ('trades', 'Number of Trades', False)
]

for idx, (metric_key, metric_name, higher_is_better) in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]

    # Extract values
    orig_train = original_results['Training'][metric_key]
    orig_val = original_results['Validation'][metric_key]
    regime_train = regime_filter_results['Training'][metric_key]
    regime_val = regime_filter_results['Validation'][metric_key]

    # Bar positions
    x = np.arange(2)
    width = 0.35

    # Plot bars
    bars1 = ax.bar(x - width/2, [orig_train, orig_val], width,
                   label='Original', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, [regime_train, regime_val], width,
                   label='With Regime Filter', alpha=0.8, color='coral')

    # Formatting
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name}', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Training', 'Validation'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        if metric_key in ['total_return', 'max_dd', 'win_rate']:
            label = f'{height*100:.1f}%'
        elif metric_key == 'trades':
            label = f'{int(height)}'
        else:
            label = f'{height:.2f}'

        ax.text(bar.get_x() + bar.get_width()/2., height,
                label, ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=9, fontweight='bold')

    # Color-code for improvement/degradation (training only)
    if idx < 3:  # Only annotate training period
        change = regime_train - orig_train
        if abs(change) > 0.01:  # Significant change
            if metric_key == 'max_dd':
                # For drawdown, less negative is better
                color = 'green' if change > 0 else 'red'
            else:
                color = 'green' if (change > 0 and higher_is_better) or (change < 0 and not higher_is_better) else 'red'

            ax.text(0, ax.get_ylim()[1] * 0.9,
                   '⚠️ WORSE' if color == 'red' else '✓ Better',
                   color=color, fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig('momentum_strategy_results/regime_filter_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved regime_filter_comparison.png")

# Create detailed metrics table
print("\n" + "="*80)
print("REGIME FILTER IMPACT ANALYSIS")
print("="*80)

print("\nTRAINING PERIOD (2014-2023):")
print("-" * 80)
print(f"{'Metric':<20} {'Original':<15} {'Regime Filter':<15} {'Change':<15}")
print("-" * 80)

for metric_key, metric_name, higher_is_better in metrics:
    orig = original_results['Training'][metric_key]
    regime = regime_filter_results['Training'][metric_key]
    change = regime - orig

    if metric_key in ['total_return', 'max_dd', 'win_rate']:
        orig_str = f'{orig*100:.2f}%'
        regime_str = f'{regime*100:.2f}%'
        change_str = f'{change*100:+.2f}%'
    elif metric_key == 'trades':
        orig_str = f'{int(orig)}'
        regime_str = f'{int(regime)}'
        change_str = f'{int(change):+d}'
    else:
        orig_str = f'{orig:.2f}'
        regime_str = f'{regime:.2f}'
        change_str = f'{change:+.2f}'

    # Determine if this is an improvement
    if metric_key == 'max_dd':
        status = '✓' if change > 0 else '⚠️'
    elif higher_is_better:
        status = '✓' if change > 0 else '⚠️'
    else:
        status = '✓' if abs(change) < 0.01 else '~'

    print(f"{metric_name:<20} {orig_str:<15} {regime_str:<15} {change_str:<15} {status}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\n❌ Regime filter WORSENED training performance:")
print("   • Max Drawdown: -18.5% → -22.7% (WORSE by 4.2%)")
print("   • Win Rate: 41.5% → 37.7% (WORSE by 3.8%)")
print("   • Total Return: 24.8% → 24.0% (WORSE by 0.8%)")
print("   • Sharpe Ratio: 0.28 → 0.28 (NO CHANGE)")
print("\n✓ Small improvements:")
print("   • Profit Factor: 1.32 → 1.35 (+0.03)")
print("\n⚠️  Validation unchanged (was already 100% above 200-day MA)")
print("\n🎯 RECOMMENDATION: Use ORIGINAL strategy without regime filter")
print("   Reason: Lower Max DD, same Sharpe, simpler implementation")

print("\n" + "="*80)
plt.close()
