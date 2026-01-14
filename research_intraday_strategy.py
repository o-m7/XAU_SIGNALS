#!/usr/bin/env python3
"""
XAUUSD Intraday Strategy Development - From First Principles

OBJECTIVE: Develop profitable 15-30min strategy meeting:
- Profit Factor ≥ 1.6
- Win Rate ≥ 52%
- Sharpe ≥ 0.25/trade
- Max Drawdown ≤ 6%
- R-multiple > 1.2
- 15-30 trades/day

METHODOLOGY:
1. Exploratory Data Analysis (returns, autocorrelation, stationarity)
2. Edge Hypothesis Generation & Testing (statistical validation)
3. Feature Engineering (driven by discovered edges)
4. Label Design (optimized for edge structure)
5. Model Selection & Training
6. Walk-Forward Validation
7. Regime Analysis & Robustness Testing

Author: Quant Research Team
Date: 2026-01-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import het_arch
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
np.random.seed(42)

# Paths
PROJECT_ROOT = Path("/Users/omar/Desktop/ML/xauusd_signals")
DATA_DIR = PROJECT_ROOT / "Raw Data"
MINUTE_DIR = DATA_DIR / "ohlcv_minute"
QUOTES_DIR = DATA_DIR / "quotes"
RESULTS_DIR = PROJECT_ROOT / "research_results"
RESULTS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("XAUUSD INTRADAY STRATEGY RESEARCH")
print("=" * 80)
print()


# =============================================================================
# PHASE 1: DATA LOADING & PREPARATION
# =============================================================================

def load_multi_year_minute_data(years=None, resample_freq='15T'):
    """
    Load multiple years of minute data and resample to desired frequency.

    Args:
        years: List of years to load. If None, loads 2020-2024 (5 years)
        resample_freq: Target frequency ('15T' or '30T')

    Returns:
        DataFrame with resampled OHLCV data
    """
    if years is None:
        years = [2020, 2021, 2022, 2023, 2024]  # 5 years for initial research

    print(f"[1] Loading {len(years)} years of minute data: {years}")
    print(f"    Resampling to {resample_freq}")
    print()

    dfs = []
    for year in years:
        path = MINUTE_DIR / f"XAUUSD_minute_{year}.parquet"
        if not path.exists():
            print(f"    WARNING: {path.name} not found, skipping")
            continue

        df = pd.read_parquet(path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp').sort_index()

        print(f"    {year}: {len(df):,} 1-min bars | "
              f"{df.index.min().date()} to {df.index.max().date()}")

        dfs.append(df)

    # Combine all years
    combined = pd.concat(dfs, axis=0)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]

    print(f"\n    Total 1-min bars: {len(combined):,}")
    print(f"    Date range: {combined.index.min()} to {combined.index.max()}")

    # Resample to target frequency
    resampled = combined.resample(resample_freq).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'vwap': 'mean',
        'trades': 'sum'
    })

    # Remove bars with no data (market closed)
    resampled = resampled.dropna(subset=['close'])

    print(f"    Resampled to {resample_freq}: {len(resampled):,} bars")
    print(f"    Avg bars/day: {len(resampled) / len(years) / 252:.1f}")
    print()

    return resampled


def add_basic_returns(df):
    """Add returns and log returns."""
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['returns_bps'] = df['returns'] * 10000  # basis points
    return df


# =============================================================================
# PHASE 2: EXPLORATORY DATA ANALYSIS
# =============================================================================

def eda_returns_distribution(df):
    """
    Analyze returns distribution.

    Questions:
    - Are returns normally distributed?
    - Evidence of fat tails?
    - Skewness/kurtosis?
    - Outliers?
    """
    print("[2] RETURNS DISTRIBUTION ANALYSIS")
    print("-" * 80)

    returns = df['returns'].dropna()

    # Descriptive statistics
    print("\nDescriptive Statistics (returns in bps):")
    print(f"  Count:    {len(returns):,}")
    print(f"  Mean:     {returns.mean() * 10000:.4f} bps")
    print(f"  Std:      {returns.std() * 10000:.4f} bps")
    print(f"  Skewness: {stats.skew(returns):.4f}")
    print(f"  Kurtosis: {stats.kurtosis(returns):.4f} (excess)")
    print(f"  Min:      {returns.min() * 10000:.4f} bps")
    print(f"  Max:      {returns.max() * 10000:.4f} bps")

    # Normality test
    _, p_value = stats.normaltest(returns)
    print(f"\n  Normality Test (D'Agostino-Pearson):")
    print(f"    p-value: {p_value:.6f}")
    print(f"    Verdict: {'NOT NORMAL' if p_value < 0.05 else 'Could be normal'} (p < 0.05 rejects normality)")

    # Percentiles
    print(f"\n  Percentiles:")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        val = returns.quantile(p/100)
        print(f"    {p:2d}%: {val * 10000:8.2f} bps")

    # Fat tails analysis
    tail_threshold = 2  # standard deviations
    left_tail = (returns < returns.mean() - tail_threshold * returns.std()).sum()
    right_tail = (returns > returns.mean() + tail_threshold * returns.std()).sum()
    expected_tail = len(returns) * 0.0228  # Normal distribution: ~2.28% beyond ±2σ

    print(f"\n  Fat Tails Analysis (beyond ±{tail_threshold}σ):")
    print(f"    Left tail:     {left_tail} ({left_tail/len(returns)*100:.2f}%)")
    print(f"    Right tail:    {right_tail} ({right_tail/len(returns)*100:.2f}%)")
    print(f"    Expected (normal): {expected_tail:.0f} ({expected_tail/len(returns)*100:.2f}%)")
    print(f"    Verdict: {'FAT TAILS PRESENT' if (left_tail + right_tail) > expected_tail * 2 else 'Normal-ish tails'}")

    print("\n")
    return returns


def eda_autocorrelation(df, lags=50):
    """
    Test autocorrelation in returns.

    Questions:
    - Do returns exhibit momentum (positive autocorr)?
    - Do returns exhibit mean reversion (negative autocorr)?
    - At what lags?
    """
    print("[3] AUTOCORRELATION ANALYSIS")
    print("-" * 80)

    returns = df['returns'].dropna()

    # ACF and PACF
    acf_vals = acf(returns, nlags=lags, fft=False)
    pacf_vals = pacf(returns, nlags=lags, method='ywm')

    # Find significant lags (beyond 95% confidence)
    conf_interval = 1.96 / np.sqrt(len(returns))

    print(f"\n  95% Confidence Interval: ±{conf_interval:.4f}")
    print(f"\n  Significant ACF lags (|ACF| > {conf_interval:.4f}):")

    sig_lags = []
    for lag in range(1, min(lags+1, len(acf_vals))):
        if abs(acf_vals[lag]) > conf_interval:
            sig_lags.append((lag, acf_vals[lag]))
            if lag <= 20:  # Print first 20
                print(f"    Lag {lag:2d}: {acf_vals[lag]:7.4f} {'(MOMENTUM)' if acf_vals[lag] > 0 else '(REVERSION)'}")

    if not sig_lags:
        print("    None - returns appear to be random walk")

    # Ljung-Box test for serial correlation
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_result = acorr_ljungbox(returns, lags=[1, 5, 10, 20], return_df=True)

    print(f"\n  Ljung-Box Test (serial correlation):")
    for idx, row in lb_result.iterrows():
        print(f"    Lag {idx:2d}: p-value = {row['lb_pvalue']:.4f} {'***' if row['lb_pvalue'] < 0.05 else ''}")

    # Interpretation
    if any(lb_result['lb_pvalue'] < 0.05):
        print(f"\n  VERDICT: SERIAL CORRELATION DETECTED (p < 0.05)")
        if acf_vals[1] > conf_interval:
            print(f"           → Lag-1 ACF = {acf_vals[1]:.4f} suggests SHORT-TERM MOMENTUM")
        elif acf_vals[1] < -conf_interval:
            print(f"           → Lag-1 ACF = {acf_vals[1]:.4f} suggests MEAN REVERSION")
    else:
        print(f"\n  VERDICT: No significant autocorrelation (random walk)")

    print("\n")
    return acf_vals, pacf_vals


def eda_stationarity(df):
    """
    Test stationarity of price and returns.

    Questions:
    - Are prices stationary? (should be NO - prices have trends)
    - Are returns stationary? (should be YES - returns are differences)
    """
    print("[4] STATIONARITY TESTS")
    print("-" * 80)

    # ADF test on prices
    print("\n  Augmented Dickey-Fuller Test on PRICES:")
    adf_price = adfuller(df['close'].dropna(), regression='ct', autolag='AIC')
    print(f"    ADF Statistic: {adf_price[0]:.4f}")
    print(f"    p-value:       {adf_price[1]:.6f}")
    print(f"    Critical Values:")
    for key, val in adf_price[4].items():
        print(f"      {key}: {val:.4f}")
    print(f"    Verdict: {'STATIONARY' if adf_price[1] < 0.05 else 'NON-STATIONARY (expected for prices)'}")

    # ADF test on returns
    print("\n  Augmented Dickey-Fuller Test on RETURNS:")
    returns = df['returns'].dropna()
    adf_returns = adfuller(returns, regression='c', autolag='AIC')
    print(f"    ADF Statistic: {adf_returns[0]:.4f}")
    print(f"    p-value:       {adf_returns[1]:.6f}")
    print(f"    Critical Values:")
    for key, val in adf_returns[4].items():
        print(f"      {key}: {val:.4f}")
    print(f"    Verdict: {'STATIONARY (good for modeling)' if adf_returns[1] < 0.05 else 'NON-STATIONARY (problem!)'}")

    print("\n")
    return adf_price, adf_returns


def eda_volatility(df):
    """
    Analyze volatility dynamics.

    Questions:
    - Is volatility constant or time-varying?
    - ARCH effects (volatility clustering)?
    - Intraday patterns?
    """
    print("[5] VOLATILITY ANALYSIS")
    print("-" * 80)

    returns = df['returns'].dropna()

    # Realized volatility (rolling std)
    df_vol = df.copy()
    df_vol['vol_20'] = df_vol['returns'].rolling(20).std() * np.sqrt(20 * 252) * 100  # Annualized %
    df_vol['vol_100'] = df_vol['returns'].rolling(100).std() * np.sqrt(100 * 252) * 100

    print(f"\n  Realized Volatility Statistics (annualized %):")
    print(f"    Mean (20-bar):  {df_vol['vol_20'].mean():.2f}%")
    print(f"    Std (20-bar):   {df_vol['vol_20'].std():.2f}%")
    print(f"    Min/Max:        {df_vol['vol_20'].min():.2f}% / {df_vol['vol_20'].max():.2f}%")

    # ARCH test for volatility clustering
    # Test if squared returns exhibit autocorrelation
    print(f"\n  ARCH Effects Test:")
    try:
        arch_test = het_arch(returns, nlags=10)
        print(f"    LM Statistic: {arch_test[0]:.4f}")
        print(f"    p-value:      {arch_test[1]:.6f}")
        print(f"    Verdict:      {'ARCH EFFECTS PRESENT (volatility clustering)' if arch_test[1] < 0.05 else 'No volatility clustering'}")
    except Exception as e:
        print(f"    ERROR: {e}")

    # Intraday volatility pattern
    df_hour = df.copy()
    df_hour['hour'] = df_hour.index.hour
    vol_by_hour = df_hour.groupby('hour')['returns'].std() * 10000  # bps

    print(f"\n  Volatility by Hour (UTC) - top 5 most volatile:")
    top_hours = vol_by_hour.nlargest(5)
    for hour, vol in top_hours.items():
        print(f"    {hour:02d}:00 - {vol:.2f} bps")

    print("\n")
    return df_vol


def eda_volume_patterns(df):
    """
    Analyze volume and trade patterns.

    Questions:
    - Volume clustering?
    - Session dependence?
    - Volume-price relationship?
    """
    print("[6] VOLUME & TRADE PATTERN ANALYSIS")
    print("-" * 80)

    # Volume statistics
    print(f"\n  Volume Statistics:")
    print(f"    Mean:   {df['volume'].mean():.2f}")
    print(f"    Median: {df['volume'].median():.2f}")
    print(f"    Std:    {df['volume'].std():.2f}")
    print(f"    CV:     {df['volume'].std() / df['volume'].mean():.2f} (coefficient of variation)")

    # Volume by hour
    df_hour = df.copy()
    df_hour['hour'] = df_hour.index.hour
    vol_by_hour = df_hour.groupby('hour')['volume'].mean()

    print(f"\n  Average Volume by Hour (UTC) - top 5:")
    top_vol_hours = vol_by_hour.nlargest(5)
    for hour, vol in top_vol_hours.items():
        session = get_session(hour)
        print(f"    {hour:02d}:00 - {vol:10.2f} ({session})")

    # Volume-returns correlation
    corr = df['volume'].corr(df['returns'].abs())
    print(f"\n  Volume vs |Returns| Correlation: {corr:.4f}")
    print(f"    Interpretation: {'Volume increases with volatility' if corr > 0.3 else 'Weak relationship'}")

    print("\n")


def get_session(hour_utc):
    """Map UTC hour to trading session."""
    if 0 <= hour_utc < 7:
        return "Asia/Pacific"
    elif 7 <= hour_utc < 16:
        return "London"
    elif 16 <= hour_utc < 22:
        return "NY"
    else:
        return "After-hours"


# =============================================================================
# PHASE 3: EDGE HYPOTHESIS GENERATION
# =============================================================================

def hypothesis_1_momentum_vs_reversion(df):
    """
    HYPOTHESIS 1: Do returns exhibit momentum or mean reversion?

    Test: Correlation between lagged returns
    Null Hypothesis: No correlation (random walk)
    Alternative: Positive (momentum) or negative (reversion) correlation
    """
    print("[7] HYPOTHESIS 1: Momentum vs Mean Reversion")
    print("-" * 80)

    df_test = df.copy()
    returns = df_test['returns'].dropna()

    # Test different lags
    print("\n  Return Autocorrelation Test:")
    for lag in [1, 2, 3, 5, 10, 20]:
        df_test[f'ret_lag{lag}'] = df_test['returns'].shift(lag)
        corr = df_test['returns'].corr(df_test[f'ret_lag{lag}'])

        # Significance test
        n = df_test[['returns', f'ret_lag{lag}']].dropna().shape[0]
        t_stat = corr * np.sqrt(n - 2) / np.sqrt(1 - corr**2)
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))

        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        edge = "MOMENTUM" if corr > 0 else "REVERSION"

        print(f"    Lag {lag:2d}: corr = {corr:7.4f}, p = {p_val:.4f} {sig:3s} {edge if sig else ''}")

    # Directional test: Are positive returns followed by positive returns?
    df_test['ret_pos'] = (df_test['returns'] > 0).astype(int)
    df_test['ret_pos_lag1'] = df_test['ret_pos'].shift(1)

    # Contingency table
    cont_table = pd.crosstab(df_test['ret_pos_lag1'], df_test['ret_pos'])
    chi2, p_val, _, _ = stats.chi2_contingency(cont_table)

    print(f"\n  Directional Persistence Test (Chi-Square):")
    print(f"    Chi2 = {chi2:.2f}, p-value = {p_val:.6f}")

    # Probability analysis
    prob_up_after_up = cont_table.loc[1, 1] / cont_table.loc[1].sum() if 1 in cont_table.index else 0
    prob_up_after_down = cont_table.loc[0, 1] / cont_table.loc[0].sum() if 0 in cont_table.index else 0

    print(f"    P(Up | Previous Up):   {prob_up_after_up:.4f}")
    print(f"    P(Up | Previous Down): {prob_up_after_down:.4f}")
    print(f"    Baseline P(Up):        {df_test['ret_pos'].mean():.4f}")

    if prob_up_after_up > prob_up_after_down + 0.02:
        print(f"\n  VERDICT: MOMENTUM EDGE DETECTED (up follows up)")
    elif prob_up_after_down > prob_up_after_up + 0.02:
        print(f"\n  VERDICT: REVERSION EDGE DETECTED (up follows down)")
    else:
        print(f"\n  VERDICT: NO CLEAR DIRECTIONAL EDGE")

    print("\n")


def hypothesis_2_session_effects(df):
    """
    HYPOTHESIS 2: Do different trading sessions have different characteristics?

    Sessions:
    - Asia/Pacific: 00:00-07:00 UTC
    - London: 07:00-16:00 UTC
    - NY: 16:00-22:00 UTC

    Test: Returns, volatility, win rate by session
    """
    print("[8] HYPOTHESIS 2: Session Effects")
    print("-" * 80)

    df_sess = df.copy()
    df_sess['hour'] = df_sess.index.hour
    df_sess['session'] = df_sess['hour'].apply(get_session)

    print("\n  Returns by Session:")
    session_stats = df_sess.groupby('session').agg({
        'returns': ['count', 'mean', 'std', lambda x: (x > 0).mean()]
    })
    session_stats.columns = ['count', 'mean_ret', 'std_ret', 'pct_positive']
    session_stats['mean_ret_bps'] = session_stats['mean_ret'] * 10000
    session_stats['std_ret_bps'] = session_stats['std_ret'] * 10000
    session_stats = session_stats.sort_values('mean_ret_bps', ascending=False)

    print(session_stats[['count', 'mean_ret_bps', 'std_ret_bps', 'pct_positive']].to_string())

    # ANOVA test: Are session returns significantly different?
    sessions = df_sess['session'].unique()
    session_groups = [df_sess[df_sess['session'] == s]['returns'].dropna() for s in sessions]
    f_stat, p_val = stats.f_oneway(*session_groups)

    print(f"\n  ANOVA Test (returns differ by session?):")
    print(f"    F-statistic = {f_stat:.4f}, p-value = {p_val:.6f}")
    print(f"    Verdict: {'SESSIONS DIFFER significantly' if p_val < 0.05 else 'No significant difference'}")

    print("\n")
    return session_stats


def hypothesis_3_volatility_regimes(df):
    """
    HYPOTHESIS 3: Does strategy performance depend on volatility regime?

    Regimes: Low / Medium / High volatility
    Based on realized volatility percentiles
    """
    print("[9] HYPOTHESIS 3: Volatility Regime Dependency")
    print("-" * 80)

    df_vol = df.copy()

    # Calculate realized vol
    df_vol['realized_vol'] = df_vol['returns'].rolling(20).std()

    # Define regimes by percentiles
    low_thresh = df_vol['realized_vol'].quantile(0.33)
    high_thresh = df_vol['realized_vol'].quantile(0.67)

    df_vol['regime'] = 'Medium'
    df_vol.loc[df_vol['realized_vol'] < low_thresh, 'regime'] = 'Low'
    df_vol.loc[df_vol['realized_vol'] > high_thresh, 'regime'] = 'High'

    print(f"\n  Volatility Regime Thresholds:")
    print(f"    Low:    vol < {low_thresh * 10000:.2f} bps")
    print(f"    Medium: vol = {low_thresh * 10000:.2f} - {high_thresh * 10000:.2f} bps")
    print(f"    High:   vol > {high_thresh * 10000:.2f} bps")

    print(f"\n  Statistics by Regime:")
    regime_stats = df_vol.groupby('regime').agg({
        'returns': ['count', 'mean', 'std', lambda x: (x > 0).mean()]
    })
    regime_stats.columns = ['count', 'mean_ret', 'std_ret', 'pct_positive']
    regime_stats['mean_ret_bps'] = regime_stats['mean_ret'] * 10000
    regime_stats['std_ret_bps'] = regime_stats['std_ret'] * 10000

    # Order by regime level
    regime_order = ['Low', 'Medium', 'High']
    regime_stats = regime_stats.reindex(regime_order)

    print(regime_stats[['count', 'mean_ret_bps', 'std_ret_bps', 'pct_positive']].to_string())

    # Test if regimes differ
    regime_groups = [df_vol[df_vol['regime'] == r]['returns'].dropna() for r in regime_order]
    f_stat, p_val = stats.f_oneway(*regime_groups)

    print(f"\n  ANOVA Test (returns differ by regime?):")
    print(f"    F-statistic = {f_stat:.4f}, p-value = {p_val:.6f}")

    print("\n")
    return df_vol


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    # Load data (5 years, 15-minute bars)
    df = load_multi_year_minute_data(
        years=[2020, 2021, 2022, 2023, 2024],
        resample_freq='15T'
    )

    # Add basic returns
    df = add_basic_returns(df)

    # Save initial dataset
    df.to_parquet(RESULTS_DIR / "data_15min_2020_2024.parquet")
    print(f"Saved: {RESULTS_DIR / 'data_15min_2020_2024.parquet'}")
    print()

    # ========================================================================
    # EXPLORATORY DATA ANALYSIS
    # ========================================================================

    returns = eda_returns_distribution(df)
    acf_vals, pacf_vals = eda_autocorrelation(df, lags=50)
    adf_price, adf_returns = eda_stationarity(df)
    df_vol = eda_volatility(df)
    eda_volume_patterns(df)

    # ========================================================================
    # EDGE HYPOTHESIS TESTING
    # ========================================================================

    hypothesis_1_momentum_vs_reversion(df)
    session_stats = hypothesis_2_session_effects(df)
    df_regime = hypothesis_3_volatility_regimes(df)

    # ========================================================================
    # SUMMARY & NEXT STEPS
    # ========================================================================

    print("=" * 80)
    print("RESEARCH PHASE 1 COMPLETE")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("  1. Review statistical test results above")
    print("  2. Identify which edges are statistically significant (p < 0.05)")
    print("  3. Design features to capture validated edges")
    print("  4. Design labels optimized for edge structure")
    print("  5. Build and train models")
    print("  6. Walk-forward validation")
    print()
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 80)
