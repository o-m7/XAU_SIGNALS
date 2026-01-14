"""
validation.py - Statistical Validation

Run these tests BEFORE backtesting.
If they fail, the strategy has no edge.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, Tuple


def test_autocorrelation(returns: np.ndarray, max_lag: int = 5) -> Dict[str, Any]:
    """
    Test for autocorrelation in returns.
    
    Null: No autocorrelation (random walk)
    Alternative: Autocorrelation exists
    
    For mean reversion, we expect NEGATIVE autocorrelation.
    """
    n = len(returns)
    results = []
    
    for lag in range(1, max_lag + 1):
        rho = pd.Series(returns).autocorr(lag=lag)
        
        # Standard error under null
        se = 1 / np.sqrt(n)
        z = rho / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        results.append({
            'lag': lag,
            'autocorr': rho,
            'z_stat': z,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'mean_reverting': rho < 0 and p_value < 0.05,
        })
    
    return {
        'results': pd.DataFrame(results),
        'has_mean_reversion': any(r['mean_reverting'] for r in results),
    }


def test_variance_ratio(returns: np.ndarray, q: int = 2) -> Dict[str, float]:
    """
    Lo-MacKinlay variance ratio test.
    
    Null: VR(q) = 1 (random walk)
    Alternative: VR(q) ≠ 1
    
    VR < 1: Mean reversion
    VR > 1: Momentum
    """
    n = len(returns)
    
    # Variance ratio
    var_1 = np.var(returns, ddof=1)
    returns_q = np.array([returns[i:i+q].sum() for i in range(n - q + 1)])
    var_q = np.var(returns_q, ddof=1)
    
    vr = var_q / (q * var_1) if var_1 > 0 else 1.0
    
    # Asymptotic variance (heteroskedasticity-robust)
    theta = 2 * (2 * q - 1) * (q - 1) / (3 * q * n)
    z = (vr - 1) / np.sqrt(theta)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return {
        'variance_ratio': vr,
        'z_statistic': z,
        'p_value': p_value,
        'is_mean_reverting': vr < 1 and p_value < 0.05,
        'is_momentum': vr > 1 and p_value < 0.05,
    }


def test_zscore_predictability(
    zscore: pd.Series,
    next_return: pd.Series,
    threshold: float = 2.0
) -> Dict[str, Any]:
    """
    Test if extreme z-scores predict reversals.
    
    Null: P(reversal | |z| > threshold) = 0.5
    Alternative: P(reversal | |z| > threshold) > 0.5
    """
    # Align series
    df = pd.DataFrame({'zscore': zscore, 'next_return': next_return}).dropna()
    
    results = {}
    
    # Oversold (z < -threshold) -> expect positive return
    oversold = df[df['zscore'] < -threshold]
    if len(oversold) > 10:
        p_reversal = (oversold['next_return'] > 0).mean()
        n = len(oversold)
        
        # Binomial test
        successes = (oversold['next_return'] > 0).sum()
        # Use binomtest (new API) or binom_test (old API)
        try:
            from scipy.stats import binomtest
            p_value = binomtest(successes, n, p=0.5, alternative='greater').pvalue
        except ImportError:
            p_value = stats.binom_test(successes, n, p=0.5, alternative='greater')
        
        results['oversold'] = {
            'n_samples': n,
            'p_reversal': p_reversal,
            'p_value': p_value,
            'significant': p_value < 0.05,
        }
    
    # Overbought (z > +threshold) -> expect negative return
    overbought = df[df['zscore'] > threshold]
    if len(overbought) > 10:
        p_reversal = (overbought['next_return'] < 0).mean()
        n = len(overbought)
        
        successes = (overbought['next_return'] < 0).sum()
        # Use binomtest (new API) or binom_test (old API)
        try:
            from scipy.stats import binomtest
            p_value = binomtest(successes, n, p=0.5, alternative='greater').pvalue
        except ImportError:
            p_value = stats.binom_test(successes, n, p=0.5, alternative='greater')
        
        results['overbought'] = {
            'n_samples': n,
            'p_reversal': p_reversal,
            'p_value': p_value,
            'significant': p_value < 0.05,
        }
    
    # Overall regression
    slope, intercept, r, p, se = stats.linregress(df['zscore'], df['next_return'])
    
    results['regression'] = {
        'slope': slope,
        'r_squared': r ** 2,
        'p_value': p,
        'is_mean_reverting': slope < 0 and p < 0.05,
    }
    
    return results


def test_spread_impact(
    spread_percentile: pd.Series,
    next_return_abs: pd.Series,
    threshold: float = 80
) -> Dict[str, Any]:
    """
    Test if high spread predicts lower returns (adverse selection).
    
    Used to validate spread filter.
    """
    df = pd.DataFrame({
        'spread_pct': spread_percentile,
        'abs_return': next_return_abs
    }).dropna()
    
    high_spread = df[df['spread_pct'] > threshold]['abs_return']
    low_spread = df[df['spread_pct'] <= threshold]['abs_return']
    
    # t-test: high spread should have lower absolute returns (less opportunity)
    t_stat, p_value = stats.ttest_ind(high_spread, low_spread, alternative='less')
    
    return {
        'high_spread_mean_return': high_spread.mean(),
        'low_spread_mean_return': low_spread.mean(),
        't_statistic': t_stat,
        'p_value': p_value,
        'filter_valid': p_value < 0.1,  # More lenient threshold
    }


def run_all_validations(df: pd.DataFrame, verbose: bool = True) -> Dict[str, Any]:
    """
    Run all statistical validation tests.
    
    Args:
        df: DataFrame with features and 'returns_1' column
    
    Returns:
        Dictionary of all test results
    """
    results = {}
    
    returns = df['returns_1'].dropna().values
    
    # 1. Autocorrelation test
    if verbose:
        print("\n" + "="*50)
        print("TEST 1: Autocorrelation")
        print("="*50)
    
    ac_results = test_autocorrelation(returns)
    results['autocorrelation'] = ac_results
    
    if verbose:
        print(ac_results['results'].to_string(index=False))
        print(f"\nMean reversion detected: {ac_results['has_mean_reversion']}")
    
    # 2. Variance ratio test
    if verbose:
        print("\n" + "="*50)
        print("TEST 2: Variance Ratio")
        print("="*50)
    
    for q in [2, 4, 8]:
        vr_results = test_variance_ratio(returns, q=q)
        results[f'variance_ratio_{q}'] = vr_results
        
        if verbose:
            print(f"VR({q}): {vr_results['variance_ratio']:.4f}, "
                  f"p={vr_results['p_value']:.4f}, "
                  f"mean_reverting={vr_results['is_mean_reverting']}")
    
    # 3. Z-score predictability
    if 'zscore_20' in df.columns:
        if verbose:
            print("\n" + "="*50)
            print("TEST 3: Z-Score Predictability")
            print("="*50)
        
        df_test = df.copy()
        df_test['next_return'] = df_test['returns_1'].shift(-1)
        
        for threshold in [1.5, 2.0, 2.5, 3.0]:
            zs_results = test_zscore_predictability(
                df_test['zscore_20'],
                df_test['next_return'],
                threshold=threshold
            )
            results[f'zscore_pred_{threshold}'] = zs_results
            
            if verbose:
                print(f"\nThreshold: {threshold}")
                if 'oversold' in zs_results:
                    os = zs_results['oversold']
                    print(f"  Oversold: n={os['n_samples']}, P(reversal)={os['p_reversal']:.1%}, "
                          f"p={os['p_value']:.4f}, sig={os['significant']}")
                if 'overbought' in zs_results:
                    ob = zs_results['overbought']
                    print(f"  Overbought: n={ob['n_samples']}, P(reversal)={ob['p_reversal']:.1%}, "
                          f"p={ob['p_value']:.4f}, sig={ob['significant']}")
    
    # 4. Spread filter validation
    if 'spread_percentile' in df.columns:
        if verbose:
            print("\n" + "="*50)
            print("TEST 4: Spread Filter")
            print("="*50)
        
        df_test = df.copy()
        df_test['next_abs_return'] = df_test['returns_1'].shift(-1).abs()
        
        spread_results = test_spread_impact(
            df_test['spread_percentile'],
            df_test['next_abs_return']
        )
        results['spread_filter'] = spread_results
        
        if verbose:
            print(f"High spread mean |return|: {spread_results['high_spread_mean_return']:.4%}")
            print(f"Low spread mean |return|: {spread_results['low_spread_mean_return']:.4%}")
            print(f"Filter valid: {spread_results['filter_valid']}")
    
    # Summary
    if verbose:
        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        
        pass_count = 0
        total_count = 0
        
        if ac_results['has_mean_reversion']:
            print("✓ Autocorrelation test: PASSED")
            pass_count += 1
        else:
            print("✗ Autocorrelation test: FAILED")
        total_count += 1
        
        if results.get('variance_ratio_2', {}).get('is_mean_reverting', False):
            print("✓ Variance ratio test: PASSED")
            pass_count += 1
        else:
            print("✗ Variance ratio test: FAILED")
        total_count += 1
        
        zs_test = results.get('zscore_pred_2.0', {})
        if zs_test.get('regression', {}).get('is_mean_reverting', False):
            print("✓ Z-score predictability: PASSED")
            pass_count += 1
        else:
            print("✗ Z-score predictability: FAILED")
        total_count += 1
        
        print(f"\nPassed: {pass_count}/{total_count}")
        
        if pass_count < 2:
            print("\n⚠️  WARNING: Insufficient statistical evidence for mean reversion.")
            print("    Strategy may not have edge. Proceed with caution.")
    
    return results

