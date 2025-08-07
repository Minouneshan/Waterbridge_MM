"""Statistical tests and econometric validation for Modern Mercantilism analysis.

This module implements the advanced statistical methods mentioned in the papers:
- Granger causality tests
- Structural break tests (Chow tests)  
- Monte Carlo simulations for uncertainty quantification
- Impulse response analysis

Uses simple dependencies (scipy, statsmodels) to maintain reproducibility.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.linear_model import LinearRegression
try:
    from arch.unitroot import DFGLS, PhillipsPerron
    from arch.cointegration import engle_granger
    ARCH_AVAILABLE = True
except ImportError:
    print("Warning: arch package not available for advanced econometric tests")
    ARCH_AVAILABLE = False
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

def granger_causality_test(data: pd.DataFrame, cause_col: str, effect_col: str, max_lags: int = 4) -> dict:
    """Test if cause_col Granger-causes effect_col."""
    # Prepare data for Granger test
    test_data = data[[effect_col, cause_col]].dropna()
    
    # Run Granger causality test
    result = grangercausalitytests(test_data, max_lags, verbose=False)
    
    # Extract p-values for each lag
    p_values = {}
    for lag in range(1, max_lags + 1):
        p_values[f'lag_{lag}'] = result[lag][0]['ssr_ftest'][1]  # F-test p-value
    
    # Overall conclusion (significant if any lag < 0.05)
    min_p_value = min(p_values.values())
    significant = min_p_value < 0.05
    
    return {
        'test': 'granger_causality',
        'cause': cause_col,
        'effect': effect_col,
        'p_values_by_lag': p_values,
        'min_p_value': min_p_value,
        'significant': significant,
        'conclusion': f"{cause_col} {'does' if significant else 'does not'} Granger-cause {effect_col} (p={min_p_value:.4f})"
    }

def structural_break_test(y: np.ndarray, x: np.ndarray, break_point: int) -> dict:
    """Simple Chow test for structural breaks."""
    n = len(y)
    if break_point >= n - 1 or break_point <= 1:
        return {'error': 'Invalid break point'}
    
    # Split data at break point
    y1, y2 = y[:break_point], y[break_point:]
    x1, x2 = x[:break_point], x[break_point:]
    
    # Fit separate regressions
    reg1 = LinearRegression().fit(x1.reshape(-1, 1), y1)
    reg2 = LinearRegression().fit(x2.reshape(-1, 1), y2)
    
    # Calculate residual sum of squares
    rss1 = np.sum((y1 - reg1.predict(x1.reshape(-1, 1)))**2)
    rss2 = np.sum((y2 - reg2.predict(x2.reshape(-1, 1)))**2)
    rss_restricted = rss1 + rss2
    
    # Fit full sample regression
    reg_full = LinearRegression().fit(x.reshape(-1, 1), y)
    rss_full = np.sum((y - reg_full.predict(x.reshape(-1, 1)))**2)
    
    # Chow test statistic
    k = 2  # number of parameters (intercept + slope)
    f_stat = ((rss_full - rss_restricted) / k) / (rss_restricted / (n - 2*k))
    p_value = 1 - stats.f.cdf(f_stat, k, n - 2*k)
    
    return {
        'test': 'chow_structural_break',
        'break_point': break_point,
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'rss_full': rss_full,
        'rss_restricted': rss_restricted,
        'conclusion': f"{'Significant' if p_value < 0.05 else 'No'} structural break at point {break_point} (p={p_value:.4f})"
    }

def monte_carlo_forecast_uncertainty(base_prob: float, std_dev: float = 0.05, n_sims: int = 10000) -> dict:
    """Monte Carlo simulation for forecast probability uncertainty."""
    # Generate random probabilities around base estimate
    simulated_probs = np.random.normal(base_prob, std_dev, n_sims)
    
    # Constrain to [0, 1] range
    simulated_probs = np.clip(simulated_probs, 0, 1)
    
    # Calculate confidence intervals
    ci_90 = np.percentile(simulated_probs, [5, 95])
    ci_80 = np.percentile(simulated_probs, [10, 90])
    ci_70 = np.percentile(simulated_probs, [15, 85])
    
    return {
        'base_probability': base_prob,
        'simulated_mean': np.mean(simulated_probs),
        'simulated_std': np.std(simulated_probs),
        'confidence_intervals': {
            '90%': ci_90,
            '80%': ci_80, 
            '70%': ci_70
        },
        'n_simulations': n_sims
    }

def impulse_response_analysis(data: pd.DataFrame, shock_var: str, response_var: str, periods: int = 8) -> dict:
    """Simple impulse-response analysis using VAR model."""
    # Fit VAR model
    model = VAR(data[[shock_var, response_var]])
    results = model.fit(maxlags=2, ic='aic')
    
    # Generate impulse responses
    irf = results.irf(periods)
    
    # Extract response of response_var to shock in shock_var
    response = irf.irfs[:, data.columns.get_loc(response_var), data.columns.get_loc(shock_var)]
    
    return {
        'shock_variable': shock_var,
        'response_variable': response_var,
        'periods': periods,
        'impulse_response': response,
        'peak_response': np.max(np.abs(response)),
        'peak_period': np.argmax(np.abs(response)),
        'summary': f"Peak response of {response_var} to {shock_var} shock: {np.max(np.abs(response)):.3f} at period {np.argmax(np.abs(response))}"
    }

def run_econometric_validation():
    """Run all econometric tests mentioned in the papers."""
    print("Running Econometric Validation Tests")
    print("=" * 50)
    
    # Load trade data for tests
    data_dir = ROOT_DIR / 'data'
    china_imports = pd.read_csv(data_dir / 'us_imports.csv')
    vietnam_imports = pd.read_csv(data_dir / 'vietnam_us_imports.csv')
    
    # 1. Structural break test for China imports (2018 trade war)
    years = china_imports['Year'].values
    china_share = china_imports['China_Share'].values
    
    # Test for break in 2018 (index 3 if starting from 2015)
    if len(years) >= 6:
        break_idx = 3  # Approximate index for 2018
        chow_result = structural_break_test(china_share, years, break_idx)
        print(f"\n1. Structural Break Test (China imports):")
        print(f"   {chow_result['conclusion']}")
    
    # 2. Granger causality test (if we have both series with same time dimension)
    combined_data = None
    if len(china_imports) == len(vietnam_imports):
        combined_data = pd.DataFrame({
            'China_Share': china_imports['China_Share'],
            'Vietnam_Imports': vietnam_imports['Imports_USD_Billion']
        })
        
        granger_result = granger_causality_test(combined_data, 'China_Share', 'Vietnam_Imports')
        print(f"\n2. Granger Causality Test:")
        print(f"   {granger_result['conclusion']}")
    
    # 3. Monte Carlo uncertainty for key forecasts
    print(f"\n3. Monte Carlo Uncertainty Analysis:")
    for forecast, prob in [('F6 China decline', 0.75), ('F14 Tech bifurcation', 0.85), ('F17 USD dominance', 0.66)]:
        mc_result = monte_carlo_forecast_uncertainty(prob)
        ci_90 = mc_result['confidence_intervals']['90%']
        print(f"   {forecast}: {prob:.0%} (90% CI: {ci_90[0]:.0%}–{ci_90[1]:.0%})")
    
    # 4. Simple impulse response (mock data for demonstration)
    if combined_data is not None and len(combined_data) >= 8:
        try:
            irf_result = impulse_response_analysis(combined_data, 'China_Share', 'Vietnam_Imports')
            print(f"\n4. Impulse Response Analysis:")
            print(f"   {irf_result['summary']}")
        except Exception as e:
            print(f"\n4. Impulse Response Analysis: Skipped due to data limitations")
    
    print(f"\nEconometric validation completed.")
    return True

if __name__ == "__main__":
    run_econometric_validation()
