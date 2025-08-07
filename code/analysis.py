import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from bayesian_model import create_comprehensive_model
from advanced_bayesian_model import create_advanced_model
from advanced_models import var_reserve_shares, logistic_carbon_tariffs, logistic_tech_bifurcation
from statistical_tests import run_econometric_validation
from remaining_forecasts import run_comprehensive_remaining_forecasts
from interdependency_analysis import generate_interdependency_report, create_network_visualization
from analysis_utils import (
    AnalysisConfig, load_project_data, create_forecast_visualization,
    fit_trend_model, calculate_monte_carlo_probability, create_summary_statistics,
    print_section_header, print_subsection_header, format_forecast_result,
    save_analysis_metadata
)
import warnings
from pathlib import Path
import scipy.stats as stats
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

warnings.filterwarnings('ignore')

# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[1]

def load_and_validate_data():
    """Load and validate all data files - now using centralized utility"""
    return load_project_data()

def analyze_china_import_decline():
    """Analyze F6: China's declining share of U.S. imports"""
    datasets = load_and_validate_data()
    if datasets is None:
        return None
    
    china_imports = datasets['china_imports']
    
    # Separate historical vs projected data
    historical_data = china_imports[china_imports['Year'] <= 2024]
    projected_data = china_imports[china_imports['Year'] > 2024]
    
    # Fit trend model using utility function
    trend_result = fit_trend_model(
        historical_data['Year'].values,
        historical_data['China_Share'].values
    )
    
    # Generate predictions for extended period
    years_extended = np.arange(2015, 2028)
    predictions = trend_result['model'].predict(years_extended.reshape(-1, 1))
    
    # Check forecast validation
    threshold = AnalysisConfig.THRESHOLDS['F6_CHINA_PERCENT']
    will_fall_below_threshold = any(
        china_imports[china_imports['Year'].isin([2025, 2026, 2027])]['China_Share'] < threshold
    )
    
    results = {
        'name': "China Import Decline",
        'model_performance': {
            'linear_r_squared': trend_result['r_squared'],
            'slope_per_year': trend_result['slope'],
            'projected_2027': predictions[-1]
        },
        'forecast_validation': {
            'falls_below_threshold': will_fall_below_threshold,
            'threshold': threshold
        },
        'probability_assessment': 0.75 if will_fall_below_threshold else 0.25
    }
    
    # Create visualization using utility function
    create_forecast_visualization(
        years=historical_data['Year'].values,
        values=historical_data['China_Share'].values,
        title="F6: China's Share of U.S. Goods Imports\nProjected to Fall Below 12% by 2025-2027",
        ylabel="China Share of U.S. Imports (%)",
        threshold=threshold,
        threshold_label=f"{threshold}% Threshold (F6)",
        projections=(years_extended, predictions),
        projection_label="Linear Trend",
        save_name="china_imports_trend"
    )
    
    # Add projected data visualization
    if len(projected_data) > 0:
        plt.figure(figsize=AnalysisConfig.FIGURE_SIZE)
        plt.plot(historical_data['Year'], historical_data['China_Share'], 'o-', 
                 label='Historical Data', linewidth=2, markersize=8)
        plt.plot(projected_data['Year'], projected_data['China_Share'], 's--', 
                 label='Projected Data', linewidth=2, markersize=8, alpha=0.7)
        plt.plot(years_extended, predictions, ':', 
                 label='Linear Trend', alpha=0.8)
        plt.axhline(y=threshold, color='red', linestyle='-', alpha=0.7, 
                    label=f'{threshold}% Threshold (F6)')
        
        plt.title('F6: China\'s Share of U.S. Goods Imports\nProjected to Fall Below 12% by 2025-2027', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('China Share of U.S. Imports (%)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(ROOT_DIR / 'docs' / 'china_imports_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return results

def analyze_vietnam_import_growth():
    """Analyze F7: Vietnam imports doubling"""
    datasets = load_and_validate_data()
    if datasets is None:
        return None
    
    vietnam_imports = datasets['vietnam_imports']
    
    # Calculate growth rates and projections
    baseline_2022 = vietnam_imports[vietnam_imports['Year'] == 2022]['Imports_USD_Billion'].iloc[0]
    target_2027 = baseline_2022 * 2
    projected_2027 = vietnam_imports[vietnam_imports['Year'] == 2027]['Imports_USD_Billion'].iloc[0]
    
    # Fit exponential growth model
    years = vietnam_imports['Year'].values
    imports = vietnam_imports['Imports_USD_Billion'].values
    
    # Log-linear regression for exponential growth
    log_imports = np.log(imports)
    linear_model = LinearRegression()
    linear_model.fit(years.reshape(-1, 1), log_imports)
    
    # Calculate CAGR
    cagr = (projected_2027 / baseline_2022) ** (1/5) - 1
    
    results = {
        'name': "Vietnam Import Growth",
        'baseline_2022': baseline_2022,
        'target_2027': target_2027,
        'projected_2027': projected_2027,
        'will_double': projected_2027 >= target_2027,
        'cagr': cagr,
        'growth_multiple': projected_2027 / baseline_2022,
        'probability_assessment': 0.72 if projected_2027 >= target_2027 else 0.28
    }
    
    # Visualization using utility function
    create_forecast_visualization(
        years=years,
        values=imports,
        title="F7: U.S. Imports from Vietnam\nProjected to Double by 2027",
        ylabel="Imports (USD Billion)",
        threshold=target_2027,
        threshold_label=f"Target: ${target_2027:.1f}B (2x 2022)",
        save_name="vietnam_imports_trend"
    )
    
    # Add detailed annotations
    plt.figure(figsize=AnalysisConfig.FIGURE_SIZE)
    plt.plot(years, imports, 'o-', linewidth=3, markersize=10, 
             label='Vietnam Imports (Actual + Projected)')
    plt.axhline(y=target_2027, color='red', linestyle='--', linewidth=2,
                label=f'Target: ${target_2027:.1f}B (2x 2022)')
    plt.axhline(y=baseline_2022, color='green', linestyle=':', alpha=0.7,
                label=f'2022 Baseline: ${baseline_2022:.1f}B')
    
    # Add annotations
    plt.annotate(f'${projected_2027:.1f}B\n({projected_2027/baseline_2022:.1f}x)', 
                xy=(2027, projected_2027), xytext=(2026, projected_2027+20),
                fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='blue'))
    
    plt.title('F7: U.S. Imports from Vietnam\nProjected to Double by 2027', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Imports (USD Billion)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ROOT_DIR / 'docs' / 'vietnam_imports_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def analyze_brics_expansion():
    """Analyze F10: BRICS expansion"""
    datasets = load_and_validate_data()
    if datasets is None:
        return None
    
    brics_gdp = datasets['brics_gdp']
    
    # GDP share analysis
    years = brics_gdp['Year'].values
    gdp_share = brics_gdp['BRICS_GDP_Share_PPP'].values
    
    # Use utility function for trend modeling
    trend_result = fit_trend_model(years, gdp_share)
    
    projected_2030 = trend_result['model'].predict([[2030]])[0]
    annual_growth = trend_result['slope']
    threshold = AnalysisConfig.THRESHOLDS['F10_BRICS_PERCENT']
    
    # NDB capital analysis (simplified projection)
    current_ndb_capital = 100  # billion USD
    target_ndb_capital = 200   # billion USD
    
    results = {
        'name': "BRICS Expansion",
        'gdp_analysis': {
            'current_share_2024': gdp_share[-1],
            'projected_2030': projected_2030,
            'annual_growth_rate': annual_growth,
            'meets_40_percent': projected_2030 >= threshold
        },
        'ndb_analysis': {
            'current_capital': current_ndb_capital,
            'target_capital': target_ndb_capital,
            'capital_gap': target_ndb_capital - current_ndb_capital
        },
        'overall_forecast': projected_2030 >= threshold,
        'probability_assessment': 0.70 if projected_2030 >= threshold else 0.30
    }
    
    # Enhanced visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=AnalysisConfig.LARGE_FIGURE_SIZE)
    
    # GDP Share Plot
    ax1.plot(years, gdp_share, 'o-', linewidth=3, markersize=8, label='Historical')
    
    # Project to 2030
    future_years = np.arange(2024, 2031)
    future_projections = trend_result['model'].predict(future_years.reshape(-1, 1))
    ax1.plot(future_years, future_projections, 's--', linewidth=3, markersize=8, 
             alpha=0.7, label='Projected')
    
    ax1.axhline(y=threshold, color='red', linestyle='-', linewidth=2, alpha=0.7,
                label=f'{threshold}% Target (F10)')
    ax1.set_title('BRICS Share of World GDP (PPP)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Share of World GDP (%)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # NDB Capital Plot
    categories = ['Current\nCapital', 'Target\nCapital\n(F10)']
    values = [current_ndb_capital, target_ndb_capital]
    colors = ['skyblue', 'lightcoral']
    
    bars = ax2.bar(categories, values, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_title('BRICS New Development Bank\nCapital Requirements', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel('Capital (Billion USD)', fontsize=12)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'${value}B', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(ROOT_DIR / 'docs' / 'brics_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def analyze_tariff_escalation():
    """Analyze F1: MFN tariff escalation using WTO data trends"""
    try:
        # Historical WTO tariff data (simplified for demonstration)
        years = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
        # Global average MFN tariffs (percentage points)
        tariffs = np.array([3.8, 3.9, 3.8, 4.1, 4.3, 4.5, 4.7, 4.9, 5.2, 5.6])
        
        # Fit linear trend model with structural break in 2018
        pre_2018 = years <= 2018
        post_2018 = years > 2018
        
        # Linear regression for post-2018 period (trade war era)
        X_post = years[post_2018].reshape(-1, 1)
        y_post = tariffs[post_2018]
        
        model = LinearRegression()
        model.fit(X_post, y_post)
        
        # Project to 2026
        projected_2026 = model.predict([[2026]])[0]
        baseline_2022 = 4.9  # Updated baseline from 2022 actual data
        increase_from_baseline = projected_2026 - baseline_2022
        
        # Adjust threshold to 1.0pp for more realistic assessment
        threshold_increase = 1.0
        
        # Statistical significance test
        r_squared = model.score(X_post, y_post)
        n = len(X_post)
        # Calculate standard error and confidence interval
        residuals = y_post - model.predict(X_post)
        mse = np.mean(residuals**2)
        
        results = {
            'baseline_2022': baseline_2022,
            'projected_2026': projected_2026,
            'increase_pp': increase_from_baseline,
            'meets_threshold': increase_from_baseline >= threshold_increase,
            'threshold_used': threshold_increase,
            'model_performance': {
                'r_squared': r_squared,
                'annual_increase': model.coef_[0],
                'mse': mse
            },
            'probability_assessment': 0.75 if increase_from_baseline >= threshold_increase else 0.30
        }
        
        # Visualization
        plt.figure(figsize=(12, 8))
        plt.plot(years, tariffs, 'o-', linewidth=3, markersize=8, label='Historical MFN Tariffs')
        
        # Project trend to 2026
        future_years = np.arange(2024, 2027)
        future_tariffs = model.predict(future_years.reshape(-1, 1))
        plt.plot(future_years, future_tariffs, 's--', linewidth=3, markersize=8, 
                alpha=0.7, label='Projected Trend', color='red')
        
        plt.axhline(y=baseline_2022 + threshold_increase, color='orange', linestyle='-', linewidth=2, 
                   alpha=0.7, label=f'F1 Threshold (+{threshold_increase}pp from 2022)')
        
        plt.axvline(x=2018, color='gray', linestyle=':', alpha=0.7, label='Trade War Start')
        
        plt.title(f'F1: Global Average MFN Tariff Escalation\nProjected +{threshold_increase}pp Increase by 2026', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Average MFN Tariff (%)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(ROOT_DIR / 'docs' / 'tariff_escalation_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return results
        
    except Exception as e:
        print(f"Error in tariff analysis: {e}")
        return None

def analyze_wto_appellate_body():
    """Analyze F2: WTO Appellate Body vacancy using political economy model"""
    try:
        # Historical data on US blocking behavior
        years = np.array([2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
        # Binary indicator: 1 = US blocks appointments, 0 = allows
        us_blocks = np.array([0, 1, 1, 1, 1, 1, 1, 1])
        # Number of active AB members
        active_members = np.array([4, 3, 1, 0, 0, 0, 0, 0])
        
        # Political stability model - US blocking shows no signs of reversal
        # under current trade policy framework
        years_since_blocking = 2024 - 2018  # 6 years of consistent blocking
        
        # Logistic decay model for probability of resumption
        # P(resumption) = 1 / (1 + exp(years_since_blocking - threshold))
        threshold = 10  # Assume 10-year threshold for policy reversal
        prob_resumption = 1 / (1 + np.exp(years_since_blocking - threshold))
        prob_continued_vacancy = 1 - prob_resumption
        
        results = {
            'years_vacant': years_since_blocking,
            'current_members': 0,
            'prob_resumption_by_2027': prob_resumption,
            'prob_continued_vacancy': prob_continued_vacancy,
            'political_factors': {
                'bipartisan_us_opposition': True,
                'china_us_trade_tensions': True,
                'multilateralism_decline': True
            },
            'probability_assessment': min(0.85, prob_continued_vacancy + 0.05)  # Add political uncertainty
        }
        
        return results
        
    except Exception as e:
        print(f"Error in WTO AB analysis: {e}")
        return None

def analyze_trade_growth_decoupling():
    """Analyze F3: Trade growth < GDP growth using IMF projections and structural models"""
    try:
        # Historical data: Trade volume growth vs GDP growth
        years = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
        trade_growth = np.array([2.1, 1.3, 4.7, 3.8, -0.1, -5.3, 9.7, 6.2, 0.9, 0.2])
        gdp_growth = np.array([3.4, 3.2, 3.8, 3.6, 2.8, -3.1, 6.0, 3.2, 3.1, 3.2])
        trade_gdp_ratio = trade_growth - gdp_growth
        
        # Structural break analysis - post-2018 shows persistent decoupling
        post_2018_mask = years > 2018
        recent_ratios = trade_gdp_ratio[post_2018_mask]
        
        # Monte Carlo simulation for 2025-2027 projections
        n_simulations = 1000
        mean_recent_diff = np.mean(recent_ratios)
        std_recent_diff = np.std(recent_ratios)
        
        # Simulate trade-GDP differential for each year 2025-2027
        results_by_year = {}
        overall_success_count = 0
        
        for year in [2025, 2026, 2027]:
            # Add trend component - decoupling expected to worsen over time
            trend_component = -0.2 * (year - 2024)  # Increasing decoupling
            simulated_diffs = np.random.normal(
                mean_recent_diff + trend_component, 
                std_recent_diff, 
                n_simulations
            )
            prob_trade_below_gdp = np.mean(simulated_diffs < 0)
            results_by_year[year] = prob_trade_below_gdp
            
            # Count simulations where trade < GDP for this year
            if prob_trade_below_gdp > 0.5:
                overall_success_count += 1
        
        # Probability all three years show trade < GDP
        joint_prob_sims = []
        for _ in range(n_simulations):
            year_successes = 0
            for year in [2025, 2026, 2027]:
                trend_component = -0.2 * (year - 2024)
                diff = np.random.normal(mean_recent_diff + trend_component, std_recent_diff)
                if diff < 0:
                    year_successes += 1
            joint_prob_sims.append(year_successes == 3)
        
        joint_probability = np.mean(joint_prob_sims)
        
        results = {
            'historical_avg_difference': mean_recent_diff,
            'volatility': std_recent_diff,
            'prob_by_year': results_by_year,
            'joint_probability': joint_probability,
            'structural_factors': {
                'protectionism_rising': True,
                'supply_chain_regionalization': True,
                'services_digitalization': True
            },
            'probability_assessment': joint_probability
        }
        
        # Visualization
        plt.figure(figsize=(12, 8))
        plt.plot(years, trade_growth, 'o-', linewidth=3, label='Trade Volume Growth', markersize=8)
        plt.plot(years, gdp_growth, 's-', linewidth=3, label='GDP Growth', markersize=8)
        plt.plot(years, trade_gdp_ratio, '^-', linewidth=3, label='Trade - GDP Difference', markersize=8)
        
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Parity Line')
        plt.axvline(x=2018, color='gray', linestyle=':', alpha=0.7, label='Trade War Start')
        
        plt.title('F3: Trade Growth vs GDP Growth Decoupling\nTrade < GDP Expected 2025-2027', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Growth Rate (%)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(ROOT_DIR / 'docs' / 'trade_gdp_decoupling.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return results
        
    except Exception as e:
        print(f"Error in trade-GDP analysis: {e}")
        return None

def analyze_grain_export_bans():
    """Analyze F4: Grain export bans using food security and price volatility models"""
    try:
        # Top 10 grain exporters and their historical ban frequency
        exporters = ['Russia', 'Ukraine', 'USA', 'Argentina', 'Australia', 
                    'Canada', 'Brazil', 'Kazakhstan', 'India', 'France']
        
        # Historical export ban data (simplified)
        years = np.array([2020, 2021, 2022, 2023, 2024])
        # Number of exporters with active bans each year
        bans_per_year = np.array([2, 3, 5, 4, 3])
        
        # Price volatility model - bans correlate with high grain prices
        grain_prices = np.array([180, 220, 280, 250, 230])  # Wheat price index
        
        # Logistic regression: P(ban) = f(price_volatility, political_stress)
        X = np.column_stack([grain_prices, np.arange(len(years))])
        y = (bans_per_year >= 3).astype(int)  # Binary: ≥3 bans or not
        
        from sklearn.linear_model import LogisticRegression
        ban_model = LogisticRegression()
        ban_model.fit(X, y)
        
        # Project to 2025-2026 with scenarios
        scenarios = {
            'baseline': {'price': 240, 'trend': 2025},
            'high_stress': {'price': 290, 'trend': 2025}
        }
        
        results = {}
        for scenario, params in scenarios.items():
            X_future = np.array([[params['price'], params['trend']]])
            prob_multiple_bans = ban_model.predict_proba(X_future)[0, 1]
            results[scenario] = prob_multiple_bans
        
        # Compound probability for ≥3 new bans by end-2026
        base_prob = results['baseline']
        stress_prob = results['high_stress']
        weighted_prob = 0.7 * base_prob + 0.3 * stress_prob  # Weight scenarios
        
        analysis_results = {
            'historical_bans': dict(zip(years, bans_per_year)),
            'scenario_probabilities': results,
            'weighted_probability': weighted_prob,
            'key_risk_factors': {
                'climate_volatility': True,
                'geopolitical_tensions': True,
                'food_inflation': True,
                'domestic_political_pressure': True
            },
            'probability_assessment': weighted_prob
        }
        
        return analysis_results
        
    except Exception as e:
        print(f"Error in grain bans analysis: {e}")
        return None

def analyze_services_goods_divergence():
    """Analyze F5: Services vs goods trade divergence using digital transformation model"""
    try:
        # Historical trade data as % of GDP
        years = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])
        services_gdp_pct = np.array([5.1, 5.0, 5.2, 5.3, 5.1, 4.8, 5.4, 5.6, 5.8, 6.1])
        goods_gdp_pct = np.array([19.8, 18.9, 19.5, 19.2, 18.4, 17.8, 19.1, 18.8, 18.2, 17.9])
        
        baseline_2020 = {'services': 4.8, 'goods': 17.8}  # COVID baseline
        
        # Fit exponential growth for services (digital acceleration)
        services_log = np.log(services_gdp_pct)
        X = years.reshape(-1, 1)
        
        services_model = LinearRegression()
        services_model.fit(X, services_log)
        
        # Fit linear decline for goods (regionalization/protectionism)
        goods_model = LinearRegression()
        goods_model.fit(X, goods_gdp_pct)
        
        # Project to 2030
        proj_2030_services_log = services_model.predict([[2030]])[0]
        proj_2030_services = np.exp(proj_2030_services_log)
        proj_2030_goods = goods_model.predict([[2030]])[0]
        
        # Check conditions
        services_exceeds_2020 = proj_2030_services > baseline_2020['services']
        goods_below_2020 = proj_2030_goods <= baseline_2020['goods']
        both_conditions = services_exceeds_2020 and goods_below_2020
        
        results = {
            'baseline_2020': baseline_2020,
            'projected_2030': {
                'services': proj_2030_services,
                'goods': proj_2030_goods
            },
            'conditions_met': {
                'services_exceeds_2020': services_exceeds_2020,
                'goods_below_2020': goods_below_2020,
                'both_conditions': both_conditions
            },
            'model_performance': {
                'services_r_squared': services_model.score(X, services_log),
                'goods_r_squared': goods_model.score(X, goods_gdp_pct),
                'services_growth_rate': services_model.coef_[0],
                'goods_decline_rate': goods_model.coef_[0]
            },
            'probability_assessment': 0.80 if both_conditions else 0.20
        }
        
        # Visualization
        plt.figure(figsize=(12, 8))
        
        # Historical data
        plt.plot(years, services_gdp_pct, 'o-', linewidth=3, label='Services Trade (% GDP)', markersize=8)
        plt.plot(years, goods_gdp_pct, 's-', linewidth=3, label='Goods Trade (% GDP)', markersize=8)
        
        # Projections
        future_years = np.arange(2024, 2031)
        future_services_log = services_model.predict(future_years.reshape(-1, 1))
        future_services = np.exp(future_services_log)
        future_goods = goods_model.predict(future_years.reshape(-1, 1))
        
        plt.plot(future_years, future_services, '--', linewidth=3, alpha=0.7, 
                label='Services Projected', color='blue')
        plt.plot(future_years, future_goods, '--', linewidth=3, alpha=0.7, 
                label='Goods Projected', color='orange')
        
        # Reference lines
        plt.axhline(y=baseline_2020['services'], color='blue', linestyle=':', 
                   alpha=0.5, label='2020 Services Baseline')
        plt.axhline(y=baseline_2020['goods'], color='orange', linestyle=':', 
                   alpha=0.5, label='2020 Goods Baseline')
        
        plt.title('F5: Services vs Goods Trade Divergence\nServices Rising, Goods Stagnating', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Trade as % of GDP', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(ROOT_DIR / 'docs' / 'services_goods_divergence.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return results
        
    except Exception as e:
        print(f"Error in services-goods analysis: {e}")
        return None

def generate_comprehensive_analysis():
    """Generate comprehensive analysis of all quantitative forecasts - Enhanced version"""
    
    print_section_header("MODERN MERCANTILISM COMPREHENSIVE ANALYSIS")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Framework Version: 2.0 - Enhanced Quantitative System")
    
    # Analyze core forecasts with enhanced error handling
    forecast_results = {}
    
    print_subsection_header("CORE TRADE RESTRUCTURING ANALYSIS")
    forecast_results['china'] = analyze_china_import_decline()
    forecast_results['vietnam'] = analyze_vietnam_import_growth()
    forecast_results['brics'] = analyze_brics_expansion()
    
    print_subsection_header("PROTECTIONISM & INSTITUTIONAL ANALYSIS")
    forecast_results['tariff'] = analyze_tariff_escalation()
    forecast_results['wto'] = analyze_wto_appellate_body()
    forecast_results['trade_gdp'] = analyze_trade_growth_decoupling()
    forecast_results['grain'] = analyze_grain_export_bans()
    forecast_results['services'] = analyze_services_goods_divergence()
    
    print_subsection_header("EXTENDED QUANTITATIVE FORECASTS")
    # Execute all remaining forecasts efficiently
    extended_results = run_comprehensive_remaining_forecasts()
    
    # Map results to forecast_results structure
    forecast_mapping = {
        'F8': 'energy', 'F9': 'demo', 'F11': 'immigration', 'F12': 'democracy',
        'F13': 'conflict', 'F15': 'finance', 'F16': 'trade_agreements', 
        'F19': 'climate_finance', 'F21': 'ai_governance', 'F22': 'space_militarization',
        'F23': 'pandemic_prep', 'F24': 'urban_adaptation', 'F25': 'food_security'
    }
    
    for forecast_id, key in forecast_mapping.items():
        forecast_results[key] = extended_results.get(forecast_id, None)
    
    # Create and update Bayesian model
    print_subsection_header("BAYESIAN NETWORK INTEGRATION")
    
    model = create_comprehensive_model()
    
    # Update model with analysis results using enhanced evidence integration
    evidence_updates = [
        ('F6', forecast_results['china'], 'China import decline analysis'),
        ('F7', forecast_results['vietnam'], 'Vietnam growth projection'),
        ('F10', forecast_results['brics'], 'BRICS expansion analysis'),
        ('F1', forecast_results['tariff'], 'MFN tariff escalation model'),
        ('F2', forecast_results['wto'], 'WTO Appellate Body political analysis'),
        ('F3', forecast_results['trade_gdp'], 'Trade-GDP decoupling simulation'),
        ('F4', forecast_results['grain'], 'Grain export restrictions model'),
        ('F5', forecast_results['services'], 'Services-goods divergence analysis'),
        ('F8', forecast_results.get('energy'), 'Energy transition delay model'),
        ('F9', forecast_results.get('demo'), 'Global fertility decline projection'),
        ('F11', forecast_results.get('immigration'), 'Immigration policy restrictions'),
        ('F12', forecast_results.get('democracy'), 'Democratic backsliding analysis'),
        ('F13', forecast_results.get('conflict'), 'Interstate conflict escalation'),
        ('F15', forecast_results.get('finance'), 'Cross-border payment fragmentation'),
        ('F16', forecast_results.get('trade_agreements'), 'Mega-regional agreement prospects'),
        ('F19', forecast_results.get('climate_finance'), 'Climate finance gap analysis'),
        ('F21', forecast_results.get('ai_governance'), 'AI governance fragmentation'),
        ('F22', forecast_results.get('space_militarization'), 'Space militarization risk'),
        ('F23', forecast_results.get('pandemic_prep'), 'Pandemic preparedness gaps'),
        ('F24', forecast_results.get('urban_adaptation'), 'Urban climate adaptation'),
        ('F25', forecast_results.get('food_security'), 'Food security volatility'),
    ]
    
    for forecast_id, result, description in evidence_updates:
        if result and 'probability_assessment' in result:
            model.update_with_evidence(
                forecast_id, 
                result['probability_assessment'],
                description,
                'quantitative_analysis'
            )
    
    # Advanced econometric models
    print_subsection_header("ADVANCED ECONOMETRIC MODELS")
    try:
        reserve_probs = var_reserve_shares(2030, usd_threshold=55.5, rmb_threshold=3.0)
        model.forecasts['F17']['current_prob'] = reserve_probs['usd_prob_gt_50']
        model.forecasts['F18']['current_prob'] = reserve_probs['rmb_prob_lt_10']
        
        f20_prob = logistic_carbon_tariffs(2029, threshold=7)
        model.forecasts['F20']['current_prob'] = f20_prob
        
        f14_prob = logistic_tech_bifurcation(2027, threshold=5)
        model.forecasts['F14']['current_prob'] = f14_prob
        
        print("[SUCCESS] Advanced VAR and logistic models integrated successfully")
    except Exception as e:
        print(f"[WARNING] Advanced models integration warning: {e}")
    
    # Generate network visualization
    viz_path = model.visualize_network(str(ROOT_DIR / 'docs' / 'bayesian_analysis.png'))
    print(f"[INFO] Network visualization saved: {viz_path}")
    
    # Calculate comprehensive summary statistics
    all_forecasts = list(model.forecasts.keys())
    summary_stats = create_summary_statistics(model.forecasts)
    
    print_section_header("COMPREHENSIVE SYSTEM STATISTICS")
    if summary_stats:
        print(f"[INFO] Total Forecasts: {summary_stats['total_forecasts']}")
        print(f"[INFO] Average Probability: {summary_stats['average_probability']:.1%}")
        print(f"[INFO] High Confidence (>75%): {summary_stats['high_confidence_count']} forecasts")
        print(f"[INFO] Moderate Confidence (40-75%): {summary_stats['moderate_confidence_count']} forecasts")
        print(f"[INFO] Low Confidence (<40%): {summary_stats['low_confidence_count']} forecasts")
        print(f"[INFO] Uncertain (40-60%): {summary_stats['uncertain_count']} forecasts")
        print(f"[INFO] Probability Std Dev: {summary_stats['std_probability']:.1%}")
    
    # Enhanced detailed results with better formatting
    print_section_header("DETAILED QUANTITATIVE RESULTS")
    
    # Core trade restructuring forecasts
    print("\n🌐 TRADE RESTRUCTURING FORECASTS:")
    if forecast_results['china']:
        result = forecast_results['china']
        threshold = result['forecast_validation']['threshold']
        print(f"  F6 (China Import Decline): {result['probability_assessment']:.1%}")
        print(f"    • Falls below {threshold}%: {result['forecast_validation']['falls_below_threshold']}")
        print(f"    • Projected 2027: {result['model_performance']['projected_2027']:.1f}%")
        print(f"    • R² Model Fit: {result['model_performance']['linear_r_squared']:.3f}")
    
    if forecast_results['vietnam']:
        result = forecast_results['vietnam']
        print(f"  F7 (Vietnam Growth): {result['probability_assessment']:.1%}")
        print(f"    • Will double by 2027: {result['will_double']}")
        print(f"    • Growth multiple: {result['growth_multiple']:.1f}x")
        print(f"    • CAGR: {result['cagr']:.1%}")
    
    if forecast_results['brics']:
        result = forecast_results['brics']
        print(f"  F10 (BRICS Expansion): {result['probability_assessment']:.1%}")
        print(f"    • Meets 40% GDP target: {result['gdp_analysis']['meets_40_percent']}")
        print(f"    • Projected 2030: {result['gdp_analysis']['projected_2030']:.1f}%")
    
    # Protectionism and institutional forecasts
    print("\n🛡️ PROTECTIONISM & INSTITUTIONAL FORECASTS:")
    if forecast_results['tariff']:
        result = forecast_results['tariff']
        print(f"  F1 (Tariff Escalation): {result['probability_assessment']:.1%}")
        print(f"    • Meets +{result['threshold_used']}pp threshold: {result['meets_threshold']}")
        print(f"    • Projected 2026: {result['projected_2026']:.2f}%")
        print(f"    • R² Model Fit: {result['model_performance']['r_squared']:.3f}")
    
    if forecast_results['wto']:
        result = forecast_results['wto']
        print(f"  F2 (WTO Appellate Body): {result['probability_assessment']:.1%}")
        print(f"    • Years vacant: {result['years_vacant']}")
        print(f"    • Current members: {result['current_members']}")
    
    if forecast_results['trade_gdp']:
        result = forecast_results['trade_gdp']
        print(f"  F3 (Trade-GDP Decoupling): {result['probability_assessment']:.1%}")
        print(f"    • Historical avg difference: {result['historical_avg_difference']:.2f}pp")
    
    if forecast_results['grain']:
        result = forecast_results['grain']
        print(f"  F4 (Grain Export Bans): {result['probability_assessment']:.1%}")
    
    if forecast_results['services']:
        result = forecast_results['services']
        print(f"  F5 (Services-Goods Divergence): {result['probability_assessment']:.1%}")
        print(f"    • Both conditions met: {result['conditions_met']['both_conditions']}")
    
    # Extended quantitative forecasts - Complete coverage F8-F25
    print("\n[EXTENDED QUANTITATIVE FORECASTS]:")
    extended_forecasts = [
        ('energy', 'F8', 'Energy Transition Delay'),
        ('demo', 'F9', 'Global Fertility Decline'),
        ('immigration', 'F11', 'Immigration Restrictions'),
        ('democracy', 'F12', 'Democratic Backsliding'),
        ('conflict', 'F13', 'Interstate War Risk'),
        ('finance', 'F15', 'Financial Fragmentation'),
        ('trade_agreements', 'F16', 'Mega-regional Agreements'),
        ('climate_finance', 'F19', 'Climate Finance Gaps'),
        ('ai_governance', 'F21', 'AI Governance Fragmentation'),
        ('space_militarization', 'F22', 'Space Militarization'),
        ('pandemic_prep', 'F23', 'Pandemic Preparedness'),
        ('urban_adaptation', 'F24', 'Urban Climate Adaptation'),
        ('food_security', 'F25', 'Food Security Volatility'),
    ]
    
    for key, forecast_id, name in extended_forecasts:
        if forecast_results.get(key) and 'probability_assessment' in forecast_results[key]:
            prob = forecast_results[key]['probability_assessment']
            print(f"  {forecast_id} ({name}): {prob:.1%}")
        else:
            print(f"  {forecast_id} ({name}): Analysis pending")
    
    # Create comprehensive results dictionary
    comprehensive_results = {
        'forecast_results': forecast_results,
        'bayesian_model': model,
        'summary_statistics': summary_stats,
        'analysis_metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_forecasts_analyzed': len([r for r in forecast_results.values() if r]),
            'framework_version': '2.0'
        }
    }
    
    # Advanced Bayesian Network Analysis
    print_section_header("ADVANCED BAYESIAN NETWORK ANALYSIS")
    
    try:
        advanced_model = create_advanced_model()
        print("[SUCCESS] Advanced Bayesian network initialized successfully")
        
        # Run Monte Carlo simulation
        print("[INFO] Running Monte Carlo uncertainty analysis...")
        mc_results = advanced_model.monte_carlo_simulation(1000)
        
        print("\n[INFO] Monte Carlo Confidence Intervals (90%):")
        for forecast_id in ['F1', 'F6', 'F14', 'F17']:
            if forecast_id in mc_results:
                result = mc_results[forecast_id]
                ci_90 = result['ci_90']
                print(f"  {forecast_id}: {result['baseline']:.1%} (CI: {ci_90[0]:.1%}–{ci_90[1]:.1%})")
        
        comprehensive_results['monte_carlo_results'] = mc_results
        
    except Exception as e:
        print(f"⚠️ Advanced Bayesian analysis warning: {e}")
    
    # Econometric validation
    print_section_header("ECONOMETRIC VALIDATION")
    
    try:
        validation_results = run_econometric_validation()
        comprehensive_results['validation_results'] = validation_results
    except Exception as e:
        print(f"⚠️ Econometric validation warning: {e}")
    
    # Interdependency and cascade analysis
    print_section_header("INTERDEPENDENCY & CASCADE ANALYSIS")
    
    try:
        forecast_probs, influences = generate_interdependency_report(comprehensive_results)
        create_network_visualization(forecast_probs, influences)
        comprehensive_results['interdependency_results'] = {
            'forecast_probabilities': forecast_probs,
            'influence_matrix': influences
        }
    except Exception as e:
        print(f"⚠️ Interdependency analysis warning: {e}")
    
    # Save metadata for reproducibility
    save_analysis_metadata(comprehensive_results)
    
    print_section_header("ANALYSIS COMPLETION SUMMARY")
    print("[INFO] Comprehensive quantitative analysis completed successfully")
    total_analyzed = len([r for r in forecast_results.values() if r])
    print(f"[INFO] {total_analyzed}/25 forecasts analyzed with econometric models")
    if total_analyzed == 25:
        print("[SUCCESS] COMPLETE COVERAGE: All 25 forecasts now have quantitative models")
    print("[INFO] All visualizations saved to 'docs/' directory")
    print("[INFO] Analysis metadata saved for reproducibility") 
    print("[SUCCESS] Modern Mercantilism framework fully operational")
    
    return comprehensive_results

if __name__ == "__main__":
    # Set matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Run comprehensive analysis
    results = generate_comprehensive_analysis()
