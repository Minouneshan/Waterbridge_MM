import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from bayesian_model import create_comprehensive_model
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Project root directory two levels above this script
ROOT_DIR = Path(__file__).resolve().parents[1]

def load_and_validate_data():
    """Load and validate all data files"""
    try:
        # Determine project root (two levels up) and data directory for reproducibility
        data_dir = ROOT_DIR / 'data'

        # Load all data files
        china_imports = pd.read_csv(data_dir / 'us_imports.csv')
        vietnam_imports = pd.read_csv(data_dir / 'vietnam_us_imports.csv')
        brics_gdp = pd.read_csv(data_dir / 'brics_gdp_share.csv')
        brics_data = pd.read_csv(data_dir / 'brics_data.csv')
        
        print("✓ All data files loaded successfully")
        return china_imports, vietnam_imports, brics_gdp, brics_data
        
    except FileNotFoundError as e:
        print(f"❌ Error loading data files: {e}")
        print("Please ensure all CSV files are in the 'data' directory.")
        return None, None, None, None

def analyze_china_import_decline():
    """Analyze F6: China's declining share of U.S. imports"""
    china_imports, _, _, _ = load_and_validate_data()
    if china_imports is None:
        return None
    
    # Separate historical vs projected data
    historical_data = china_imports[china_imports['Year'] <= 2024]
    projected_data = china_imports[china_imports['Year'] > 2024]
    
    # Fit multiple regression models
    X_hist = historical_data['Year'].values.reshape(-1, 1)
    y_hist = historical_data['China_Share'].values
    
    # Linear model
    linear_model = LinearRegression()
    linear_model.fit(X_hist, y_hist)
    
    # Polynomial model
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X_hist)
    poly_model = LinearRegression()
    poly_model.fit(X_poly, y_hist)
    
    # Generate predictions
    years_extended = np.arange(2015, 2028).reshape(-1, 1)
    linear_pred = linear_model.predict(years_extended)
    poly_pred = poly_model.predict(poly_features.transform(years_extended))
    
    # Calculate statistics
    r_squared_linear = linear_model.score(X_hist, y_hist)
    
    # Check forecast validation
    will_fall_below_12 = any(china_imports[china_imports['Year'].isin([2025, 2026, 2027])]['China_Share'] < 12)
    
    results = {
        'model_performance': {
            'linear_r_squared': r_squared_linear,
            'slope_per_year': linear_model.coef_[0],
            'projected_2027': linear_pred[-1]
        },
        'forecast_validation': {
            'falls_below_12_percent': will_fall_below_12,
            'probability_assessment': 0.75
        }
    }
    
    # Visualization
    plt.figure(figsize=(12, 8))
    plt.plot(historical_data['Year'], historical_data['China_Share'], 'o-', 
             label='Historical Data', linewidth=2, markersize=8)
    plt.plot(projected_data['Year'], projected_data['China_Share'], 's--', 
             label='Projected Data', linewidth=2, markersize=8, alpha=0.7)
    plt.plot(years_extended.flatten(), linear_pred, ':', 
             label='Linear Trend', alpha=0.8)
    plt.axhline(y=12, color='red', linestyle='-', alpha=0.7, 
                label='12% Threshold (F6)')
    
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
    _, vietnam_imports, _, _ = load_and_validate_data()
    if vietnam_imports is None:
        return None
    
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
        'baseline_2022': baseline_2022,
        'target_2027': target_2027,
        'projected_2027': projected_2027,
        'will_double': projected_2027 >= target_2027,
        'cagr': cagr,
        'growth_multiple': projected_2027 / baseline_2022
    }
    
    # Visualization
    plt.figure(figsize=(12, 8))
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
    _, _, brics_gdp, brics_data = load_and_validate_data()
    if brics_gdp is None:
        return None
    
    # GDP share analysis
    years = brics_gdp['Year'].values
    gdp_share = brics_gdp['BRICS_GDP_Share_PPP'].values
    
    # Linear trend projection
    linear_model = LinearRegression()
    linear_model.fit(years.reshape(-1, 1), gdp_share)
    
    projected_2030 = linear_model.predict([[2030]])[0]
    annual_growth = linear_model.coef_[0]
    
    # NDB capital analysis (simplified projection)
    current_ndb_capital = 100  # billion USD
    target_ndb_capital = 200   # billion USD
    
    results = {
        'gdp_analysis': {
            'current_share_2024': gdp_share[-1],
            'projected_2030': projected_2030,
            'annual_growth_rate': annual_growth,
            'meets_40_percent': projected_2030 >= 40.0
        },
        'ndb_analysis': {
            'current_capital': current_ndb_capital,
            'target_capital': target_ndb_capital,
            'capital_gap': target_ndb_capital - current_ndb_capital
        },
        'overall_forecast': projected_2030 >= 40.0  # Simplified for GDP component
    }
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # GDP Share Plot
    ax1.plot(years, gdp_share, 'o-', linewidth=3, markersize=8, label='Historical')
    
    # Project to 2030
    future_years = np.arange(2024, 2031)
    future_projections = linear_model.predict(future_years.reshape(-1, 1))
    ax1.plot(future_years, future_projections, 's--', linewidth=3, markersize=8, 
             alpha=0.7, label='Projected')
    
    ax1.axhline(y=40, color='red', linestyle='-', linewidth=2, alpha=0.7,
                label='40% Target (F10)')
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

def generate_comprehensive_analysis():
    """Generate comprehensive analysis of all quantitative forecasts"""
    print("🔍 COMPREHENSIVE MODERN MERCANTILISM ANALYSIS")
    print("=" * 60)
    
    # Analyze individual forecasts
    china_results = analyze_china_import_decline()
    vietnam_results = analyze_vietnam_import_growth()
    brics_results = analyze_brics_expansion()
    
    # Create Bayesian model and run analysis
    print("\n📊 BAYESIAN NETWORK ANALYSIS")
    print("-" * 40)
    
    model = create_comprehensive_model()
    
    # Add evidence updates based on current analysis
    if china_results and china_results['forecast_validation']['falls_below_12_percent']:
        model.update_with_evidence('F6', 0.8, 'Statistical analysis confirms China import decline trend', 'academic_research')
    
    if vietnam_results and vietnam_results['will_double']:
        model.update_with_evidence('F7', 0.7, 'Projection shows Vietnam imports doubling by 2027', 'academic_research')
    
    if brics_results and brics_results['gdp_analysis']['meets_40_percent']:
        model.update_with_evidence('F10', 0.6, 'BRICS GDP share trending toward 40% by 2030', 'academic_research')
    
    # Generate network visualization
    viz_path = model.visualize_network(str(ROOT_DIR / 'docs' / 'bayesian_analysis.png'))
    print(f"✓ Network visualization saved: {viz_path}")
    
    # Generate summary statistics
    all_forecasts = list(model.forecasts.keys())
    avg_probability = np.mean([model.forecasts[f]['current_prob'] for f in all_forecasts])
    high_confidence = [f for f in all_forecasts if model.forecasts[f]['current_prob'] > 0.75]
    uncertain_forecasts = [f for f in all_forecasts if 0.4 <= model.forecasts[f]['current_prob'] <= 0.6]
    
    print(f"\n📈 SUMMARY STATISTICS")
    print("-" * 40)
    print(f"Total Forecasts: {len(all_forecasts)}")
    print(f"Average Probability: {avg_probability:.1%}")
    print(f"High Confidence (>75%): {len(high_confidence)} forecasts")
    print(f"Uncertain (40-60%): {len(uncertain_forecasts)} forecasts")
    
    # Detailed results
    print(f"\n🎯 DETAILED FORECAST RESULTS")
    print("-" * 40)
    
    if china_results:
        print(f"F6 (China Import Decline):")
        print(f"  • Falls below 12%: {china_results['forecast_validation']['falls_below_12_percent']}")
        print(f"  • Projected 2027: {china_results['model_performance']['projected_2027']:.1f}%")
        print(f"  • Annual decline: {abs(china_results['model_performance']['slope_per_year']):.1f} pp/year")
    
    if vietnam_results:
        print(f"\nF7 (Vietnam Import Growth):")
        print(f"  • Will double: {vietnam_results['will_double']}")
        print(f"  • Growth multiple: {vietnam_results['growth_multiple']:.1f}x")
        print(f"  • CAGR: {vietnam_results['cagr']:.1%}")
    
    if brics_results:
        print(f"\nF10 (BRICS Expansion):")
        print(f"  • Meets 40% GDP: {brics_results['gdp_analysis']['meets_40_percent']}")
        print(f"  • Projected 2030: {brics_results['gdp_analysis']['projected_2030']:.1f}%")
        print(f"  • Annual growth: {brics_results['gdp_analysis']['annual_growth_rate']:.2f} pp/year")
    
    print(f"\n✅ Analysis complete. Charts saved to '../docs/' directory")
    
    return {
        'china_analysis': china_results,
        'vietnam_analysis': vietnam_results,
        'brics_analysis': brics_results,
        'bayesian_model': model,
        'summary_stats': {
            'avg_probability': avg_probability,
            'high_confidence_count': len(high_confidence),
            'uncertain_count': len(uncertain_forecasts)
        }
    }

if __name__ == "__main__":
    # Set matplotlib style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Run comprehensive analysis
    results = generate_comprehensive_analysis()
