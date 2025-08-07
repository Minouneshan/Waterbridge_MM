"""
Utility functions for Modern Mercantilism Analysis
Consolidated common functionality for cleaner code
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Project root directory
ROOT_DIR = Path(__file__).resolve().parents[1]

class AnalysisConfig:
    """Configuration settings for all analyses"""
    
    # Chart styling
    FIGURE_SIZE = (12, 8)
    LARGE_FIGURE_SIZE = (16, 8)
    DPI = 300
    
    # Colors
    PRIMARY_COLOR = 'blue'
    SECONDARY_COLOR = 'red'
    ACCENT_COLOR = 'orange'
    
    # Forecast thresholds
    THRESHOLDS = {
        'F1_TARIFF_PP': 1.0,  # Percentage points increase
        'F6_CHINA_PERCENT': 12.0,  # China import share threshold
        'F10_BRICS_PERCENT': 40.0,  # BRICS GDP share threshold
        'F12_VDEM_SCORE': 0.4,  # V-Dem democracy score threshold
    }

def load_project_data():
    """Load and validate all project data files with error handling"""
    try:
        data_dir = ROOT_DIR / 'data'
        
        datasets = {
            'china_imports': pd.read_csv(data_dir / 'us_imports.csv'),
            'vietnam_imports': pd.read_csv(data_dir / 'vietnam_us_imports.csv'),
            'brics_gdp': pd.read_csv(data_dir / 'brics_gdp_share.csv')
        }
        
        print("[SUCCESS] All data files loaded successfully")
        return datasets
        
    except FileNotFoundError as e:
        print(f"[ERROR] Error loading data files: {e}")
        print("Please ensure all CSV files are in the 'data' directory.")
        return None

def create_forecast_visualization(years, values, title, ylabel, 
                                threshold=None, threshold_label=None,
                                projections=None, projection_label="Projected",
                                save_name=None):
    """
    Create standardized forecast visualization
    
    Args:
        years: Array of years (x-axis)
        values: Array of values (y-axis)
        title: Chart title
        ylabel: Y-axis label
        threshold: Optional horizontal threshold line
        threshold_label: Label for threshold line
        projections: Optional tuple (proj_years, proj_values) for future projections
        projection_label: Label for projection line
        save_name: Filename to save (without extension)
    """
    plt.figure(figsize=AnalysisConfig.FIGURE_SIZE)
    
    # Main data line
    plt.plot(years, values, 'o-', linewidth=3, markersize=8, 
             label='Historical Data', color=AnalysisConfig.PRIMARY_COLOR)
    
    # Projections if provided
    if projections is not None:
        proj_years, proj_values = projections
        plt.plot(proj_years, proj_values, 's--', linewidth=3, markersize=8,
                alpha=0.7, label=projection_label, color=AnalysisConfig.SECONDARY_COLOR)
    
    # Threshold line if provided
    if threshold is not None:
        plt.axhline(y=threshold, color=AnalysisConfig.ACCENT_COLOR, 
                   linestyle='-', linewidth=2, alpha=0.7, 
                   label=threshold_label or f'Threshold: {threshold}')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_name:
        plt.savefig(ROOT_DIR / 'docs' / f'{save_name}.png', 
                   dpi=AnalysisConfig.DPI, bbox_inches='tight')
    plt.close()

def fit_trend_model(years, values, model_type='linear'):
    """
    Fit trend model to time series data
    
    Args:
        years: Array of years
        values: Array of values
        model_type: 'linear' or 'polynomial'
    
    Returns:
        dict: Model results including fitted model, predictions, and statistics
    """
    X = years.reshape(-1, 1)
    
    if model_type == 'linear':
        model = LinearRegression()
        model.fit(X, values)
        predictions = model.predict(X)
        r_squared = model.score(X, values)
        
        return {
            'model': model,
            'predictions': predictions,
            'r_squared': r_squared,
            'slope': model.coef_[0] if hasattr(model, 'coef_') else None,
            'intercept': model.intercept_ if hasattr(model, 'intercept_') else None
        }
    
    elif model_type == 'polynomial':
        poly_features = PolynomialFeatures(degree=2)
        X_poly = poly_features.fit_transform(X)
        model = LinearRegression()
        model.fit(X_poly, values)
        predictions = model.predict(X_poly)
        r_squared = model.score(X_poly, values)
        
        return {
            'model': model,
            'poly_features': poly_features,
            'predictions': predictions,
            'r_squared': r_squared
        }

def calculate_monte_carlo_probability(base_value, std_dev, n_simulations=1000):
    """
    Run Monte Carlo simulation for probability estimation
    
    Args:
        base_value: Base scenario value
        std_dev: Standard deviation for uncertainty
        n_simulations: Number of Monte Carlo runs
    
    Returns:
        dict: Simulation results with confidence intervals
    """
    simulations = np.random.normal(base_value, std_dev, n_simulations)
    
    return {
        'mean': np.mean(simulations),
        'median': np.median(simulations),
        'std': np.std(simulations),
        'ci_90': np.percentile(simulations, [5, 95]),
        'ci_95': np.percentile(simulations, [2.5, 97.5]),
        'simulations': simulations
    }

def create_summary_statistics(forecast_results):
    """
    Generate summary statistics for all forecasts
    
    Args:
        forecast_results: Dictionary of forecast results
    
    Returns:
        dict: Summary statistics
    """
    probabilities = []
    
    for result in forecast_results.values():
        if result and 'probability_assessment' in result:
            probabilities.append(result['probability_assessment'])
    
    if not probabilities:
        return None
    
    probabilities = np.array(probabilities)
    
    return {
        'total_forecasts': len(probabilities),
        'average_probability': np.mean(probabilities),
        'median_probability': np.median(probabilities),
        'std_probability': np.std(probabilities),
        'high_confidence_count': np.sum(probabilities > 0.75),
        'moderate_confidence_count': np.sum((probabilities >= 0.40) & (probabilities <= 0.75)),
        'low_confidence_count': np.sum(probabilities < 0.40),
        'uncertain_count': np.sum((probabilities >= 0.40) & (probabilities <= 0.60))
    }

def print_section_header(title, width=60, char="="):
    """Print formatted section header"""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

def print_subsection_header(title, width=40, char="-"):
    """Print formatted subsection header"""
    print(f"\n{title}")
    print(char * width)

def format_forecast_result(forecast_id, result, details=None):
    """
    Format forecast result for consistent display
    
    Args:
        forecast_id: Forecast identifier (e.g., 'F1')
        result: Result dictionary
        details: Optional list of specific details to display
    """
    if not result:
        return f"\n{forecast_id}: Analysis failed"
    
    output = f"\n{forecast_id} ({result.get('name', 'Unnamed Forecast')}):"
    
    # Standard probability assessment
    if 'probability_assessment' in result:
        prob = result['probability_assessment']
        output += f"\n  • Probability: {prob:.1%}"
        
        # Confidence level indicator
        if prob > 0.75:
            output += " 🔴 HIGH"
        elif prob > 0.40:
            output += " 🟡 MODERATE"
        else:
            output += " 🟢 LOW"
    
    # Custom details if provided
    if details:
        for detail in details:
            if detail in result:
                value = result[detail]
                if isinstance(value, bool):
                    output += f"\n  • {detail.replace('_', ' ').title()}: {value}"
                elif isinstance(value, (int, float)):
                    output += f"\n  • {detail.replace('_', ' ').title()}: {value:.2f}"
                else:
                    output += f"\n  • {detail.replace('_', ' ').title()}: {value}"
    
    return output

def save_analysis_metadata(results, filename="analysis_metadata.json"):
    """Save analysis metadata for reproducibility"""
    import json
    from datetime import datetime
    
    metadata = {
        'analysis_timestamp': datetime.now().isoformat(),
        'total_forecasts_analyzed': len([r for r in results.values() if r is not None]),
        'configuration': {
            'thresholds': AnalysisConfig.THRESHOLDS,
            'figure_settings': {
                'size': AnalysisConfig.FIGURE_SIZE,
                'dpi': AnalysisConfig.DPI
            }
        }
    }
    
    with open(ROOT_DIR / 'docs' / filename, 'w') as f:
        json.dump(metadata, f, indent=2)
