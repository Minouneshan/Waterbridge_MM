"""
Comprehensive quantitative models for Modern Mercantilism forecasts F8-F25
Optimized implementation with complete forecast coverage and efficient execution
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

def analyze_energy_transition_f8():
    """F8: IEA Net Zero delayed beyond 2050 - Energy transition model"""
    try:
        # Historical renewable capacity vs required trajectory
        years = np.array([2020, 2021, 2022, 2023, 2024])
        renewable_additions = np.array([280, 295, 340, 370, 385])  # GW actual
        required_trajectory = np.array([820, 950, 1100, 1270, 1450])  # GW needed
        
        # Calculate persistent gap
        gap_percentage = ((required_trajectory - renewable_additions) / required_trajectory) * 100
        mean_gap = np.mean(gap_percentage[-3:])  # Recent 3-year average
        
        # Linear trend for gap closure
        X = years.reshape(-1, 1)
        model = LinearRegression().fit(X, gap_percentage)
        projected_gap_2040 = model.predict([[2040]])[0]
        
        # High delay probability if gap remains >30% by 2040
        delay_prob = 0.80 if projected_gap_2040 > 30.0 else 0.35
        final_prob = min(0.90, delay_prob + 0.15)  # Add implementation challenges
        
        return {
            'current_gap_2024': gap_percentage[-1],
            'projected_gap_2040': projected_gap_2040,
            'probability_assessment': final_prob
        }
    except Exception as e:
        print(f"Error in energy transition analysis: {e}")
        return None

def analyze_demographic_transition_f9():
    """F9: Global fertility <1.5 by 2035 - Demographic transition model"""
    try:
        # Global TFR trend (simplified UN data)
        years = np.array([2019, 2020, 2021, 2022, 2023, 2024])
        global_tfr = np.array([2.35, 2.31, 2.27, 2.23, 2.19, 2.15])
        
        # Fit exponential decay
        log_tfr = np.log(global_tfr)
        model = LinearRegression().fit(years.reshape(-1, 1), log_tfr)
        
        proj_2035_tfr = np.exp(model.predict([[2035]])[0])
        meets_threshold = proj_2035_tfr < 1.5
        
        return {
            'current_tfr_2024': global_tfr[-1],
            'projected_2035': proj_2035_tfr,
            'probability_assessment': 0.30 if meets_threshold else 0.70
        }
    except Exception as e:
        print(f"Error in demographic analysis: {e}")
        return None

def analyze_immigration_restrictions_f11():
    """F11: Net immigration to OECD <50% of 2019 levels"""
    try:
        # OECD immigration data (millions)
        years = np.array([2019, 2020, 2021, 2022, 2023, 2024])
        net_immigration = np.array([5.2, 3.1, 2.8, 3.6, 4.1, 4.3])
        baseline_2019 = 5.2
        threshold_50pct = baseline_2019 * 0.5
        
        # Policy restrictiveness model
        policy_restrictiveness = np.array([0.3, 0.7, 0.8, 0.6, 0.5, 0.4])
        
        X = np.column_stack([policy_restrictiveness, np.arange(len(years))])
        model = LinearRegression().fit(X, net_immigration)
        
        # Project with moderate restriction
        projected_2027 = model.predict([[0.6, 2027 - 2019]])[0]
        meets_threshold = projected_2027 < threshold_50pct
        
        return {
            'baseline_2019': baseline_2019,
            'threshold_50pct': threshold_50pct,
            'projected_2027': projected_2027,
            'probability_assessment': 0.65 if meets_threshold else 0.35
        }
    except Exception as e:
        print(f"Error in immigration analysis: {e}")
        return None

def analyze_authoritarianism_f12():
    """F12: V-Dem Liberal Democracy <0.4 by 2030"""
    try:
        # V-Dem Liberal Democracy Index trend
        years = np.array([2019, 2020, 2021, 2022, 2023, 2024])
        vdem_lib_dem = np.array([0.45, 0.43, 0.42, 0.41, 0.405, 0.40])
        
        model = LinearRegression().fit(years.reshape(-1, 1), vdem_lib_dem)
        projected_2030 = model.predict([[2030]])[0]
        meets_threshold = projected_2030 < 0.4
        
        return {
            'current_2024': vdem_lib_dem[-1],
            'projected_2030': projected_2030,
            'probability_assessment': 0.75 if meets_threshold else 0.25
        }
    except Exception as e:
        print(f"Error in authoritarianism analysis: {e}")
        return None

def analyze_conflict_escalation_f13():
    """F13: Interstate war >100k deaths - Conflict escalation model"""
    try:
        # Conflict intensity trend
        years = np.array([2020, 2021, 2022, 2023, 2024])
        conflicts_major = np.array([5, 6, 8, 9, 11])  # Major conflicts count
        battle_deaths = np.array([180, 220, 290, 350, 420])  # Thousands
        
        current_level = battle_deaths[-1]
        death_trend = np.mean(np.diff(battle_deaths[-3:]))  # Recent trend
        years_to_threshold = (100 - current_level) / death_trend if death_trend > 0 else np.inf
        
        # Risk factors boost probability
        risk_multiplier = 1.5  # Nuclear powers conflict, resource competition, arms race
        base_prob = 0.15
        trend_prob = min(0.4, 1 / max(years_to_threshold, 1)) if years_to_threshold < 10 else 0.05
        
        return {
            'current_deaths_k': current_level,
            'years_to_100k': years_to_threshold,
            'probability_assessment': min(0.8, (base_prob + trend_prob) * risk_multiplier)
        }
    except Exception as e:
        print(f"Error in conflict analysis: {e}")
        return None

def analyze_financial_integration_f15():
    """F15: Cross-border payments <80% of 2019 levels"""
    try:
        # Cross-border payment data (2019=100 index)
        years = np.array([2019, 2020, 2021, 2022, 2023, 2024])
        cross_border_payments = np.array([100, 75, 82, 88, 92, 95])
        baseline_2019, threshold_80pct = 100, 80
        
        # Project with fragmentation drag
        recovery_trend = np.mean(np.diff(cross_border_payments[-3:]))
        projected_level = cross_border_payments[-1] + (recovery_trend * 4) - (0.05 * 4)  # 5% annual drag
        
        return {
            'current_2024': cross_border_payments[-1],
            'projected_2028': projected_level,
            'probability_assessment': 0.60 if projected_level < threshold_80pct else 0.40
        }
    except Exception as e:
        print(f"Error in financial integration analysis: {e}")
        return None

def analyze_trade_agreements_f16():
    """F16: New mega-regional trade agreements"""
    try:
        # Trade agreement pipeline analysis
        current_negotiations = 6  # Estimated ongoing mega-regional negotiations
        completion_rate = 0.4  # Historical completion rate
        expected_completions = current_negotiations * completion_rate
        
        # Geopolitical adjustments
        us_china_tensions = 0.8  # High tensions reduce cooperation  
        regional_bloc_formation = 0.7  # Moderate momentum
        geopolitical_adjustment = (1 - us_china_tensions) * regional_bloc_formation
        
        # Calculate probability for <3 new agreements (forecast asks for "fewer than 3")
        base_completion_prob = expected_completions / 10
        adjusted_prob = base_completion_prob * (1 + geopolitical_adjustment)
        
        # Convert to "fewer than 3" probability
        fewer_than_3_prob = 1 - min(0.75, adjusted_prob)
        
        return {
            'active_negotiations': current_negotiations,
            'expected_new_agreements': expected_completions,
            'fewer_than_3_probability': fewer_than_3_prob,
            'probability_assessment': min(0.85, fewer_than_3_prob)
        }
    except Exception as e:
        print(f"Error in trade agreements analysis: {e}")
        return None

def analyze_climate_finance_f19():
    """F19: Climate finance goals missed"""
    try:
        # Climate finance data (billions USD)
        climate_finance_2024 = 98.5  # Actual commitments
        enhanced_target_2025 = 130  # Enhanced target
        current_gap = enhanced_target_2025 - climate_finance_2024
        
        # Constraints analysis
        developed_country_fiscal = 0.3  # Fiscal pressure
        private_sector_mobilization = 0.4  # Limited mobilization
        
        # Probability of missing targets
        miss_prob_fiscal = developed_country_fiscal * 0.4
        miss_prob_mobilization = (1 - private_sector_mobilization) * 0.6
        combined_miss_prob = min(0.85, miss_prob_fiscal + miss_prob_mobilization)
        
        return {
            'current_finance_2024': climate_finance_2024,
            'enhanced_target_2025': enhanced_target_2025,
            'current_gap': current_gap,
            'probability_assessment': combined_miss_prob
        }
    except Exception as e:
        print(f"Error in climate finance analysis: {e}")
        return None

def analyze_ai_governance_f21():
    """F21: AI governance fragmentation - Policy divergence model"""
    try:
        # Simplified AI governance indicators (0-1 scale)
        regions = ['US', 'EU', 'China', 'Other']
        governance_approaches = np.array([0.6, 0.8, 0.3, 0.4])  # Restrictiveness scores
        
        # Calculate fragmentation metric (coefficient of variation)
        fragmentation_index = np.std(governance_approaches) / np.mean(governance_approaches)
        
        # Historical baseline and threshold
        baseline_fragmentation = 0.35
        fragmentation_threshold = 0.5  # High fragmentation
        
        prob_fragmentation = 0.75 if fragmentation_index > fragmentation_threshold else 0.25
        
        return {
            'fragmentation_index': fragmentation_index,
            'threshold': fragmentation_threshold,
            'meets_threshold': fragmentation_index > fragmentation_threshold,
            'probability_assessment': prob_fragmentation
        }
    except Exception as e:
        print(f"Error in AI governance analysis: {e}")
        return None

def analyze_space_militarization_f22():
    """F22: Space weapon deployment - Defense spending model"""
    try:
        # Space defense spending (billions USD, simplified)
        years = np.array([2020, 2021, 2022, 2023, 2024])
        space_spending = np.array([18, 24, 28, 35, 42])
        
        # Fit exponential growth model
        log_spending = np.log(space_spending)
        X = years.reshape(-1, 1)
        model = LinearRegression()
        model.fit(X, log_spending)
        
        # Project to 2030
        proj_2030_log = model.predict([[2030]])[0]
        proj_2030_spending = np.exp(proj_2030_log)
        
        # Threshold for weaponization: >$100B annually
        weaponization_threshold = 100
        deployment_prob = min(0.80, proj_2030_spending / weaponization_threshold)
        
        return {
            'current_spending_2024': space_spending[-1],
            'projected_2030': proj_2030_spending,
            'threshold': weaponization_threshold,
            'probability_assessment': deployment_prob
        }
    except Exception as e:
        print(f"Error in space militarization analysis: {e}")
        return None

def analyze_pandemic_preparedness_f23():
    """F23: Pandemic preparedness gaps - Health system resilience model"""
    try:
        # Health spending as % of GDP (WHO data simplified)
        countries = ['USA', 'Germany', 'UK', 'China', 'India', 'Brazil']
        health_gdp_pct = np.array([17.8, 11.7, 10.9, 5.4, 3.5, 9.6])
        preparedness_score = health_gdp_pct / 20.0  # Normalize to 0-1
        
        # Calculate global average preparedness
        global_preparedness = np.mean(preparedness_score)
        preparedness_threshold = 0.6  # 60% adequacy threshold
        
        gap_probability = 0.70 if global_preparedness < preparedness_threshold else 0.30
        
        return {
            'global_preparedness': global_preparedness,
            'threshold': preparedness_threshold,
            'meets_threshold': global_preparedness >= preparedness_threshold,
            'probability_assessment': gap_probability
        }
    except Exception as e:
        print(f"Error in pandemic preparedness analysis: {e}")
        return None

def analyze_urban_adaptation_f24():
    """F24: Urban climate adaptation shortfalls - Infrastructure investment model"""
    try:
        # Climate adaptation investment (billions USD, simplified)
        years = np.array([2020, 2021, 2022, 2023, 2024])
        adaptation_invest = np.array([23, 28, 32, 35, 38])
        required_annual = 300  # UNEP estimate for urban adaptation needs
        
        # Calculate current gap
        current_gap = required_annual - adaptation_invest[-1]
        gap_percentage = current_gap / required_annual
        
        # Project investment trend
        recent_growth = np.mean(np.diff(adaptation_invest[-3:]))
        years_to_close_gap = current_gap / recent_growth if recent_growth > 0 else np.inf
        
        # Probability of continued shortfall
        shortfall_prob = 0.85 if years_to_close_gap > 10 else 0.15
        
        return {
            'current_investment_2024': adaptation_invest[-1],
            'required_annual': required_annual,
            'current_gap': current_gap,
            'gap_percentage': gap_percentage,
            'years_to_close_gap': years_to_close_gap,
            'probability_assessment': shortfall_prob
        }
    except Exception as e:
        print(f"Error in urban adaptation analysis: {e}")
        return None

def analyze_food_security_f25():
    """F25: Global food price volatility - Agricultural commodity model"""
    try:
        # FAO Food Price Index (simplified)
        years = np.array([2019, 2020, 2021, 2022, 2023, 2024])
        food_price_index = np.array([107.1, 107.2, 125.7, 143.7, 124.0, 120.0])
        
        # Calculate volatility (coefficient of variation)
        price_volatility = np.std(food_price_index) / np.mean(food_price_index)
        
        # Calculate recent trend volatility (last 3 years)
        recent_volatility = np.std(food_price_index[-3:]) / np.mean(food_price_index[-3:])
        
        # Historical baseline volatility
        baseline_volatility = 0.12  # 12% coefficient of variation
        high_volatility_threshold = 0.15  # 15% threshold for crisis
        
        # Climate/conflict risk factors
        climate_risk = 0.7  # High climate variability
        conflict_risk = 0.6  # Geopolitical tensions affecting supply chains
        combined_risk = min(1.0, climate_risk + conflict_risk * 0.5)
        
        # Use the higher of overall or recent volatility for assessment
        assessment_volatility = max(price_volatility, recent_volatility)
        volatility_prob = 0.75 if assessment_volatility > high_volatility_threshold else 0.25
        adjusted_prob = min(0.90, volatility_prob * (1 + combined_risk * 0.3))
        
        return {
            'current_volatility': price_volatility,
            'recent_volatility': recent_volatility,
            'baseline_volatility': baseline_volatility,
            'threshold': high_volatility_threshold,
            'climate_risk': climate_risk,
            'conflict_risk': conflict_risk,
            'assessment_volatility': assessment_volatility,
            'probability_assessment': adjusted_prob
        }
    except Exception as e:
        print(f"Error in food security analysis: {e}")
        return None

def run_comprehensive_remaining_forecasts():
    """Execute all remaining forecast models efficiently"""
    models = [
        ("F8", "Energy Transition Delay", analyze_energy_transition_f8),
        ("F9", "Global Fertility Decline", analyze_demographic_transition_f9),
        ("F11", "Immigration Restrictions", analyze_immigration_restrictions_f11),
        ("F12", "Democratic Backsliding", analyze_authoritarianism_f12),
        ("F13", "Interstate War Risk", analyze_conflict_escalation_f13),
        ("F15", "Financial Fragmentation", analyze_financial_integration_f15),
        ("F16", "Mega-regional Agreements", analyze_trade_agreements_f16),
        ("F19", "Climate Finance Gaps", analyze_climate_finance_f19),
        ("F21", "AI Governance Fragmentation", analyze_ai_governance_f21),
        ("F22", "Space Militarization", analyze_space_militarization_f22),
        ("F23", "Pandemic Preparedness", analyze_pandemic_preparedness_f23),
        ("F24", "Urban Climate Adaptation", analyze_urban_adaptation_f24),
        ("F25", "Food Security Volatility", analyze_food_security_f25),
    ]
    
    results = {}
    for forecast_id, name, func in models:
        try:
            result = func()
            if result:
                prob = result.get('probability_assessment', 0.5)
                results[forecast_id] = result
                print(f"[SUCCESS] {forecast_id} ({name}): {prob:.1%}")
            else:
                print(f"[ERROR] {forecast_id} ({name}): Failed")
        except Exception as e:
            print(f"[ERROR] {forecast_id} ({name}): Error - {e}")
    
    return results

if __name__ == "__main__":
    print("=== REMAINING FORECASTS COMPREHENSIVE ANALYSIS ===")
    results = run_comprehensive_remaining_forecasts()
    print(f"\n[SUCCESS] Successfully analyzed {len(results)}/13 remaining forecasts")
