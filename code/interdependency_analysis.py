"""
Comprehensive interdependency analysis for Modern Mercantilism forecasting framework.
This module analyzes how the 25 forecasts influence each other through causal pathways.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).resolve().parents[1]

def analyze_forecast_interdependencies(all_results):
    """Analyze causal relationships and interdependencies between forecasts"""
    
    # Extract probability data for all forecasts with quantitative models
    forecast_probs = {
        'F1_tariff_escalation': all_results.get('tariff_analysis', {}).get('probability_assessment', 0.30),
        'F2_wto_vacancy': all_results.get('wto_analysis', {}).get('probability_assessment', 0.85),
        'F3_trade_gdp_decouple': all_results.get('trade_gdp_analysis', {}).get('probability_assessment', 0.25),
        'F4_grain_bans': all_results.get('grain_analysis', {}).get('probability_assessment', 1.00),
        'F5_services_goods_div': all_results.get('services_analysis', {}).get('probability_assessment', 0.80),
        'F6_china_decline': all_results.get('china_analysis', {}).get('forecast_validation', {}).get('probability_assessment', 0.75),
        'F7_vietnam_growth': 0.72 if all_results.get('vietnam_analysis', {}).get('will_double', False) else 0.28,
        'F8_energy_delay': all_results.get('energy_analysis', {}).get('probability_assessment', 0.90),
        'F9_fertility_decline': all_results.get('demo_analysis', {}).get('probability_assessment', 0.70),
        'F10_brics_expansion': 0.70 if all_results.get('brics_analysis', {}).get('gdp_analysis', {}).get('meets_40_percent', False) else 0.30,
        'F11_immigration_restrict': all_results.get('immigration_analysis', {}).get('probability_assessment', 0.65),
        'F12_democracy_backslide': all_results.get('democracy_analysis', {}).get('probability_assessment', 0.75),
        'F13_interstate_war': all_results.get('conflict_analysis', {}).get('probability_assessment', 0.80),
        'F14_tech_bifurcation': 0.85,  # From advanced models
        'F15_finance_fragment': all_results.get('finance_analysis', {}).get('probability_assessment', 0.40),
        'F16_trade_agreements': all_results.get('trade_agreements_analysis', {}).get('probability_assessment', 0.27),
        'F17_usd_dominance': 0.66,  # From VAR model
        'F18_rmb_limited': 0.85,    # From VAR model
        'F19_climate_finance': all_results.get('climate_finance_analysis', {}).get('probability_assessment', 0.48),
        'F20_carbon_tariffs': 0.82  # From logistic model
    }
    
    # Define causal influence matrix (simplified)
    # Rows influence columns (row -> column causation)
    influences = {
        # Economic fragmentation cluster
        ('F1_tariff_escalation', 'F3_trade_gdp_decouple'): 0.7,
        ('F1_tariff_escalation', 'F6_china_decline'): 0.8,
        ('F2_wto_vacancy', 'F3_trade_gdp_decouple'): 0.6,
        ('F2_wto_vacancy', 'F16_trade_agreements'): -0.5,  # Negative correlation
        
        # Trade restructuring
        ('F6_china_decline', 'F7_vietnam_growth'): 0.9,  # Strong substitution
        ('F6_china_decline', 'F14_tech_bifurcation'): 0.6,
        ('F7_vietnam_growth', 'F5_services_goods_div'): 0.4,
        
        # Institutional decay
        ('F12_democracy_backslide', 'F13_interstate_war'): 0.6,
        ('F12_democracy_backslide', 'F11_immigration_restrict'): 0.5,
        ('F13_interstate_war', 'F4_grain_bans'): 0.8,
        ('F13_interstate_war', 'F15_finance_fragment'): 0.7,
        
        # Economic bloc formation
        ('F10_brics_expansion', 'F17_usd_dominance'): -0.6,
        ('F10_brics_expansion', 'F15_finance_fragment'): 0.5,
        ('F17_usd_dominance', 'F18_rmb_limited'): -0.4,
        
        # Climate-energy nexus
        ('F8_energy_delay', 'F19_climate_finance'): 0.7,
        ('F8_energy_delay', 'F20_carbon_tariffs'): 0.6,
        ('F19_climate_finance', 'F20_carbon_tariffs'): 0.5,
        
        # Demographic pressures
        ('F9_fertility_decline', 'F11_immigration_restrict'): -0.4,  # More need, more restriction
        ('F11_immigration_restrict', 'F12_democracy_backslide'): 0.3,
    }
    
    return forecast_probs, influences

def calculate_cascade_effects(forecast_probs, influences, shock_forecast, shock_magnitude):
    """Calculate cascade effects of a shock to one forecast on others"""
    
    # Initialize shock
    shocked_probs = forecast_probs.copy()
    shocked_probs[shock_forecast] = max(0, min(1, shocked_probs[shock_forecast] + shock_magnitude))
    
    # Iterative propagation (simplified)
    for iteration in range(3):  # 3 rounds of propagation
        for (source, target), strength in influences.items():
            if source == shock_forecast or source in shocked_probs:
                # Calculate influence
                source_prob = shocked_probs.get(source, forecast_probs.get(source, 0.5))
                target_prob = shocked_probs.get(target, forecast_probs.get(target, 0.5))
                
                # Apply influence (dampened by iteration)
                influence_effect = strength * (source_prob - 0.5) * (0.5 ** iteration)
                new_target_prob = target_prob + influence_effect * 0.1  # Damping factor
                
                shocked_probs[target] = max(0, min(1, new_target_prob))
    
    return shocked_probs

def generate_interdependency_report(all_results):
    """Generate comprehensive interdependency analysis report"""
    
    print("\n" + "="*70)
    print("MODERN MERCANTILISM INTERDEPENDENCY ANALYSIS")
    print("="*70)
    
    forecast_probs, influences = analyze_forecast_interdependencies(all_results)
    
    # Summary statistics
    high_prob_forecasts = {k: v for k, v in forecast_probs.items() if v > 0.75}
    moderate_prob_forecasts = {k: v for k, v in forecast_probs.items() if 0.4 <= v <= 0.75}
    low_prob_forecasts = {k: v for k, v in forecast_probs.items() if v < 0.4}
    
    print(f"\n[INFO] PROBABILITY DISTRIBUTION:")
    print(f"  High Probability (>75%): {len(high_prob_forecasts)} forecasts")
    print(f"  Moderate Probability (40-75%): {len(moderate_prob_forecasts)} forecasts")
    print(f"  Low Probability (<40%): {len(low_prob_forecasts)} forecasts")
    
    print(f"\n🔥 HIGHEST RISK FORECASTS:")
    sorted_probs = sorted(forecast_probs.items(), key=lambda x: x[1], reverse=True)
    for i, (forecast, prob) in enumerate(sorted_probs[:8]):
        clean_name = forecast.replace('_', ' ').replace('F', 'F').title()
        print(f"  {i+1}. {clean_name}: {prob:.1%}")
    
    # Scenario analysis
    print(f"\n⚡ SCENARIO ANALYSIS:")
    print("-" * 50)
    
    # Scenario 1: China-US trade war escalation
    print("📈 SCENARIO 1: Trade War Escalation (+20% F1 Tariff Risk)")
    shocked_probs_1 = calculate_cascade_effects(forecast_probs, influences, 'F1_tariff_escalation', 0.20)
    
    major_changes_1 = []
    for forecast in forecast_probs:
        original = forecast_probs[forecast]
        shocked = shocked_probs_1[forecast]
        change = shocked - original
        if abs(change) > 0.05:  # 5% threshold
            major_changes_1.append((forecast, original, shocked, change))
    
    for forecast, orig, shock, change in sorted(major_changes_1, key=lambda x: abs(x[3]), reverse=True)[:5]:
        clean_name = forecast.replace('_', ' ').title()
        print(f"  • {clean_name}: {orig:.1%} → {shock:.1%} ({change:+.1%})")
    
    # Scenario 2: Democratic backsliding acceleration
    print("\n📉 SCENARIO 2: Democratic Collapse (+15% F12 Democracy Risk)")
    shocked_probs_2 = calculate_cascade_effects(forecast_probs, influences, 'F12_democracy_backslide', 0.15)
    
    major_changes_2 = []
    for forecast in forecast_probs:
        original = forecast_probs[forecast]
        shocked = shocked_probs_2[forecast]
        change = shocked - original
        if abs(change) > 0.05:
            major_changes_2.append((forecast, original, shocked, change))
    
    for forecast, orig, shock, change in sorted(major_changes_2, key=lambda x: abs(x[3]), reverse=True)[:5]:
        clean_name = forecast.replace('_', ' ').title()
        print(f"  • {clean_name}: {orig:.1%} → {shock:.1%} ({change:+.1%})")
    
    # Critical paths analysis
    print(f"\n🔗 CRITICAL CAUSAL PATHWAYS:")
    print("-" * 50)
    
    critical_paths = [
        ("Trade War → China Decline → Vietnam Rise", 
         ["F1_tariff_escalation", "F6_china_decline", "F7_vietnam_growth"]),
        ("Democracy Decline → Conflict → Food Crisis",
         ["F12_democracy_backslide", "F13_interstate_war", "F4_grain_bans"]),
        ("Energy Delay → Climate Finance Failure → Carbon Tariffs",
         ["F8_energy_delay", "F19_climate_finance", "F20_carbon_tariffs"]),
        ("BRICS Rise → USD Decline → Financial Fragmentation",
         ["F10_brics_expansion", "F17_usd_dominance", "F15_finance_fragment"])
    ]
    
    for path_name, path_forecasts in critical_paths:
        path_probs = [forecast_probs.get(f, 0.5) for f in path_forecasts]
        joint_prob = np.prod(path_probs)  # Simplified joint probability
        print(f"  • {path_name}: {joint_prob:.1%}")
        for i, forecast in enumerate(path_forecasts):
            clean_name = forecast.replace('_', ' ').title()
            print(f"    {i+1}. {clean_name} ({forecast_probs.get(forecast, 0.5):.1%})")
    
    # System fragility assessment
    print(f"\n⚠️  SYSTEM FRAGILITY ASSESSMENT:")
    print("-" * 50)
    
    # Calculate overall system stress
    avg_prob = np.mean(list(forecast_probs.values()))
    prob_variance = np.var(list(forecast_probs.values()))
    high_risk_count = len(high_prob_forecasts)
    
    # Fragility score (0-1 scale)
    fragility_score = (avg_prob * 0.4) + (prob_variance * 0.3) + (high_risk_count / 20 * 0.3)
    
    print(f"  Overall System Fragility: {fragility_score:.2f}/1.00")
    if fragility_score > 0.7:
        print("  🚨 Status: HIGH FRAGILITY - Multiple cascading risks")
    elif fragility_score > 0.5:
        print("  ⚠️  Status: MODERATE FRAGILITY - Significant vulnerabilities")
    else:
        print("  [SUCCESS] Status: LOW FRAGILITY - System resilience maintained")
    
    print(f"  Key Drivers:")
    print(f"    • Average Risk Level: {avg_prob:.1%}")
    print(f"    • Risk Concentration: {prob_variance:.3f}")
    print(f"    • High-Risk Forecasts: {high_risk_count}/20")
    
    return forecast_probs, influences

def create_network_visualization(forecast_probs, influences):
    """Create network visualization of forecast interdependencies"""
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for forecast in forecast_probs:
        clean_name = forecast.replace('_', '\n').replace('F', 'F')
        G.add_node(forecast, 
                  label=clean_name,
                  prob=forecast_probs[forecast],
                  size=forecast_probs[forecast] * 1000 + 300)
    
    # Add edges
    for (source, target), strength in influences.items():
        if abs(strength) > 0.3:  # Only show significant influences
            G.add_edge(source, target, weight=abs(strength), strength=strength)
    
    # Create visualization
    plt.figure(figsize=(16, 12))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Node colors based on probability
    node_colors = [forecast_probs[node] for node in G.nodes()]
    node_sizes = [forecast_probs[node] * 1000 + 300 for node in G.nodes()]
    
    # Draw nodes
    nodes = nx.draw_networkx_nodes(G, pos, 
                                  node_color=node_colors,
                                  node_size=node_sizes,
                                  cmap=plt.cm.Reds,
                                  alpha=0.8)
    
    # Draw edges
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, 
                          width=edge_weights,
                          alpha=0.6,
                          edge_color='gray',
                          arrows=True,
                          arrowsize=20,
                          arrowstyle='->')
    
    # Add labels
    labels = {node: f"F{node.split('_')[0][1:]}" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    plt.title('Modern Mercantilism Forecast Network\nNode Size & Color = Probability, Edge Width = Influence Strength', 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add colorbar
    ax = plt.gca()  # Get current axes
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Reds, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Forecast Probability', fontsize=12)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(ROOT_DIR / 'docs' / 'forecast_network.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Network visualization saved to: {ROOT_DIR / 'docs' / 'forecast_network.png'}")

if __name__ == "__main__":
    # This would be called from the main analysis
    print("Interdependency analysis module ready for integration")
