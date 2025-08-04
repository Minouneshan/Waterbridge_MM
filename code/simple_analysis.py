import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
import networkx as nx
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Project root directory two levels above this script
ROOT_DIR = Path(__file__).resolve().parents[1]

def run_simple_analysis():
    """Run simple analysis and generate key visualizations"""
    print("🔍 MODERN MERCANTILISM ANALYSIS")
    print("=" * 50)
    
    # Load data
    data_dir = ROOT_DIR / 'data'
    china_imports = pd.read_csv(data_dir / 'us_imports.csv')
    vietnam_imports = pd.read_csv(data_dir / 'vietnam_us_imports.csv')
    brics_gdp = pd.read_csv(data_dir / 'brics_gdp_share.csv')
    
    # Analyze China imports decline (F6)
    print("\n📉 F6: China Import Share Decline")
    china_2027 = china_imports[china_imports['Year'] == 2027]['China_Share'].iloc[0]
    print(f"Projected 2027 China Share: {china_2027:.1f}%")
    print(f"Below 12% threshold: {'✓' if china_2027 < 12 else '✗'}")
    
    # Analyze Vietnam imports growth (F7)
    print("\n📈 F7: Vietnam Import Growth")
    vietnam_2022 = vietnam_imports[vietnam_imports['Year'] == 2022]['Imports_USD_Billion'].iloc[0]
    vietnam_2027 = vietnam_imports[vietnam_imports['Year'] == 2027]['Imports_USD_Billion'].iloc[0]
    target = vietnam_2022 * 2
    print(f"2022 Baseline: ${vietnam_2022:.1f}B")
    print(f"2027 Projected: ${vietnam_2027:.1f}B")
    print(f"Target (2x): ${target:.1f}B")
    print(f"Will double: {'✓' if vietnam_2027 >= target else '✗'}")
    
    # Analyze BRICS expansion (F10)
    print("\n🌍 F10: BRICS GDP Share")
    brics_2030 = brics_gdp[brics_gdp['Year'] == 2030]['BRICS_GDP_Share_PPP'].iloc[0]
    print(f"Projected 2030 BRICS Share: {brics_2030:.1f}%")
    print(f"Above 40% threshold: {'✓' if brics_2030 >= 40 else '✗'}")
    
    # Create visualizations
    create_visualizations()
    
    print("\n✅ Analysis complete! Charts saved to docs/ directory")

def create_visualizations():
    """Create key visualizations"""
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Load data
    data_dir = ROOT_DIR / 'data'
    china_imports = pd.read_csv(data_dir / 'us_imports.csv')
    vietnam_imports = pd.read_csv(data_dir / 'vietnam_us_imports.csv')
    brics_gdp = pd.read_csv(data_dir / 'brics_gdp_share.csv')
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. China import share decline
    historical = china_imports[china_imports['Year'] <= 2024]
    projected = china_imports[china_imports['Year'] >= 2024]
    
    ax1.plot(historical['Year'], historical['China_Share'], 'o-', 
             linewidth=3, markersize=8, label='Historical', color='darkred')
    ax1.plot(projected['Year'], projected['China_Share'], 's--', 
             linewidth=3, markersize=8, label='Projected', color='red', alpha=0.7)
    ax1.axhline(y=12, color='orange', linestyle='-', linewidth=2, 
                label='12% Threshold (F6)', alpha=0.8)
    ax1.set_title('F6: China\'s Share of U.S. Imports', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Share (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Vietnam import growth
    ax2.plot(vietnam_imports['Year'], vietnam_imports['Imports_USD_Billion'], 
             'o-', linewidth=3, markersize=10, color='darkgreen', label='Vietnam Imports')
    baseline_2022 = vietnam_imports[vietnam_imports['Year'] == 2022]['Imports_USD_Billion'].iloc[0]
    ax2.axhline(y=baseline_2022*2, color='orange', linestyle='--', 
                linewidth=2, label=f'Target: ${baseline_2022*2:.0f}B (2x)', alpha=0.8)
    ax2.set_title('F7: U.S. Imports from Vietnam', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Imports (USD Billion)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. BRICS GDP share
    ax3.plot(brics_gdp['Year'], brics_gdp['BRICS_GDP_Share_PPP'], 
             'o-', linewidth=3, markersize=8, color='purple', label='BRICS GDP Share')
    ax3.axhline(y=40, color='orange', linestyle='-', linewidth=2, 
                label='40% Target (F10)', alpha=0.8)
    ax3.set_title('F10: BRICS Share of World GDP (PPP)', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Share (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Simple network diagram
    G = nx.DiGraph()
    
    # Add key nodes
    nodes = ['F1\nTariffs', 'F6\nChina\nDecline', 'F7\nVietnam\nRise', 
             'F10\nBRICS\nExpansion', 'F25\nDefense\nSpending']
    G.add_nodes_from(nodes)
    
    # Add key edges
    edges = [('F1\nTariffs', 'F6\nChina\nDecline'), 
             ('F6\nChina\nDecline', 'F7\nVietnam\nRise'),
             ('F1\nTariffs', 'F25\nDefense\nSpending'),
             ('F25\nDefense\nSpending', 'F10\nBRICS\nExpansion')]
    G.add_edges_from(edges)
    
    # Draw network
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, ax=ax4, node_color='lightblue', 
                          node_size=2000, alpha=0.8)
    nx.draw_networkx_edges(G, pos, ax=ax4, edge_color='gray', 
                          arrows=True, arrowsize=20, width=2, alpha=0.7)
    nx.draw_networkx_labels(G, pos, ax=ax4, font_size=9, font_weight='bold')
    ax4.set_title('Key Forecast Dependencies', fontweight='bold', fontsize=12)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(ROOT_DIR / 'docs' / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create simple Bayesian network visualization
    create_bayesian_network()

def create_bayesian_network():
    """Create Bayesian network visualization"""
    plt.figure(figsize=(14, 10))
    
    # Create network
    G = nx.DiGraph()
    
    # Define forecast clusters
    trade_forecasts = ['F1', 'F3', 'F6', 'F7']
    tech_forecasts = ['F13', 'F14', 'F23', 'F24']
    finance_forecasts = ['F10', 'F17', 'F18', 'F22']
    security_forecasts = ['F25', 'F21', 'F12', 'F20']
    
    all_forecasts = trade_forecasts + tech_forecasts + finance_forecasts + security_forecasts
    
    # Add nodes
    for f in all_forecasts:
        G.add_node(f)
    
    # Add key edges (simplified)
    key_edges = [
        ('F1', 'F3'), ('F1', 'F6'), ('F6', 'F7'),  # Trade chain
        ('F23', 'F14'), ('F14', 'F24'), ('F24', 'F13'),  # Tech chain
        ('F22', 'F10'), ('F10', 'F17'), ('F17', 'F18'),  # Finance chain
        ('F25', 'F12'), ('F12', 'F20'), ('F20', 'F21'),  # Security chain
        # Cross-connections
        ('F1', 'F25'), ('F25', 'F21'), ('F12', 'F13')
    ]
    
    G.add_edges_from(key_edges)
    
    # Position nodes by cluster
    pos = {}
    # Trade cluster (top-left)
    for i, f in enumerate(trade_forecasts):
        pos[f] = (-2 + i*0.8, 2)
    # Tech cluster (top-right)  
    for i, f in enumerate(tech_forecasts):
        pos[f] = (1 + i*0.8, 2)
    # Finance cluster (bottom-left)
    for i, f in enumerate(finance_forecasts):
        pos[f] = (-2 + i*0.8, -1)
    # Security cluster (bottom-right)
    for i, f in enumerate(security_forecasts):
        pos[f] = (1 + i*0.8, -1)
    
    # Color nodes by cluster
    node_colors = []
    for node in G.nodes():
        if node in trade_forecasts:
            node_colors.append('lightblue')
        elif node in tech_forecasts:
            node_colors.append('lightcoral')
        elif node in finance_forecasts:
            node_colors.append('lightgreen')
        else:
            node_colors.append('lightyellow')
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=1500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                          arrows=True, arrowsize=15, width=1.5, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                  markersize=10, label='Trade'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                  markersize=10, label='Technology'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                  markersize=10, label='Finance'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightyellow', 
                  markersize=10, label='Security')
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.title('Bayesian Belief Network: Modern Mercantilism Forecasts\n' +
              'Four Major Causal Chains with Cross-Cluster Dependencies', 
              fontsize=14, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(ROOT_DIR / 'docs' / 'bayesian_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_simple_analysis()
