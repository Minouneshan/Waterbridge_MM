import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Example index calc: US import share from China (mock data)
ROOT_DIR = Path(__file__).resolve().parents[1]

data_file = ROOT_DIR / 'data' / 'us_imports.csv'
try:
    data = pd.read_csv(data_file)
except FileNotFoundError:
    print(f"Creating dummy data for {data_file.relative_to(ROOT_DIR)}")
    dummy_data = {
        'Year': range(2015, 2028),
        'China_Share': [21.5, 21.1, 21.6, 21.2, 18.1, 18.6, 17.9, 16.5,
                        14.6, 13.9, 13.2, 12.5, 11.8]
    }
    data = pd.DataFrame(dummy_data)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_file, index=False)

# Basic visualisation (saved into docs/ for consistency)
plt.plot(data['Year'], data['China_Share'], marker='o')
plt.axhline(y=12, color='r', linestyle='--', label='Forecast Threshold')
plt.title('China Share of US Imports')
plt.xlabel('Year')
plt.ylabel('% Share')
plt.legend()
plt.grid(True)
plt.savefig(ROOT_DIR / 'docs' / 'china_imports_trend_quick.png', dpi=300, bbox_inches='tight')
print("Chart saved to docs/china_imports_trend_quick.png")
