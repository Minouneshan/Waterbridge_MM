import pandas as pd
import matplotlib.pyplot as plt

# Example index calc: US import share from China (mock data)
# Create a dummy csv for demonstration if it doesn't exist
try:
    data = pd.read_csv("data/us_imports.csv")
except FileNotFoundError:
    print("Creating dummy data for us_imports.csv")
    dummy_data = {'Year': range(2015, 2028), 'China_Share': [21.5, 21.1, 21.6, 21.2, 18.1, 18.6, 17.9, 16.5, 14.6, 13.9, 13.2, 12.5, 11.8]}
    data = pd.DataFrame(dummy_data)
    data.to_csv("data/us_imports.csv", index=False)


plt.plot(data['Year'], data['China_Share'], marker='o')
plt.axhline(y=12, color='r', linestyle='--', label='Forecast Threshold')
plt.title('China Share of US Imports')
plt.xlabel('Year')
plt.ylabel('% Share')
plt.legend()
plt.grid(True)
plt.savefig("code/china_imports_trend.png")
print("Chart saved to code/china_imports_trend.png")
