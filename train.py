import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
df = pd.read_csv("df_clean.csv")

# Convert hour string "06" to integer 6
df['hour'] = pd.to_numeric(df['hour'], errors='coerce')
df.dropna(subset=['hour'], inplace=True)

# Create targets: next latitude and longitude
df['next_lat'] = df['lat'].shift(-1)
df['next_lon'] = df['lon'].shift(-1)
df.dropna(inplace=True)

# Calculate center lat/lon for feature engineering
center_lat = df['lat'].mean()
center_lon = df['lon'].mean()
print(f"Center Latitude: {center_lat}, Center Longitude: {center_lon}")

# Feature engineering
df['dist_from_center'] = np.sqrt((df['lat'] - center_lat)**2 + (df['lon'] - center_lon)**2)
df['angle_from_center'] = np.arctan2(df['lat'] - center_lat, df['lon'] - center_lon)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

features = ['dist_from_center', 'angle_from_center', 'hour_sin', 'hour_cos', 'hour']

# Target differences (deltas)
df['delta_lat'] = df['next_lat'] - df['lat']
df['delta_lon'] = df['next_lon'] - df['lon']

# Train Random Forest regressors
lat_model = RandomForestRegressor(n_estimators=100, random_state=42)
lon_model = RandomForestRegressor(n_estimators=100, random_state=42)

lat_model.fit(df[features], df['delta_lat'])
lon_model.fit(df[features], df['delta_lon'])

# Save models to disk
joblib.dump(lat_model, "lat_model.pkl")
joblib.dump(lon_model, "lon_model.pkl")

# Save center point to disk for later use
np.savez("center_point.npz", center_lat=center_lat, center_lon=center_lon)

print("✅ Models trained and saved successfully.")
