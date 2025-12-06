import numpy as np
import joblib

def load_models():
    lat_model = joblib.load("lat_model.pkl")
    lon_model = joblib.load("lon_model.pkl")
    return lat_model, lon_model

def predict_next_location(lat_model, lon_model, user_lat, user_lon, user_hour):
    center = np.load("center_point.npz")
    center_lat = center['center_lat']
    center_lon = center['center_lon']

    dist_from_center = np.sqrt((user_lat - center_lat)**2 + (user_lon - center_lon)**2)
    angle_from_center = np.arctan2(user_lat - center_lat, user_lon - center_lon)
    hour_sin = np.sin(2 * np.pi * user_hour / 24)
    hour_cos = np.cos(2 * np.pi * user_hour / 24)

    features = np.array([[dist_from_center, angle_from_center, hour_sin, hour_cos, user_hour]])

    delta_lat = lat_model.predict(features)[0]
    delta_lon = lon_model.predict(features)[0]

    next_lat = user_lat + delta_lat
    next_lon = user_lon + delta_lon

    return next_lat, next_lon
