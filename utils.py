import numpy as np
import pandas as pd

def load_center():
    center = np.load("center_point.npz")
    return center['center_lat'], center['center_lon']

def prepare_input(lat, lon, hour):
    center_lat, center_lon = load_center()
    dist = np.sqrt((lat - center_lat) ** 2 + (lon - center_lon) ** 2)
    angle = np.arctan2(lat - center_lat, lon - center_lon)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    return pd.DataFrame([{
        'dist_from_center': dist,
        'angle_from_center': angle,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
        'hour': hour
    }])
