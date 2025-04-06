import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)

# Number of datapoints
n = 1000
start_time = datetime.now()

data = []

for i in range(n):
    sensor_timestamp = start_time + timedelta(minutes=i)
    
    # Simulate sensor values (smoke levels from MQ2)
    # Higher values indicate more smoke/gas presence
    base_value = np.random.normal(loc=300, scale=100)
    fire_event = np.random.rand() < 0.3  # 30% chance of fire
    if fire_event:
        sensor_value = int(np.clip(base_value + np.random.randint(150, 300), 100, 1023))
        confidence = np.random.randint(70, 100)
        fire_radiative_power = round(np.random.uniform(20.0, 150.0), 2)
    else:
        sensor_value = int(np.clip(base_value, 100, 400))
        confidence = np.random.randint(0, 60)
        fire_radiative_power = round(np.random.uniform(0.0, 20.0), 2)

    # MODIS Fire detection coordinates - mostly within Africa for realism
    fire_lat = round(np.random.uniform(-35.0, 35.0), 4)
    fire_lon = round(np.random.uniform(-20.0, 55.0), 4)

    modis_timestamp = sensor_timestamp - timedelta(seconds=np.random.randint(30, 180))

    # Label if fire is detected (based on high sensor value and MODIS confidence)
    fire_detected = int(sensor_value > 400 and confidence >= 70 and fire_radiative_power >= 20)

    data.append([
        sensor_timestamp, sensor_value, fire_lat, fire_lon,
        confidence, fire_radiative_power, modis_timestamp, fire_detected
    ])

columns = [
    "sensor_timestamp", "sensor_value", "fire_lat", "fire_lon",
    "confidence", "fire_radiative_power", "modis_timestamp", "fire_detected"
]

df_realistic = pd.DataFrame(data, columns=columns)

# Save to CSV
realistic_file_path = "realistic_fire_detection_dataset.csv"
df_realistic.to_csv(realistic_file_path, index=False)
realistic_file_path
