import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)

# Number of datapoints
n = 1000000  # Updated to 1,000,000 rows
start_time = datetime.now()

data = []

for i in range(n):
    sensor_timestamp = start_time + timedelta(minutes=i)
    
    # Simulate sensor values (smoke levels from MQ2)
    # Higher values indicate more smoke/gas presence
    base_value = np.random.normal(loc=300, scale=100)  # Base sensor value (random normal distribution)
    
    # Add some noise to the base sensor value
    noise = np.random.normal(0, 50)  # Gaussian noise
    base_value += noise  # Adding noise to the sensor value
    
    # Introduce random variability in fire occurrence
    fire_event = np.random.rand() < 0.3  # 30% chance of fire event, randomly decided
    
    if fire_event:
        # If a fire event occurs, increase the sensor value significantly
        sensor_value = int(np.clip(base_value + np.random.randint(150, 300), 100, 1023))
        confidence = np.random.randint(70, 100)  # Higher confidence during fire
        fire_radiative_power = round(np.random.uniform(50.0, 200.0), 2)  # Higher radiative power during fire
    else:
        # If no fire event, keep the sensor value lower and decrease confidence
        sensor_value = int(np.clip(base_value, 100, 400))
        confidence = np.random.randint(0, 60)  # Lower confidence when no fire detected
        fire_radiative_power = round(np.random.uniform(0.0, 20.0), 2)  # Low radiative power in absence of fire

    # MODIS Fire detection coordinates (in realistic range)
    fire_lat = round(np.random.uniform(-35.0, 35.0), 4)
    fire_lon = round(np.random.uniform(-20.0, 55.0), 4)
    
    # Simulate modis timestamp with slight randomness (fire detection lag)
    modis_timestamp = sensor_timestamp - timedelta(seconds=np.random.randint(30, 180))
    
    # Randomly set some modis_timestamp and fire_radiative_power values to NaN (simulate missing data)
    if np.random.rand() < 0.05:  # 5% chance of missing satellite data
        modis_timestamp = np.nan  # Missing MODIS timestamp
        fire_radiative_power = np.nan  # Missing fire radiative power

    # Introduce a slightly more complex fire detection rule (non-linear threshold with random variation)
    # Randomize fire detection thresholds slightly each time to add complexity
    threshold_sensor_value = 400 + np.random.randint(-50, 50)
    threshold_confidence = 70 + np.random.randint(-5, 10)
    threshold_fire_radiative_power = 20 + np.random.randint(-5, 10)
    
    fire_detected = int(
        sensor_value > threshold_sensor_value and
        confidence >= threshold_confidence and
        fire_radiative_power >= threshold_fire_radiative_power
    )
    
    # Add random fluctuation to the sensor value for added noise
    sensor_value += np.random.normal(0, 20)  # Small fluctuation

    data.append([
        sensor_timestamp, sensor_value, fire_lat, fire_lon,
        confidence, fire_radiative_power, modis_timestamp, fire_detected
    ])

columns = [
    "sensor_timestamp", "sensor_value", "fire_lat", "fire_lon",
    "confidence", "fire_radiative_power", "modis_timestamp", "fire_detected"
]

# Create DataFrame
df_realistic = pd.DataFrame(data, columns=columns)

# Save to CSV
realistic_file_path = "realistic_fire_detection_dataset_with_nans.csv"
df_realistic.to_csv(realistic_file_path, index=False)

realistic_file_path  # Output file path
