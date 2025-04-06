import psycopg2
import requests
import time
from datetime import datetime, timedelta

# Database configuration
DB_HOST = "localhost"
DB_NAME = "fire_detection_db"
DB_USER = "your_db_user"
DB_PASSWORD = "your_db_password"

# MODIS API endpoint (sample)
MODIS_API_URL = "https://modis-fire-api-endpoint.com/fire-detection"

# Time window for synchronization (e.g., 5 minutes)
TIME_WINDOW = timedelta(minutes=5)

# Connect to PostgreSQL database
def connect_to_db():
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST
        )
        return conn
    except Exception as e:
        print("Error connecting to database:", e)
        return None

# Function to fetch the latest sensor data from the database
def fetch_latest_sensor_data():
    db_conn = connect_to_db()
    if db_conn is None:
        return None
    
    cursor = db_conn.cursor()
    cursor.execute("SELECT * FROM sensor_data ORDER BY timestamp DESC LIMIT 1;")
    result = cursor.fetchone()
    
    db_conn.close()
    return result

# Function to get MODIS fire data (via API call)
def get_modis_data(sensor_timestamp):
    params = {
        'start_time': (sensor_timestamp - TIME_WINDOW).isoformat(),
        'end_time': sensor_timestamp.isoformat(),
    }

    try:
        response = requests.get(MODIS_API_URL, params=params)
        response.raise_for_status()  # Check if request was successful
        modis_data = response.json()  # Assuming the response is in JSON format
        return modis_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching MODIS data: {e}")
        return None

# Function to insert synchronized data into the database
def insert_synchronized_data_to_db(sensor_timestamp, sensor_value, modis_data):
    db_conn = connect_to_db()
    if db_conn is None:
        return
    
    cursor = db_conn.cursor()
    
    if modis_data:
        for fire in modis_data['fires']:
            # Insert synchronized data into the synchronized_fire_data table
            cursor.execute("""
                INSERT INTO synchronized_fire_data (sensor_timestamp, sensor_value, fire_lat, fire_lon, confidence, fire_radiative_power, modis_timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s);
            """, (
                sensor_timestamp, 
                sensor_value, 
                fire['lat'], 
                fire['lon'], 
                fire['confidence'], 
                fire['fire_radiative_power'],
                fire['timestamp']  # Assuming the MODIS data has a timestamp
            ))

    db_conn.commit()
    db_conn.close()

# Main loop to fetch data periodically
while True:
    # Fetch the latest sensor data (timestamp, sensor value)
    latest_sensor_data = fetch_latest_sensor_data()
    if latest_sensor_data:
        sensor_timestamp = latest_sensor_data[1]  # Assuming timestamp is in the 2nd column
        sensor_value = latest_sensor_data[2]  # Assuming sensor_value is in the 3rd column
        
        print(f"Latest Sensor Data - Timestamp: {sensor_timestamp}, Sensor Value: {sensor_value}")
        
        # Get MODIS data for this timestamp
        modis_data = get_modis_data(sensor_timestamp)
        
        if modis_data:
            # Insert both sensor and MODIS data into the synchronized_fire_data table
            insert_synchronized_data_to_db(sensor_timestamp, sensor_value, modis_data)
            print("Data synchronized and stored in the synchronized_fire_data table.")
        else:
            print("No MODIS data available for this timestamp.")
    
    # Sleep for the defined time window (e.g., 5 minutes) before checking for new data
    time.sleep(TIME_WINDOW.total_seconds())


#tables to store data
#     CREATE TABLE sensor_data (
#     id SERIAL PRIMARY KEY,
#     timestamp TIMESTAMP NOT NULL,
#     sensor_value INT NOT NULL
# );

# CREATE TABLE synchronized_fire_data (
#     id SERIAL PRIMARY KEY,
#     sensor_timestamp TIMESTAMP NOT NULL,
#     sensor_value INT NOT NULL,
#     fire_lat FLOAT,
#     fire_lon FLOAT,
#     confidence INT,
#     fire_radiative_power INT,
#     modis_timestamp TIMESTAMP NOT NULL
# );


