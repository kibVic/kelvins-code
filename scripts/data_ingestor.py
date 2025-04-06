import pandas as pd
from sqlalchemy import create_engine

# Path to the CSV file
csv_file_path = 'prediction_dataset.csv'

# Read the CSV into a pandas DataFrame
df = pd.read_csv(csv_file_path)

# Show the first few rows of the DataFrame to verify data
print(df.head())

# PostgreSQL connection details
postgres_user = 'root'  
postgres_password = 'root' 
postgres_host = '172.18.0.2'  
postgres_port = '5432' 
postgres_db = 'fire_detection' 

# Create the connection URL for SQLAlchemy
connection_url = f'postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}'

# Create a SQLAlchemy engine
engine = create_engine(connection_url)

# Load the DataFrame into PostgreSQL
df.to_sql('prediction_data', engine, if_exists='replace', index=False)

print("Data loaded into PostgreSQL successfully!")
