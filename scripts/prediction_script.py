# import pandas as pd
import matplotlib.pyplot as plt
import psycopg2
import logging
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Set up logging for error handling
logging.basicConfig(level=logging.INFO)

# Define the data loading function using psycopg2
def load_data_from_postgres(db_params, query):
    try:
        # Establish a connection to the database
        conn = psycopg2.connect(**db_params)
        
        # Load data into pandas DataFrame
        df = pd.read_sql(query, conn)
        
        # Close the connection
        conn.close()
        
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

# Define data preprocessing function
def preprocess_data(df):
    # Convert timestamps to datetime format
    df['sensor_timestamp'] = pd.to_datetime(df['sensor_timestamp'])
    df['modis_timestamp'] = pd.to_datetime(df['modis_timestamp'])

    # Handle missing values (fill NaNs with median or drop)
    df['fire_radiative_power'].fillna(df['fire_radiative_power'].median(), inplace=True)
    df.dropna(subset=['modis_timestamp'], inplace=True)

    # Optional: Handle duplicates
    df.drop_duplicates(inplace=True)

    # Optional: Scale features (if required)
    scaler = StandardScaler()
    df[['sensor_value', 'fire_lat', 'fire_lon', 'confidence']] = scaler.fit_transform(df[['sensor_value', 'fire_lat', 'fire_lon', 'confidence']])

    # Create additional time-based features
    df['hour_of_day'] = df['sensor_timestamp'].dt.hour
    df['day_of_week'] = df['sensor_timestamp'].dt.dayofweek

    # Drop unnecessary columns
    df_model = df.drop(columns=['sensor_timestamp', 'modis_timestamp'])
    
    # Define features (X) and target (y)
    X = df_model.drop(columns='fire_detected')
    y = df_model['fire_detected']

    return X, y

# Define data splitting function
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

# Define the model training function with Grid Search for hyperparameter tuning
def train_model_with_tuning(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'class_weight': ['balanced', None]  # Handle class imbalance by adjusting class weights
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
    
    try:
        grid_search.fit(X_train, y_train)
        logging.info("Best Parameters: %s", grid_search.best_params_)
        return grid_search.best_estimator_
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

# Define the model evaluation function
def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        
        # Calculate additional metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-Score: {f1}")
        print(f"AUC-ROC: {roc_auc}")
        
    except Exception as e:
        logging.error(f"Error evaluating model: {e}")
        raise

# Define a function to make predictions on new data
def predict_fire(model, new_data):
    try:
        prediction = model.predict(new_data)[0]
        return "Fire Detected!" if prediction == 1 else "No Fire Detected"
    except Exception as e:
        logging.error(f"Error making prediction: {e}")
        raise

# Define the function to plot feature importance
def plot_feature_importance(model, X):
    try:
        feature_importance = pd.Series(model.feature_importances_, index=X.columns)
        feature_importance.sort_values().plot(kind='barh')
        plt.title("Feature Importance")
        plt.show()
    except Exception as e:
        logging.error(f"Error plotting feature importance: {e}")
        raise

# Function to plot learning curves
def plot_learning_curves(model, X, y):
    try:
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
        
        train_mean = train_scores.mean(axis=1)
        test_mean = test_scores.mean(axis=1)
        
        plt.plot(train_sizes, train_mean, label="Training Score", color="blue")
        plt.plot(train_sizes, test_mean, label="Test Score", color="red")
        plt.xlabel("Training Size")
        plt.ylabel("Score")
        plt.title("Learning Curve")
        plt.legend()
        plt.show()
        
    except Exception as e:
        logging.error(f"Error plotting learning curves: {e}")
        raise

# Main function to run the entire process
def main():
    db_params = {
        'dbname': 'fire_detection',  
        'user': 'root',    
        'password': 'root', 
        'host': 'localhost',       
        'port': '5432'          
    }
    query = "SELECT * FROM prediction_dataset;"  
    
    try:
        # Load data from PostgreSQL
        df = load_data_from_postgres(db_params, query)

        # Data Preprocessing
        X, y = preprocess_data(df)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Train the model with hyperparameter tuning
        model = train_model_with_tuning(X_train, y_train)

        # Evaluate the model
        evaluate_model(model, X_test, y_test)

        # Example new data for prediction
        new_data = pd.DataFrame([{
            'sensor_value': 530,
            'fire_lat': 1.2921,
            'fire_lon': 36.8219,
            'confidence': 82,
            'fire_radiative_power': 45.7
        }])

        # Predict and print result
        prediction = predict_fire(model, new_data)
        print(prediction)

        # Plot feature importance
        plot_feature_importance(model, X)

        # Plot learning curves
        plot_learning_curves(model, X, y)
        
        # Get and print feature importances
        feature_importance = pd.Series(model.feature_importances_, index=X.columns)
        sorted_feature_importance = feature_importance.sort_values(ascending=False)
        print("Feature Importance:")
        print(sorted_feature_importance)
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
