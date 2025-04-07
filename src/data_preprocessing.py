import sys
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import yaml
import logging
import math
from joblib import dump, load
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Dynamically handle imports based on execution context
try:
    # When running from `main.py`
    from utils import load_data, encode_categorical, normalize_data, split_data, load_config
    from anomaly_detection import AnomalyDetector
    from models.deep_learning_model import build_and_evaluate_deep_learning_model
    from models.logistic_regression_model import train_and_evaluate_logistic_regression
    from visualization  import perform_shap_analysis
except ModuleNotFoundError:
    # When running from FastAPI (e.g., `uvicorn`)
    from src.utils import load_data, encode_categorical, normalize_data, split_data, load_config
    from src.anomaly_detection import AnomalyDetector
    from src.models.deep_learning_model import build_and_evaluate_deep_learning_model
    from src.models.logistic_regression_model import train_and_evaluate_logistic_regression
    from src.visualization  import perform_shap_analysis

# Add the src directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

config= load_config()
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):

    logging.info(f"Type of df before head(): {type(df)}")

    logging.info(f"df.shape:\n{df.shape}")
    # Standardize column names: replace spaces and slashes with underscores, and convert to lowercase
    df.columns = df.columns.str.replace(' ', '_').str.replace('/', '_').str.lower()
    logging.info(f"Standardize column names: replace spaces and slashes with underscores, and convert to lowercase:\n{df.columns}")


    # Handle missing values
    df.ffill(inplace=True)
    logging.info(f"after removing missing values:\n{df.columns}")

    # Remove duplicates
    df = df.drop_duplicates(keep='first')

        
    # Display categorical features
    categorical_features = []

    # Extract configuration values
    nominal_columns = config["columns"]["nominal"]
    # Standardize column names: replace spaces with underscores and convert to lowercase
    nominal_columns = [col.replace(' ', '_').replace('/', '_').lower() for col in nominal_columns]
    date_format = config["columns"]["date_format"]
    columns_to_encode = config["columns"]["labeled"]
    columns_to_encode = [col.replace(' ', '_').replace('/', '_').lower() for col in columns_to_encode]

    date_columns = config["columns"]["date_columns"]
    date_columns = [col.replace(' ', '_').replace('/', '_').lower() for col in date_columns]
    glove_path = config["glove"]["path"]

    logging.info(f"Nominal columns: {nominal_columns}")
    logging.info(f"Date format: {date_format}")
    logging.info(f"Columns to encode: {columns_to_encode}")
    logging.info(f"Date columns: {date_columns}")
    logging.info(f"GloVe path: {glove_path}")


    # Convert object columns to category
    #df = convert_object_columns_to_category(df, categorical_features, date_format)


    # Encode selected categorical columns
    #categorical_columns = df.select_dtypes(include=['category']).columns.tolist()


    if columns_to_encode:
        df =encode_selected_columns(df, columns_to_encode)
    else:
        logging.info("No categorical columns to encode.")


    #glove_path = r'data//glove/glove.6B.50d.txt'  # Update with the actual path to GloVe embeddings
    df = convert_nominal_to_numeric_with_glove_single_value(df, nominal_columns, glove_path, aggregation='magnitude')

    
    # Convert date columns to numeric
    df = convert_date_columns_to_numeric(df, date_columns, date_format)

    # Initialize the anomaly detector
    anomaly_detector = AnomalyDetector(contamination=0.1)

    # Fit the model and detect anomalies
    #anomaly_detector.fit(data)
    df_with_anomalies = anomaly_detector.detect_anomalies(df)


    # logging.info the detected anomalies
    logging.info("Detected anomalies:")
    logging.info(df_with_anomalies[df_with_anomalies['any_anomaly'] == 1].head())
     
    logging.info(f"final DF:\n{df_with_anomalies}") 

    # Normalize the data
    #df_with_anomalies_normalized = normalize_data(df_with_anomalies) 


    return df_with_anomalies


def display_categorical_features(df):
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    logging.info("Categorical Features:")
    logging.info(categorical_columns)
    return categorical_columns

def convert_object_columns_to_category(df, categorical_features, date_format=None):
    """
    Converts 'object' type columns to 'category' type, excluding specified date-like columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        categorical_features (list): List of column names to exclude as date-like columns.
        date_format (str, optional): The expected date format (e.g., '%d-%m-%Y').

    Returns:
        pd.DataFrame: The updated DataFrame with non-date 'object' columns converted to 'category'.
    """
    # Select only 'object' type columns
    features = df.select_dtypes(include=['object'])

    logging.info(f"Inside convert_object_columns_to_category categorical_features:\n{categorical_features}")
    for col in features.columns:
        try:
            # Try converting to datetime to identify date columns
            if date_format:
                pd.to_datetime(df[col], format=date_format, errors='coerce')
            else:
                pd.to_datetime(df[col], errors='coerce')
            categorical_features.append(col)
        except (ValueError, TypeError):
            continue

    logging.info(f"Inside convert_object_columns_to_category categorical_features:\n{categorical_features}")
    
    # Exclude identified date columns from the conversion process
    non_date_features = [col for col in features.columns if col not in categorical_features]

    logging.info(f"Inside convert_object_columns_to_category categorical_features:\n{non_date_features}")
    
    # Convert non-date 'object' columns to 'category' type
    for col in non_date_features:
        df[col] = df[col].astype('category')

    # Display the updated DataFrame information
    logging.info("Updated DataFrame info after converting 'object' columns to 'category':")
    df.info()

    return df

def encode_selected_columns(df, columns_to_encode):
    """
    Applies label encoding to the specified columns in the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns_to_encode (list): List of column names to encode.

    Returns:
        pd.DataFrame: The updated DataFrame with encoded columns.
    """
    # Create a copy of the original DataFrame
    #df = df.copy()
    logging.info(f"Columns to encode: {columns_to_encode}")
    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Apply label encoding to selected columns
    for column in columns_to_encode:
        if column in df.columns:
            df[column + '_encoded'] = label_encoder.fit_transform(df[column])
        else:
            logging.info(f"Warning: Column '{column}' not found in the DataFrame.")

    # Display the updated DataFrame information
    logging.info(f"Updated DataFrame columns after label encoding: {df.columns.tolist()}")
    
    return df

import numpy as np


def load_glove_embeddings(glove_path):
    """
    Loads GloVe embeddings from the specified file.

    Parameters:
        glove_path (str): Path to the GloVe embeddings file.

    Returns:
        dict: A dictionary where keys are words and values are their GloVe vectors.
    """
    logging.info("Loading GloVe embeddings...")
    glove_embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype='float32')
            glove_embeddings[word] = vector
    logging.info("GloVe embeddings loaded.")
    return glove_embeddings

def get_column_embeddings_single_value(df, column, glove_embeddings, aggregation='magnitude'):
    """
    Computes a single numeric value for GloVe embeddings for a specific column in the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column to compute embeddings for.
        glove_embeddings (dict): The preloaded GloVe embeddings.
        aggregation (str): The aggregation method to reduce the vector to a single value.
                           Options: 'magnitude', 'mean', 'sum'.

    Returns:
        pd.Series: A pandas Series containing the aggregated GloVe embedding value for each row.
    """
    if column not in df.columns:
        logging.info(f"Warning: Column '{column}' not found in the DataFrame.")
        return pd.Series([0] * len(df), index=df.index)

    numeric_values = []
    for text in df[column].astype(str):
        words = text.split()  # Split text into words
        word_vectors = [glove_embeddings[word] for word in words if word in glove_embeddings]
        if word_vectors:
            # Compute the mean vector for the text
            mean_vector = np.mean(word_vectors, axis=0)
            # Aggregate the vector into a single numeric value
            if aggregation == 'magnitude':
                # Compute the magnitude (Euclidean norm) of the vector
                numeric_value = np.linalg.norm(mean_vector)
            elif aggregation == 'mean':
                # Compute the mean of the vector values
                numeric_value = np.mean(mean_vector)
            elif aggregation == 'sum':
                # Compute the sum of the vector values
                numeric_value = np.sum(mean_vector)
            else:
                raise ValueError(f"Unsupported aggregation method: {aggregation}")
        else:
            # If no words have embeddings, use 0 as the default value
            numeric_value = 0
        numeric_values.append(numeric_value)

    # Return the aggregated values as a pandas Series
    return pd.Series(numeric_values, index=df.index)

def convert_nominal_to_numeric_with_glove_single_value(df, columns_to_convert, glove_path, aggregation='magnitude'):
    """
    Converts nominal fields to numeric representations using GloVe embeddings and stores them as single numeric values.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns_to_convert (list): List of nominal columns to convert.
        glove_path (str): Path to the GloVe embeddings file.
        aggregation (str): The aggregation method to reduce the vector to a single value.

    Returns:
        pd.DataFrame: The updated DataFrame with new numeric columns for the specified nominal fields.
    """
    # Load GloVe embeddings once
    glove_embeddings = load_glove_embeddings(glove_path)

    # Process each column and add the new numeric column to the DataFrame
    for column in columns_to_convert:
        numeric_column = get_column_embeddings_single_value(df, column, glove_embeddings, aggregation)
        # Standardize the column name before adding the new GloVe column
        standardized_column_name = column.replace(' ', '_').lower()
        df[f"{standardized_column_name}_glove"] = numeric_column  # Add the new column to the DataFrame

    # Display the updated DataFrame information
    logging.info("Updated DataFrame after glove encoding:")
    return df

def convert_date_columns_to_numeric(df, date_columns, date_format=None):
    """
    Converts date columns in the DataFrame to numeric format (e.g., days since epoch).

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        date_columns (list): List of column names containing date values.
        date_format (str, optional): The expected date format (e.g., '%d-%m-%Y').

    Returns:
        pd.DataFrame: The updated DataFrame with date columns converted to numeric.
    """
    for column in date_columns:
        if column in df.columns:
            try:
                # Convert the column to datetime format with the specified format
                if date_format:
                    df[column] = pd.to_datetime(df[column], format=date_format, errors='coerce')
                else:
                    df[column] = pd.to_datetime(df[column], errors='coerce')

                # Convert datetime to numeric (e.g., days since epoch)
                df[column + '_numeric'] = (df[column] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1d')
            except Exception as e:
                logging.info(f"Error converting column '{column}': {e}")
        else:
            logging.info(f"Warning: Column '{column}' not found in the DataFrame.")

    return df


def preprocess_raw_input(raw_input):
    import os
    import yaml

    # Ensure raw_input is a flat dictionary
    if isinstance(raw_input, dict):
        # Flatten nested dictionaries if necessary
        raw_input = {k: v for k, v in raw_input.items()}
    else:
        raise ValueError("Input data must be a dictionary.")

    sanitize_input(raw_input)

    # Convert raw input to DataFrame
    df_raw = pd.DataFrame([raw_input])

    df_raw = preprocess_data(df_raw)
    df_raw = df_raw.select_dtypes(include=['number'])
    logging.info(f"Processed DataFrame:\n{df_raw.head().T}")
    return df_raw

# Replace invalid float values in the raw_input dictionary
def sanitize_input(data):
    for key, value in data.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            logging.warning(f"Replacing invalid float value for key '{key}': {value}")
            data[key] = 0  # Replace with a default value

def retrain_lr_model():
    """Retrains the ML model with updated dataset."""

    # Move up one level from `src` to reach project root
    current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    logging.info(f"Current path root: {current_path}")
    master_data = os.path.join(current_path, config["storage"]["master_csv"])
    logging.info(f"Master data path: {master_data}")
    # Load latest data
    df = pd.read_csv(master_data)
    logging.info(f"Data loaded from {master_data}")
    logging.info(f"Type of df before head(): {type(df)}")    
    logging.info(f"df: {df.shape}")

    df = preprocess_data(df)
    # Preprocess features and labels
    X = df.drop(columns=[config["columns"]["target"]])  # Adjust as needed
    X = X.select_dtypes(include=['number'])
    X_lr = X.copy()
    Y = df[config["columns"]["target"]]  # Adjust as needed

    # Split the data into training, validation, and test sets
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X, Y, config["data_split"]["test_size"], config["data_split"]["validation_size"], config["data_split"]["test_size"])
    print("Checking for NaN values...")
    print(X_train.isnull().sum())  # Verify missing values
    print(X_train.describe())  # Look for unexpected values

    # Ensure no NaNs remain
    X_train.fillna(0, inplace=True)  # Replace NaNs with zeros
    X_val.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)



    # Train new model
    model = train_and_evaluate_logistic_regression(X_lr, X_train, Y_train, X_val, Y_val, X_test, Y_test)
    logging.info("Model Logistic Regression model retrained & saved!")

def trigger_model_retraining():
    """Triggers automated model retraining."""
    retrain_lr_model()
    retrain_dnn_model()
    logging.info("Model successfully retrained after new upload!")            



def retrain_dnn_model():
    """Retrains the ML model with updated dataset."""

    # Move up one level from `src` to reach project root
    current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    logging.info(f"Current path root: {current_path}")
    master_data = os.path.join(current_path, config["storage"]["master_csv"])
    logging.info(f"Master data path: {master_data}")
    # Load latest data
    df = pd.read_csv(master_data)
    logging.info(f"Data loaded from {master_data}")
    logging.info(f"Type of df before head(): {type(df)}")    
    logging.info(f"df: {df.shape}")

    df = preprocess_data(df)
    # Preprocess features and labels
    X = df.drop(columns=[config["columns"]["target"]])  # Adjust as needed
    X = X.select_dtypes(include=['number'])
    X_nn = X.copy()
    Y = df[config["columns"]["target"]]  # Adjust as needed

    # Split the data into training, validation, and test sets
    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X, Y, config["data_split"]["test_size"], config["data_split"]["validation_size"], config["data_split"]["test_size"])
    
    # Train new model
    dnn_model, history = build_and_evaluate_deep_learning_model(X_nn, X_train, Y_train, X_val, Y_val, X_test, Y_test)
    dnn_model.save("models/deep_learning_model")  # SavedModel format
    print("Model Deep Learing retrained & saved!")    