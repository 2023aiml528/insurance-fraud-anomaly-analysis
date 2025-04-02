import sys
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import yaml
import logging


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):

    # Load the configuration file
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Handle missing values
    df.ffill(inplace=True)
    
    # Display categorical features
    categorical_features = []

    nominal_columns = config["columns"]["nominal"]
    date_format = config["columns"]["date_format"]
    columns_to_encode = config["columns"]["labeled"]
    date_columns = config["columns"]["date_columns"]
    glove_path = config["glove_path"]["glove_path"]

    logging.info("Nominal columns:", nominal_columns) 
    logging.info("Date format:", date_format)
    logging.info("Columns to encode:", columns_to_encode)
    logging.info("Date columns:", date_columns)
    logging.info("GloVe path:", glove_path)


    # Convert object columns to category
    df = convert_object_columns_to_category(df, categorical_features, date_format)


    # Encode selected categorical columns
    categorical_columns = df.select_dtypes(include=['category']).columns.tolist()


    if columns_to_encode:
        df =encode_selected_columns(df, columns_to_encode)
    else:
        logging.info("No categorical columns to encode.")


    #glove_path = r'data//glove/glove.6B.50d.txt'  # Update with the actual path to GloVe embeddings
    df = convert_nominal_to_numeric_with_glove_single_value(df, nominal_columns, glove_path, aggregation='magnitude')

    
    # Convert date columns to numeric
    df = convert_date_columns_to_numeric(df, date_columns, date_format)


     
    return df


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

    logging.info("Inside convert_object_columns_to_category features:\n", features.columns)
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

    logging.info("Inside convert_object_columns_to_category categorical_features:\n", categorical_features)

    # Exclude identified date columns from the conversion process
    non_date_features = [col for col in features.columns if col not in categorical_features]

    logging.info("Inside convert_object_columns_to_category non_date_features:\n", non_date_features)

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

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Apply label encoding to selected columns
    for column in columns_to_encode:
        if column in df.columns:
            df[column + '_encoded'] = label_encoder.fit_transform(df[column])
        else:
            logging.info(f"Warning: Column '{column}' not found in the DataFrame.")

    # Display the updated DataFrame information
    logging.info("Updated DataFrame after label encoding:")
    df.info()

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
        df[f"{column}_glove"] = numeric_column  # Add the new column to the DataFrame

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
