from datetime import datetime
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
import logging
import sys
import os
import json
import os
import yaml
import os
import glob


# Add the src directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_data(filepath):
    import pandas as pd
    return pd.read_csv(filepath)

def save_results(results, filepath):
    import pandas as pd
    results.to_csv(filepath, index=False)


def encode_categorical(df, columns):
    """
    Encodes categorical columns in a DataFrame into numerical codes.

    This function converts the specified categorical columns in the given
    DataFrame into numerical codes using pandas' `astype('category').cat.codes`.
    It modifies the DataFrame in place and returns it.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        columns (list of str): A list of column names in the DataFrame to be encoded.

    Returns:
        pandas.DataFrame: The DataFrame with the specified columns encoded as numerical codes.

    Example:
        >>> import pandas as pd
        >>> data = {'Category': ['A', 'B', 'A', 'C']}
        >>> df = pd.DataFrame(data)
        >>> encode_categorical(df, ['Category'])
        >>> logging.info(df)
           Category
        0         0
        1         1
        2         0
        3         2
    """
    for col in columns:
        df[col] = df[col].astype('category').cat.codes
    return df

def normalize_data(df):
    import pandas as pd  # Add this import
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df.select_dtypes(include=['number'])), columns=df.select_dtypes(include=['number']).columns )

def split_data(X, Y, train_size, val_size, test_size, random_state=42):
    """
    Splits the dataset into training, validation, and test sets.

    Parameters:
        X (pd.DataFrame): The feature matrix.
        Y (pd.Series or np.array): The target variable.
        train_size (int): Number of samples for the training set.
        val_size (int): Number of samples for the validation set.
        test_size (int): Number of samples for the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_val, X_test, Y_train, Y_val, Y_test)
    """
    from sklearn.model_selection import train_test_split

    # Ensure the total number of rows matches the required split sizes
    total_size = train_size + val_size + test_size
    if len(X) < total_size:
        raise ValueError(f"The dataset must have at least {total_size} rows to split into "
                         f"{train_size} training, {val_size} validation, and {test_size} test samples.")

    # Split the data into training and remaining
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, train_size=train_size, random_state=random_state, stratify=Y)

    # Split the remaining data into validation and test
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, train_size=val_size, random_state=random_state, stratify=Y_temp)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def save_feature_metadata(X_train, target_feature, metadata_path="models/feature_metadata.json"):
    """
    Save feature names and target feature to a file.

    Parameters:
        X_train (pd.DataFrame): Training data containing feature columns.
        target_feature (str): Name of the target feature.
        metadata_path (str): Path to save the metadata file.
    """
    import json

    # Extract feature names
    feature_names = X_train.columns.tolist()

    # Ensure the directory exists
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

    with open(metadata_path, "w") as file:
        json.dump({"features": feature_names}, file, indent=4)

    print("Feature names saved to feature_metadata.json successfully!", metadata_path)

def load_expected_headers():
    config = load_config()
    return set(config["expected_headers"])


def load_config():
    """Loads application configuration from YAML."""
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


import os
import shutil
import pandas as pd
from datetime import datetime
import logging

def merge_csv(file_path):
    config = load_config()
    
    """Merges uploaded CSV into the main training dataset while taking a backup."""
    
    # Move up one level from `src` to reach project root
    current_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logging.info(f"Current path root: {current_path}")

    master_data = os.path.join(current_path, config["storage"]["master_csv"])
    logging.info(f"Master data path: {master_data}")

    # Ensure the backup folder is defined before checking the master file
    backup_folder = os.path.join(current_path, config["storage"]["backup"]["folder"])
    os.makedirs(backup_folder, exist_ok=True)
    logging.info(f"Backup folder: {backup_folder}")

    # Backup the existing master file before merging
    if os.path.exists(master_data):
        # Generate a unique backup filename with a timestamp
        backup_file = os.path.join(
            backup_folder, f"train_data_backup_with_master_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

    # Use shutil.copy2 to preserve metadata
    shutil.copy2(master_data,backup_file)
    logging.info(f"Merged filed saved successfully at: {backup_file}")

    # Load the new data
    uploaded_df = pd.read_csv(file_path)
    logging.info(f"Uploaded data shape: {uploaded_df.shape}")

    # Merge with existing data
    if os.path.exists(backup_folder):
        exiting_df = pd.read_csv(backup_file)
        combined_df = pd.concat([exiting_df, uploaded_df], ignore_index=True)
    else:
        combined_df = uploaded_df  # If no previous file exists, use the new upload

    # Save the updated master file
    combined_df.to_csv(backup_file, index=False)
    logging.info(f"merged data set shape: {combined_df.shape}")
    logging.info(f"Type of df before head(): {type(combined_df)}")
    logging.info("Master dataset updated for retraining!")



def get_latest_file(folder_path, file_extension=".csv"):
    """
    Get the latest file from a specified folder based on modification time.

    Parameters:
        folder_path (str): Path to the folder.
        file_extension (str): File extension to filter (default: "*" for all files).

    Returns:
        str: Path of the latest file.
    """

    files = glob.glob(os.path.join(folder_path, f"*.{file_extension}"))  # List all matching files
    
    if not files:  # If no files are found
        logging.info(f"No file present on folder_path  : {folder_path}") 
        return None

    latest_file = max(files, key=os.path.getmtime)  # Find the most recently modified file
    logging.info(f"Latest file found on folder_path  : {folder_path, latest_file}")  
    return latest_file


