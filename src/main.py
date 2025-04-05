import sys
import os
from joblib import dump, load
from utils import load_data, encode_categorical, normalize_data, split_data, save_feature_metadata
from data_preprocessing import preprocess_data
from models.logistic_regression_model import train_and_evaluate_logistic_regression
from models.deep_learning_model import build_and_evaluate_deep_learning_model
import yaml
from logging_config import setup_logging
import logging
import requests
from tensorflow.keras.models import load_model
from visualization  import perform_shap_analysis

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages

# Define the path to the model file
model_path = "models/logistic_regression_model.pkl"

# Define the path to the DNN model file
dnn_model_path = "models/deep_learning_model"

# Function to call the API
def call_api_with_raw_input():
    # Define the API endpoint
    api_url = "http://127.0.0.1:8000/predict"

    # Example raw input data
    raw_input = {
        "total_amount": 5000.0,
        "hospital_bills": 2000.0,
        "claim_limits": 10000.0,
        "premium_amount": 1500.0,
        "treatment_expenses": 3000.0,
        "covered": "Yes",
        "claim_documents_submitted": "Yes",
        "fraud_history_approval_rejection_status": "No",
        "benefits": "Basic",
        "billing_frequency": "Monthly",
        "policy_type": "Individual",
        "provider_id": "12345",
        "patient_id": "67890",
        "doctor": "Dr. Smith",
        "hospital": "City Hospital",
        "contact_details": "123-456-7890",
        "diagnosis_report": "Mild fever",
        "discharge_summary": "Recovered",
        "prescriptions_and_bills": "Paracetamol",
        "insurance_company_name": "ABC Insurance",
        "policy_number": "POL123",
        "email": "example@example.com",
        "address": "123 Main St",
        "phone_number": "1234567890",
        "policy_name": "Health Plus",
        "procedure_codes_cpt_code": "CPT123",
        "network_partners": "Partner1",
        "bank_account": "123456789",
        "policy_holder_name": "John Doe",
        "start_date": "01-01-2023",
        "end_date": "31-12-2023",
        "renewal_date": "01-01-2024",
        "hospitalized_date": "15-01-2023"
    }

    # Log the API call
    logging.info("Sending POST request to API...")
    try:
        # Send a POST request to the API
        response = requests.post(api_url, json=raw_input)

        # Process the response
        if response.status_code == 200:
            logging.info("Logictic Regression API Response: %s", response.json())
        else:
            logging.error("Logictic Regression API Error: %s", response.text)
    except Exception as e:
        logging.error("Error while calling Logictic Regression API: %s", str(e))


# Function to call the API
def call_dnn_api_with_raw_input():
    # Define the API endpoint
    api_url = "http://127.0.0.1:8000/nn/predict"

    # Example raw input data
    raw_input = {
        "total_amount": 5000.0,
        "hospital_bills": 2000.0,
        "claim_limits": 10000.0,
        "premium_amount": 1500.0,
        "treatment_expenses": 3000.0,
        "covered": "Yes",
        "claim_documents_submitted": "Yes",
        "fraud_history_approval_rejection_status": "No",
        "benefits": "Basic",
        "billing_frequency": "Monthly",
        "policy_type": "Individual",
        "provider_id": "12345",
        "patient_id": "67890",
        "doctor": "Dr. Smith",
        "hospital": "City Hospital",
        "contact_details": "123-456-7890",
        "diagnosis_report": "Mild fever",
        "discharge_summary": "Recovered",
        "prescriptions_and_bills": "Paracetamol",
        "insurance_company_name": "ABC Insurance",
        "policy_number": "POL123",
        "email": "example@example.com",
        "address": "123 Main St",
        "phone_number": "1234567890",
        "policy_name": "Health Plus",
        "procedure_codes_cpt_code": "CPT123",
        "network_partners": "Partner1",
        "bank_account": "123456789",
        "policy_holder_name": "John Doe",
        "start_date": "01-01-2023",
        "end_date": "31-12-2023",
        "renewal_date": "01-01-2024",
        "hospitalized_date": "15-01-2023"
    }

    # Log the API call
    logging.info("Sending POST request to API...")
    try:
        # Send a POST request to the API
        response = requests.post(api_url, json=raw_input)

        # Process the response
        if response.status_code == 200:
            logging.info("DNN API Response: %s", response.json())
        else:
            logging.error("DNN API Error: %s", response.text)
    except Exception as e:
        logging.error("Error while calling DNN API: %s", str(e))


# Main script
if __name__ == "__main__":
    # Load the configuration file
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Set up logging
    setup_logging(config_path)
    logging.info("Logging is configured successfully.")

    # Access the dataset path and columns
    dataset_path = config["dataset"]["path"]
    target_column = config["columns"]["target"]
    # Standardize column names: replace spaces with underscores and convert to lowercase
    target_column = target_column.replace(" ", "_").lower()

    test_size = config["dataset"]["test_size"]
    validation_size = config["dataset"]["validation_size"]
    train_size = config["dataset"]["train_size"]

    logging.info(f"Dataset Path: {dataset_path}")
    logging.info(f"Target Column: {target_column}")


     # Check if the model file exists
    if os.path.exists(model_path):
        logging.info(f"Model file found at {model_path}. Loading the model...")
        model = load(model_path)
    else:
        logging.info(f"Model file not found at {model_path}. Training a new model...")

        # Load the dataset
        data = load_data(dataset_path)

        # Preprocess the data
        data = preprocess_data(data)
        logging.info(f"Post preprocess_data head:\n{data.head().T}")

        # Prepare X with only numeric fields
        X = data.select_dtypes(include=['number'])

        logging.info(f"X Columns: {X.columns}")

        Y = data[target_column]

        # Split the data into training, validation, and test sets
        X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X, Y, train_size, validation_size, test_size)
        # Perform SHAP analysis
        X_train_numeric = X_train.select_dtypes(include=['number'])  # Ensure numeric columns
    
        # Save feature names and target feature
        save_feature_metadata(X_train, target_column)

        # Train and evaluate the logistic regression model
        model = train_and_evaluate_logistic_regression(X, X_train, Y_train, X_val, Y_val, X_test, Y_test)

         # Perform SHAP analysis
        perform_shap_analysis(model, X_train_numeric, model_name="Logistic Regression")




        # Perform SHAP analysis
        X_train_numeric = X_train.select_dtypes(include=['number'])  # Ensure numeric columns
        X_train_array = X_train_numeric.to_numpy()  # Convert to NumPy array for deep learning models
        feature_names = X_train_numeric.columns.tolist()  # Extract feature names

    if os.path.exists(dnn_model_path):
    # Load the deep learning model
        dnn_model = load_model(dnn_model_path)  # SavedModel format
    else:    
        logging.info(f"Model file not found at {dnn_model_path}. Training a new model...")

        # Load the dataset
        data = load_data(dataset_path)

        # Preprocess the data
        data = preprocess_data(data)
        logging.info(f"Post preprocess_data head:\n{data.head().T}")

        # Prepare X with only numeric fields
        X_dnn = data.select_dtypes(include=['number'])

        logging.info(f"X Columns: {X_dnn.columns}")

        Y = data[target_column]

        # Split the data into training, validation, and test sets
        X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X_dnn, Y, train_size, validation_size, test_size)
        # Perform SHAP analysis
        X_train_numeric = X_train.select_dtypes(include=['number'])  # Ensure numeric columns

        logging.info(f"DNN X_train {X_train.shape}")

        # Train and evaluate the deep learning model
        dnn_model, history = build_and_evaluate_deep_learning_model(X_dnn, X_train, Y_train, X_val, Y_val, X_test, Y_test)
        dnn_model.save("models/deep_learning_model")  # SavedModel format
        # Save the trained model        
        logging.info(f"Deep Learning Model saved to models/deep_learning_model")

        #perform_shap_analysis(dnn_model, X_train_array, model_name="Deep Learning Model", feature_names=feature_names)
    
    # Call the API
    call_api_with_raw_input()

    call_dnn_api_with_raw_input()