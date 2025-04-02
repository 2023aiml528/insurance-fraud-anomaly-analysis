import sys
import os
from joblib import dump, load
from utils import load_data, encode_categorical, normalize_data, split_data
from data_preprocessing import preprocess_data
from models.logistic_regression_model import train_and_evaluate_logistic_regression
from utils import perform_shap_analysis
import yaml
from logging_config import setup_logging
import logging
import requests

# Define the path to the model file
model_path = "models/logistic_regression_model.pkl"

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
            logging.info("API Response: %s", response.json())
        else:
            logging.error("API Error: %s", response.text)
    except Exception as e:
        logging.error("Error while calling API: %s", str(e))


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
    nominal_columns = config["columns"]["nominal"]

    test_size = config["dataset"]["test_size"]
    validation_size = config["dataset"]["validation_size"]
    train_size = config["dataset"]["train_size"]

    logging.info(f"Dataset Path: {dataset_path}")
    logging.info(f"Target Column: {target_column}")
    logging.info(f"Nominal Columns: {nominal_columns}")

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

        # Train and evaluate the logistic regression model
        model = train_and_evaluate_logistic_regression(X_train, Y_train, X_val, Y_val, X_test, Y_test)

        # Save the trained model
        os.makedirs("models", exist_ok=True)  # Ensure the models directory exists
        dump(model, model_path)
        logging.info(f"Model saved to {model_path}.")

    # Call the API
    call_api_with_raw_input()