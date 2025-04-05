import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load
from src.data_preprocessing import preprocess_raw_input
from tensorflow.keras.models import load_model
import logging
from src.logging_config import setup_logging
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import yaml

# Add the src directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load the configuration file
config_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Load the scaler
scaler_path = config["dnn_scaler_path"]
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler not found at {scaler_path}. Please train the model and save the scaler.")
with open(scaler_path, "rb") as f:
    loaded_data = pickle.load(f)

scaler = loaded_data["scaler"]
feature_names = loaded_data["feature_names"]

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level to INFO
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",  # Include the file name in the log format
    handlers=[
        logging.StreamHandler()  # Output logs to the console
    ]
)



# Load the trained model
model_path = config["lr_model_path"]
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Logistic Regression model not found at {model_path}. Please train and save the model.")

model_data = load(model_path)
model = model_data["model"]
lr_feature_names = model_data["feature_names"]
lr_scaler = model_data["scaler"]


# Load the trained model
nn_model_path = config["dnn_model_path"]

if not os.path.exists(nn_model_path):
    raise FileNotFoundError(f"Deep Learning model not found at {nn_model_path}. Please train and save the model.")

nn_model = load_model(nn_model_path)

# Define the input schema using Pydantic
class PredictionInput(BaseModel):
    total_amount: float
    hospital_bills: float
    claim_limits: float
    premium_amount: float
    treatment_expenses: float
    covered: str
    claim_documents_submitted: str
    fraud_history_approval_rejection_status: str
    benefits: str
    billing_frequency: str
    policy_type: str
    provider_id: str
    patient_id: str
    doctor: str
    hospital: str
    contact_details: str
    diagnosis_report: str
    discharge_summary: str
    prescriptions_and_bills: str
    insurance_company_name: str
    policy_number: str
    email: str
    address: str
    phone_number: str
    policy_name: str
    procedure_codes_cpt_code: str
    network_partners: str
    bank_account: str
    policy_holder_name: str
    start_date: str
    end_date: str
    renewal_date: str
    hospitalized_date: str

# Initialize FastAPI
app = FastAPI()


# Define the root endpoint
@app.get("/")
def read_root():
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    return {"message": "Logistic Regression Model API is running!"}

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: PredictionInput):
    # Convert input_data (Pydantic model) to a dictionary
    raw_input = input_data.dict()

    # Preprocess the raw input
    processed_df = preprocess_raw_input(raw_input)

    # Ensure the processed data has the same columns as the model expects
    missing_columns = set(lr_feature_names) - set(processed_df.columns)
    for col in missing_columns:
        processed_df[col] = 0  # Add missing columns with default values

    # Reorder columns to match the model's training data
    processed_df = processed_df[lr_feature_names]

    # Normalize the input using the trained scaler
    processed_df = lr_scaler.transform(processed_df)

    logging.info(f"Normalized input DataFrame:\n{processed_df}")

    # Make predictions
    prediction = model.predict(processed_df)
    prediction_proba = model.predict_proba(processed_df)

    # Return the prediction and probabilities
    return {
        "prediction": int(prediction[0]),
        "probability": prediction_proba[0].tolist()
    }

@app.post("/nn/predict")
def nn_predict(input_data: PredictionInput):
    # Convert input_data (Pydantic model) to a dictionary
    raw_input = input_data.dict()

    # Preprocess the raw input
    processed_df = preprocess_raw_input(raw_input)

    logging.info(f"Processed DataFrame before conversion:\n{processed_df.head()}")
    logging.info(f"Processed DataFrame dtypes:\n{processed_df.dtypes}")

    logging.info(f"feature names:\n{feature_names}")

    # Ensure all data is numeric
    processed_df = processed_df.apply(pd.to_numeric, errors='coerce')
    processed_df.fillna(0, inplace=True)

    logging.info(f"Processed DataFrame for NN: {processed_df.shape}")

    # Ensure the processed data has the same columns as the deep learning model expects
    missing_columns = set(feature_names) - set(processed_df.columns)
    for col in missing_columns:
        processed_df[col] = 0  # Add missing columns with default values

    # Reorder columns to match the scaler's training data
    processed_df = processed_df[feature_names]

    # Check if `processed_df` has valid data before assigning to `processed_array`
    if processed_df.empty:
        logging.error("Processed DataFrame is empty. Cannot proceed.")
        return {"error": "Processed data is empty"}

    # Convert DataFrame to NumPy array
    processed_array = processed_df.to_numpy()

    # Ensure `processed_array` is properly assigned before conversion
    if processed_array is None or processed_array.size == 0:
        logging.error("Processed array is None or empty.")
        return {"error": "Processed array is empty"}

    # Convert to DataFrame with feature names before normalization
    processed_array_df = pd.DataFrame(processed_array, columns=feature_names)

    # Normalize the input using the trained scaler
    processed_array_normalized = scaler.transform(processed_array_df)

    logging.info(f"Normalized input array for NN model:\n{processed_array_normalized}")

    # Make predictions
    prediction_proba = nn_model.predict(processed_array_normalized)
    prediction_proba = np.nan_to_num(prediction_proba)  # Convert NaN to 0
    prediction = (prediction_proba > 0.5).astype(int)  # Convert probabilities to binary predictions

    # Debugging logs for model output
    logging.info(f"Prediction probabilities before sanitization: {prediction_proba}")
    logging.info(f"Prediction before sanitization: {prediction}")

    # Sanitize the model output
    prediction_proba[0] = sanitize_output(prediction_proba[0])
    prediction = sanitize_output(prediction)

    # Prepare the response
    response = {
        "prediction": int(prediction[0][0]),
        "probability": prediction_proba[0].tolist()
    }

    # Validate the response
    try:
        validate_response(response)
    except ValueError as e:
        logging.error(f"Validation error in response: {e}")
        return {"error": "Invalid response data"}

    # Return the sanitized and validated response
    return response


# Validate the model output
def validate_output(data):
    if isinstance(data, (list, np.ndarray)):
        for value in data:
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                raise ValueError(f"Invalid float value in model output: {value}")
    elif isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        raise ValueError(f"Invalid float value in model output: {data}")
    

# Replace invalid float values in the model output
def sanitize_output(data):
    if isinstance(data, (list, np.ndarray)):
        return [0 if (isinstance(value, float) and (math.isnan(value) or math.isinf(value))) else value for value in data]
    elif isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return 0
    return data

# Validate the response dictionary
def validate_response(response):
    for key, value in response.items():
        if isinstance(value, (list, np.ndarray)):
            for v in value:
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    raise ValueError(f"Invalid float value in response for key '{key}': {v}")
        elif isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError(f"Invalid float value in response for key '{key}': {value}")