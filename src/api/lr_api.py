import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load
from src.data_preprocessing import preprocess_raw_input

# Add the src directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load the trained model
model_path = "models/logistic_regression_model.pkl"
model = load(model_path)

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
    missing_columns = set(model.feature_names_in_) - set(processed_df.columns)
    for col in missing_columns:
        processed_df[col] = 0  # Add missing columns with default values

    # Reorder columns to match the model's training data
    processed_df = processed_df[model.feature_names_in_]

    # Make predictions
    prediction = model.predict(processed_df)
    prediction_proba = model.predict_proba(processed_df)

    # Return the prediction and probabilities
    return {
        "prediction": int(prediction[0]),
        "probability": prediction_proba[0].tolist()
    }