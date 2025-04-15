from datetime import datetime
import sys
import os
from fastapi import FastAPI, BackgroundTasks,File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pydantic import BaseModel
import pandas as pd
from joblib import load
from src.data_preprocessing import preprocess_raw_input, trigger_model_retraining
from tensorflow.keras.models import load_model
import logging
from src.logging_config import setup_logging
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import yaml
from src.utils import load_config, merge_csv
import webbrowser

# Add the src directory to PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Initialize FastAPI
app = FastAPI()

# Global variable to track training status
training_status = {"status": "Idle", "progress": 0}

# Load configuration
config = load_config()

# Configure logging
log_folder = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")), "logs")
os.makedirs(log_folder, exist_ok=True)
log_file_path = os.path.join(log_folder, "app.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - Line: %(lineno)d - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

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

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Logistic Regression Model API is running!"}

# Prediction endpoint for Logistic Regression
@app.post("/predict")
def predict(input_data: PredictionInput):
    model_path = config["models"]["logistic_regression"]
    if not os.path.exists(model_path):
        raise HTTPException(status_code=400, detail="Logistic Regression model not found.")
    model_data = load(model_path)
    model = model_data["model"]
    lr_feature_names = model_data["feature_names"]
    lr_scaler = model_data["scaler"]
    raw_input = input_data.dict()
    processed_df = preprocess_raw_input(raw_input)
    missing_columns = set(lr_feature_names) - set(processed_df.columns)
    for col in missing_columns:
        processed_df[col] = 0
    processed_df = processed_df[lr_feature_names]
    processed_df = lr_scaler.transform(processed_df)
    prediction = model.predict(processed_df)
    prediction_proba = model.predict_proba(processed_df)
    return {"prediction": int(prediction[0]), "probability": prediction_proba[0].tolist()}

# Prediction endpoint for Neural Network
@app.post("/nn/predict")
def nn_predict(input_data: PredictionInput):
    scaler_path = config["models"]["scaler"]
    nn_model_path = config["models"]["deep_learning"]
    if not os.path.exists(scaler_path) or not os.path.exists(nn_model_path):
        raise HTTPException(status_code=400, detail="Model or scaler not found.")
    nn_model = load_model(nn_model_path)
    with open(scaler_path, "rb") as f:
        loaded_data = pickle.load(f)
    scaler = loaded_data["scaler"]
    feature_names = loaded_data["feature_names"]
    raw_input = input_data.dict()
    processed_df = preprocess_raw_input(raw_input)
    processed_df = processed_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    missing_columns = set(feature_names) - set(processed_df.columns)
    for col in missing_columns:
        processed_df[col] = 0
    processed_df = processed_df[feature_names]
    processed_array_normalized = scaler.transform(processed_df)
    prediction_proba = nn_model.predict(processed_array_normalized)
    prediction = (prediction_proba > 0.5).astype(int)
    return {"prediction": int(prediction[0][0]), "probability": float(prediction_proba[0][0])}

@app.post("/train")
async def train(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Endpoint to upload and sanitize CSV training data.

    Args:
        file (UploadFile): CSV file uploaded by the user.

    Returns:
        dict: Confirmation message or error if invalid.
    """
    global training_status
    training_status = {"status": "Training started", "progress": 0}

    # Load configuration
    config = load_config()

    upload_folder = config["upload"]["folder"]
    os.makedirs(upload_folder, exist_ok=True)

    # Validate file format
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV allowed.")

    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(upload_folder, f"train_data_{timestamp}.csv")

    # Save the uploaded file
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    logging.info(f"File saved successfully: {file_path}")

    # Run training in the background
    background_tasks.add_task(run_training, file_path)
    return {"message": "Training started"}

def run_training(file_path):
    global training_status
    try:
        # Step 1: Merge the uploaded CSV into the master dataset
        training_status["status"] = "Merging CSV with master dataset"
        training_status["progress"] = 20
        logging.info("Merging CSV with master dataset...")
        merge_csv(file_path)
        logging.info("CSV merged successfully.")

        # Step 2: Trigger model retraining
        training_status["status"] = "Retraining the model"
        training_status["progress"] = 60
        logging.info("Retraining the model...")
        trigger_model_retraining()
        logging.info("Model retraining completed.")

        # Step 3: Finalize the process
        training_status["status"] = "Training completed"
        training_status["progress"] = 100
        logging.info("Training process completed successfully.")

    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        training_status["status"] = "Training failed"
        training_status["progress"] = 0

@app.get("/train/status")
async def get_training_status():
    return JSONResponse(content=training_status)

# Serve EDA HTML
@app.get("/eda")
async def serve_html():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    HTML_FILE_PATH = os.path.join(project_root, "eda_visualizations/eda_report.html")
    if not os.path.exists(HTML_FILE_PATH):
        raise HTTPException(status_code=404, detail="HTML file not found.")
    webbrowser.open(HTML_FILE_PATH)

# Serve Logistic Regression ROC Curve
@app.get("/plot/lr/roc")
def get_lr_plot():
    return FileResponse("lr_roc_curve.png", media_type="image/png")

# Serve DNN Training Accuracy Plot
@app.get("/plot/dnn/accuracy")
def get_dnn_plot():
    return FileResponse("dnn_training_accuracy.png", media_type="image/png")

# Serve Dashboard
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    dashboard_path = os.path.join(project_root, "src/templates/dashboard.html")
    if not os.path.exists(dashboard_path):
        raise HTTPException(status_code=404, detail="Dashboard HTML file not found.")
    with open(dashboard_path, "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read())