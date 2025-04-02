import sys
import os
from utils import load_data, encode_categorical, normalize_data, split_data
from anomaly_detection import AnomalyDetector
from data_preprocessing import load_data, preprocess_data
from models.logistic_regression_model import train_and_evaluate_logistic_regression
from utils import perform_shap_analysis
import yaml
from logging_config import setup_logging
import logging



# Load the configuration file
config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

    
# Set up logging
setup_logging(config_path)

# Example log message
logging.info("Logging is configured successfully.")

# Access the dataset path and columns
dataset_path = config["dataset"]["path"]
target_column = config["columns"]["target"]
nominal_columns = config["columns"]["nominal"]

test_size = config["dataset"]["test_size"]
validation_size = config["dataset"]["validation_size"]
train_size = config["dataset"]["train_size"]


logging.info("Dataset Path:", dataset_path)
logging.info("Target Column:", target_column)
logging.info("Nominal Columns:", nominal_columns)

logging.info(sys.executable)

# Specify the path to your dataset
#dataset_path = r'data/updated_health_insurance_data_Benefits_with_discharge_summary.csv'

# Load the dataset
data = load_data(dataset_path)

# Preprocess the data
data = preprocess_data(data)

logging.info("Post preprocess_data head", data.head().T)

logging.info("Post preprocess_data \n", data.info())




# Initialize the anomaly detector
anomaly_detector = AnomalyDetector(contamination=0.1)

# Fit the model and detect anomalies
#anomaly_detector.fit(data)
df_with_anomalies = anomaly_detector.detect_anomalies(data)

# logging.info the detected anomalies
logging.info("Detected anomalies:")
logging.info(df_with_anomalies[df_with_anomalies['any_anomaly'] == 1].head())

# Normalize the data
df_with_anomalies_normalized = normalize_data(df_with_anomalies)

# Prepare X with only numeric fields
X = df_with_anomalies_normalized.select_dtypes(include=['number'])

Y = df_with_anomalies_normalized['Fraud history approval/rejection status_encoded']

# Debugging: logging.info the shapes of X and Y
logging.info("Shape of X (features):", X.shape)
logging.info("Shape of Y (target):", Y.shape)

# Debugging: Check the first few rows of X and Y
logging.info("First few rows of X:")
logging.info(X.head())
logging.info("First few values of Y:")
logging.info(Y[:5])

# Split the data into training, validation, and test sets
X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X, Y, train_size, validation_size, test_size)

# Debugging: logging.info the shapes of the splits
logging.info("Training set shape:", X_train.shape, Y_train.shape)
logging.info("Validation set shape:", X_val.shape, Y_val.shape)
logging.info("Test set shape:", X_test.shape, Y_test.shape)

# Train and evaluate the logistic regression model
model = train_and_evaluate_logistic_regression(X_train, Y_train, X_val, Y_val, X_test, Y_test)

# Perform SHAP analysis
shap_values = perform_shap_analysis(model, X_train, X_test, feature_names=X.columns)


# call the deep learning model
from models.deep_learning_model import build_and_evaluate_deep_learning_model

dl_model = build_and_evaluate_deep_learning_model(X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=20, batch_size=20)
# Perform SHAP analysis for the deep learning model