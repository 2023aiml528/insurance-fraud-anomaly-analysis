import sys
from utils import load_data, encode_categorical, normalize_data, split_data
from anomaly_detection import AnomalyDetector
from data_preprocessing import load_data, preprocess_data
from models.logistic_regression_model import train_and_evaluate_logistic_regression

print(sys.executable)

# Specify the path to your dataset
dataset_path = r'data/updated_health_insurance_data_Benefits_with_discharge_summary.csv'

# Load the dataset
data = load_data(dataset_path)

# Preprocess the data
data = preprocess_data(data)

print("Post preprocess_data head", data.head().T)

print("Post preprocess_data \n", data.info())


# Normalize the data
#data = normalize_data(data)

# Initialize the anomaly detector
anomaly_detector = AnomalyDetector(contamination=0.1)

# Fit the model and detect anomalies
#anomaly_detector.fit(data)
df_with_anomalies = anomaly_detector.detect_anomalies(data)

# Print the detected anomalies
print("Detected anomalies:")
print(df_with_anomalies[df_with_anomalies['any_anomaly'] == 1].head())


# Prepare X with only numeric fields
X = df_with_anomalies.select_dtypes(include=['number'])

Y = df_with_anomalies['Fraud history approval/rejection status_encoded']

# Debugging: Print the shapes of X and Y
print("Shape of X (features):", X.shape)
print("Shape of Y (target):", Y.shape)

# Debugging: Check the first few rows of X and Y
print("First few rows of X:")
print(X.head())
print("First few values of Y:")
print(Y[:5])

# Split the data into training, validation, and test sets
X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(X, Y, 800, 150, 50)

# Debugging: Print the shapes of the splits
print("Training set shape:", X_train.shape, Y_train.shape)
print("Validation set shape:", X_val.shape, Y_val.shape)
print("Test set shape:", X_test.shape, Y_test.shape)

# Train and evaluate the logistic regression model
model = train_and_evaluate_logistic_regression(X_train, Y_train, X_val, Y_val, X_test, Y_test)
