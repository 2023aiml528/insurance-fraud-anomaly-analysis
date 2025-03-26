import sys
from utils import load_data, preprocess_data, encode_categorical, normalize_data
from anomaly_detection import AnomalyDetector

print(sys.executable)

# Specify the path to your dataset
dataset_path = r'data/updated_health_insurance_data_Benefits_with_discharge_summary.csv'

# Load the dataset
data = load_data(dataset_path)

# Preprocess the data
data = preprocess_data(data)

print(data.head())

# Automatically identify object-type columns (categorical columns)
categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns identified:", categorical_columns)

# Encode categorical columns if any exist
if categorical_columns:
    data = encode_categorical(data, categorical_columns)
else:
    print("No categorical columns to encode.")

# Normalize the data
data = normalize_data(data)

# Initialize the anomaly detector
anomaly_detector = AnomalyDetector(contamination=0.1)

# Fit the model and detect anomalies
anomaly_detector.fit(data)
anomalies = anomaly_detector.detect_anomalies(data)

# Print the detected anomalies
print("Detected anomalies:")
print(anomalies)