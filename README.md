# Project Overview
This project aims to identify fraudulent insurance claims using anomaly detection techniques applied to various patterns in the dataset. By analyzing discrepancies in billing details, patient records, policy information, and claim history, the system predicts whether a claim is fraudulent or legitimate.

-🔹 Key Features:

  -✅ Anomaly Detection → Identifies irregularities in insurance claims.
  -✅ Machine Learning Models → Predict fraud with high accuracy.
  -✅ Data-Driven Insights → Uses policyholder details and historical patterns.
  -✅ Automated Processing → Reduces manual review efforts


## Project Structure
- **data/**: Contains the dataset and related documentation.
The data folder is organized into three directories, each serving a specific purpose:
-1️⃣ Backup - Contains archived versions of datasets for safekeeping.

-2️⃣ Glove - Contains word embedding files (glove.6B.50d.txt) for NLP tasks.

-3️⃣ Uploaded Files - Holds user-uploaded datasets for analysis.

-Additionally, the folder includes one master dataset file, which serves as the primary data source for processing and analysis.

- **config/**: Holds the primary YAML configuration file

- **log/** :Application logging helps with debugging and monitoring

- **models/**: Directory contains stored model dumps for:
  - **Logistic Regression Model →**
      - A traditional statistical method for fraud detection.
  - **Deep Neural Network (DNN) Model →**
      - A more advanced machine learning approach for complex fraud patterns.

- **notebooks/**: Contains Jupyter notebooks for analysis.

- **src/**: Contains the source code for the project.
    - **__init__.py**: Marks the directory as a Python package.
  - **data_preprocessing.py**: Functions for loading and preprocessing the dataset.
  - **anomaly_detection.py**: Classes and functions for detecting anomalies in the dataset.
  - **visualization.py**: Functions for visualizing analysis results.
  - **utils.py**: Utility functions used across the project.
  - **models/deep_learning_model.py**: script is responsible for training and building a Deep Neural Network (DNN) model designed for insurance fraud detection. This model leverages advanced machine learning techniques to analyze patterns in claims data and predict fraudulent cases.
   - **models/logistic_regression_model.py**:script implements a logistic regression model for detecting insurance fraud. It trains the model using the provided dataset and saves the trained model as a  file, enabling future predictions on unseen data.
   - **models/tranformer_model.py**: script is currently in the initial phase of development. The plan is to integrate it with an existing Hugging Face transformer model, leveraging the latest advancements in OpenAI-powered architectures for fraud detection.

-🔹 Future Goals for the Transformer Model
-✅ Advanced NLP Capabilities → Utilize transformer-based models for fraud pattern analysis.
-✅ Integration with Hugging Face → Leverage pretrained models for better accuracy and efficiency.
-✅ Enhanced Fraud Prediction → Improve classification of insurance claims using deep learning.
-✅ Optimized Performance → Fine-tune the model for real-time fraud detection.

As development progresses, this model will complement existing Logistic Regression and Deep Learning approaches, offering a state-of-the-art fraud detection framework.
- **tests/**: Contains unit tests for the project.
  - **__init__.py**: Marks the directory as a Python package for testing.
  - **test_anomaly_detection.py**: Unit tests for the anomaly detection logic.

- **requirements.txt**: Lists the dependencies required for the project.

- **.gitignore**: Specifies files and directories to be ignored by version control.

"""
- **README.md**: 
    api_instructions = """


 -Key Function:
  -✅ Data Preprocessing → Cleans and prepares the dataset for model training.
  -✅ Model Architecture Design → Defines layers, activation functions, and optimization methods.
  -✅ Training Process → Trains the DNN model using labeled insurance fraud data.
  -✅ Evaluation & Metrics → Assesses accuracy, precision, recall, and F1-score.


## API Instructions

1. **Start the FastAPI server:**
   ```bash
     uvicorn src.api.lr_api:app --reload --log-level info   # Insurance Fraud Anomaly Analysis

This project focuses on analyzing anomalies in insurance fraud detection. It aims to identify and visualize various types of anomalies present in insurance claims data.


## Installation

### 1. Create Python env  
  python -m venv env

### 2. Activate the env
  source env/bin/activate  

### 3. Clone the repository:
   ```
   git clone https://github.com/2023aiml528/insurance-fraud-anomaly-analysis.git
   ```
### 4. Navigate to the project directory:
   ```
   cd insurance-fraud-anomaly-analysis
   ```
### 5. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Preprocess the dataset using the functions in `src/data_preprocessing.py`.
2. Detect anomalies using the classes and functions in `src/anomaly_detection.py`.
3. Visualize the results with the functions in `src/visualization.py`.
4. Run the Jupyter notebook in `notebooks/` for an interactive analysis.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
