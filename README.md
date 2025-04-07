# **Insurance Fraud Anomaly Analysis**  

## **📌 Project Overview**  
This project aims to **detect fraudulent insurance claims** using **anomaly detection techniques**. By analyzing inconsistencies in **billing details, patient records, policy information, and claim history**, the system predicts whether a claim is **fraudulent or legitimate**.  

## **🔹 Key Features**  
- ✅ **Anomaly Detection** → Identifies irregularities in insurance claims.  
- ✅ **Machine Learning Models** → Predict fraud with high accuracy.  
- ✅ **Data-Driven Insights** → Uses historical patterns to identify fraud.  
- ✅ **Automated Processing** → Reduces manual review efforts.  

---

## **📂 Project Structure**  

### **Data Folder (`data/`)**  
Stores the main dataset and supporting files, organized into:  
- **Backup/** → Archived versions of datasets for safekeeping.  
- **Glove/** → Word embeddings (`glove.6B.50d.txt`) for NLP tasks.  
- **Uploaded Files/** → Stores user-uploaded datasets.  
- **Master Dataset** → `"data/updated_health_insurance_data_Benefits_with_discharge_summary.csv"`  

### **Configuration (`config/`)**  
Contains the **primary YAML configuration file** specifying dataset paths, model settings, and logging configurations.  

### **Logging (`log/`)**  
Provides application logs for **debugging and monitoring**.  

### **Models (`models/`)**  
Contains stored models for fraud prediction:  
- **`logistic_regression_model.py`** → Implements a logistic regression model for fraud detection.  
- **`deep_learning_model.py`** → Builds and trains a **DNN model** to classify fraudulent claims.  
- **`transformer_model.py`** → Currently in the **initial phase of development**, with plans to integrate it with an existing **Hugging Face transformer model**, leveraging the latest OpenAI-powered architectures for fraud detection.  

### **🔹 Future Goals for the Transformer Model**  
- ✅ **Advanced NLP Capabilities** → Utilize transformer-based models for fraud pattern analysis.  
- ✅ **Integration with Hugging Face** → Leverage pretrained models for better accuracy and efficiency.  
- ✅ **Enhanced Fraud Prediction** → Improve classification of insurance claims using deep learning.  
- ✅ **Optimized Performance** → Fine-tune the model for real-time fraud detection.  

As development progresses, this model will complement existing **Logistic Regression and Deep Learning approaches**, offering a **state-of-the-art fraud detection framework**.

### **Source Code (`src/`)**  
Contains main project logic:  
- `data_preprocessing.py` → Functions for loading and cleaning data.  
- `anomaly_detection.py` → Methods for detecting anomalies in insurance claims.  
- `visualization.py` → Generates charts and graphs.  
- `utils.py` → Utility functions used across the project.  

### **Tests (`tests/`)**  
Includes unit tests:  
- `test_anomaly_detection.py` → Validates anomaly detection logic.  

### **Others**  
- **`notebooks/`** → Contains Jupyter notebooks for analysis.  
- **`requirements.txt`** → Lists dependencies required for the project.  
- **`.gitignore`** → Specifies ignored files for version control.  
- **`README.md`** → Project documentation.  

## **🚀 API Instructions**  
### **Start the FastAPI Server**  
```bash
 uvicorn src.api.lr_api:app --reload --log-level info  # Insurance Fraud Anomaly Analysis
```

## 💻 Installation
1️⃣ Create a Python Environment
```bash
    python -m venv env

2️⃣ Activate the Environment
    source env/bin/activate  # Mac/Linux
    env\Scripts\activate  # Windows

3️⃣ Clone the Repository
    git clone https://github.com/2023aiml528/insurance-fraud-anomaly-analysis.git

4️⃣ Navigate to the Project Directory
    cd insurance-fraud-anomaly-analysis

5️⃣ Install Dependencies
    pip install -r requirements.txt