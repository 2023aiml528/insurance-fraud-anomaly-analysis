from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load FinBERT (optimized for financial text)
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
from datasets import load_dataset

def detect_fraud(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    
    return "Fraudulent" if prediction == 1 else "Legitimate"

# Example usage
sample_text = "This insurance claim seems suspicious due to inconsistent details."
print(detect_fraud(sample_text))


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset (replace with actual fraud dataset)
data = pd.read_csv("updated_health_insurance_data_Benefits_with_discharge_summary.csv")
X_train, X_test, y_train, y_test = train_test_split(data["text"], data["label"], test_size=0.2)

# Train model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Save model
import pickle
pickle.dump(rf_model, open("fraud_model.pkl", "wb"))


from fastapi import FastAPI
import pickle

app = FastAPI()
model = pickle.load(open("fraud_model.pkl", "rb"))

@app.post("transformer/predict")
def predict(text: str):
    prediction = model.predict([text])
    return {"fraud_status": "Fraudulent" if prediction[0] == 1 else "Legitimate"}

# Run API: uvicorn api:app --reload