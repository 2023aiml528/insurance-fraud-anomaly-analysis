from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)  # Binary classification

from datasets import load_dataset

# Load dataset
dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})

# Preview the dataset
print(dataset)


def tokenize_function(example):
    return tokenizer(example["text_column"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

import torch

# Tokenize sample data
sample_text = "Suspicious transaction detected"
inputs = tokenizer(sample_text, return_tensors="pt")

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1)

# Print results
print(predictions))