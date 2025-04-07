from fastapi.testclient import TestClient
import sys
import os
import logging

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.api.api import app  # Now Python should find `src.api.api`


client = TestClient(app)

import pdb; pdb.set_trace()


@pytest.fixture
def test_logger():
    """Creates a logger for pytest"""
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    return logger


def test_upload_train_data():
    """Loads a file from the folder and tests the upload API."""
    
    file_path = "data\test.csv"  # Define file location
    
    assert os.path.exists(file_path), f"Test file not found: {file_path}"  # Ensure file exists
    
    with open(file_path, "rb") as file:
        file_data = {"file": (file.name, file.read(), "text/csv")}
    
    response = client.post("/upload_train_data/", files=file_data)
    
    assert response.status_code == 200
    assert "message" in response.json()
    print("Test passed: File uploaded successfully!")

def test_prediction():
    """Test prediction endpoint with sample input."""
    input_data = {"feature1": 1.5, "feature2": 3.2}
    
    response = client.get("/predict/", json=input_data)
    
    assert response.status_code == 200
    assert "prediction" in response.json()