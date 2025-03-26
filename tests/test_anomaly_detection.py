import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from anomaly_detection import AnomalyDetector
import pytest
from anomaly_detection import AnomalyDetector
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
import sys


def test_anomaly_detection_initialization():
    detector = AnomalyDetector()
    assert detector is not None

def test_anomaly_detection_with_sample_data():
    sample_data = [
        {'claim_amount': 1000, 'is_fraudulent': 0},
        {'claim_amount': 5000, 'is_fraudulent': 1},
        {'claim_amount': 2000, 'is_fraudulent': 0},
    ]
    detector = AnomalyDetector()
    results = detector.detect_anomalies(sample_data)
    assert isinstance(results, list)
    assert len(results) == 3

def test_anomaly_detection_fraudulent_claims():
    sample_data = [
        {'claim_amount': 1000, 'is_fraudulent': 0},
        {'claim_amount': 5000, 'is_fraudulent': 1},
        {'claim_amount': 2000, 'is_fraudulent': 0},
    ]
    detector = AnomalyDetector()
    results = detector.detect_anomalies(sample_data)
    fraudulent_claims = [claim for claim in results if claim['is_fraudulent'] == 1]
    assert len(fraudulent_claims) == 1

def test_anomaly_detection_no_fraudulent_claims():
    sample_data = [
        {'claim_amount': 1000, 'is_fraudulent': 0},
        {'claim_amount': 2000, 'is_fraudulent': 0},
    ]
    detector = AnomalyDetector()
    results = detector.detect_anomalies(sample_data)
    fraudulent_claims = [claim for claim in results if claim['is_fraudulent'] == 1]
    assert len(fraudulent_claims) == 0
    def test_fit_with_valid_data():
        detector = AnomalyDetector()
        sample_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [10.0, 20.0, 30.0, 40.0]
        })
        try:
            detector.fit(sample_data)
        except Exception as e:
            pytest.fail(f"fit method raised an exception: {e}")

    def test_fit_with_empty_data():
        detector = AnomalyDetector()
        empty_data = pd.DataFrame()
        with pytest.raises(ValueError):
            detector.fit(empty_data)

    def test_fit_with_non_numeric_data():
        detector = AnomalyDetector()
        non_numeric_data = pd.DataFrame({
            'feature1': ['a', 'b', 'c'],
            'feature2': ['x', 'y', 'z']
        })
        with pytest.raises(ValueError):
            detector.fit(non_numeric_data)

    def test_fit_model_is_fitted():
        detector = AnomalyDetector()
        sample_data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0],
            'feature2': [10.0, 20.0, 30.0, 40.0]
        })
        detector.fit(sample_data)
        try:
            detector.model.predict([[0.0, 0.0]])
        except NotFittedError:
            pytest.fail("Model was not fitted after calling fit method")
