import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from joblib import dump
import logging
from sklearn.preprocessing import MinMaxScaler

def train_and_evaluate_logistic_regression(X, X_train, Y_train, X_val, Y_val, X_test, Y_test):
    """
    Trains and evaluates a logistic regression model.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        Y_train (pd.Series): Training target set.
        X_val (pd.DataFrame): Validation feature set.
        Y_val (pd.Series): Validation target set.
        X_test (pd.DataFrame): Test feature set.
        Y_test (pd.Series): Test target set.

    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    # Initialize the logistic regression model
    model = LogisticRegression(random_state=42, max_iter=1000)

    # Normalize the input data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)



    # Train the model on the training set
    model.fit(X_train, Y_train)

    model_data = {
        "model": model,
        "feature_names": X.columns.tolist(),  # Ensure X is a DataFrame
        "scaler": scaler  # Save the scaler for future use
    }


    # Save the trained model
    dump(model_data, "models/logistic_regression_model.pkl")
    logging.info("Model saved successfully!")

    # Validate the model on the validation set
    Y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(Y_val, Y_val_pred)
    val_precision = precision_score(Y_val, Y_val_pred)
    val_recall = recall_score(Y_val, Y_val_pred)
    val_f1 = f1_score(Y_val, Y_val_pred)

    logging.info("Validation Set Evaluation:")
    logging.info(f"Accuracy: {val_accuracy:.2f}")
    logging.info(f"Precision: {val_precision:.2f}")
    logging.info(f"Recall: {val_recall:.2f}")
    logging.info(f"F1-Score: {val_f1:.2f}")
    logging.info("\nClassification Report (Validation):")
    logging.info(classification_report(Y_val, Y_val_pred))

    # Evaluate the model on the test set
    Y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)
    test_precision = precision_score(Y_test, Y_test_pred)
    test_recall = recall_score(Y_test, Y_test_pred)
    test_f1 = f1_score(Y_test, Y_test_pred)

    logging.info("\nTest Set Evaluation:")
    logging.info(f"Accuracy: {test_accuracy:.2f}")
    logging.info(f"Precision: {test_precision:.2f}")
    logging.info(f"Recall: {test_recall:.2f}")
    logging.info(f"F1-Score: {test_f1:.2f}")
    logging.info("\nClassification Report (Test):")
    logging.info(classification_report(Y_test, Y_test_pred))

    # Confusion Matrix
    logging.info("\nConfusion Matrix (Test):")
    logging.info(confusion_matrix(Y_test, Y_test_pred))

    return model

