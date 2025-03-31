import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix

def train_and_evaluate_logistic_regression(X_train, Y_train, X_val, Y_val, X_test, Y_test):
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

    # Train the model on the training set
    model.fit(X_train, Y_train)

    # Validate the model on the validation set
    Y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(Y_val, Y_val_pred)
    val_precision = precision_score(Y_val, Y_val_pred)
    val_recall = recall_score(Y_val, Y_val_pred)
    val_f1 = f1_score(Y_val, Y_val_pred)

    print("Validation Set Evaluation:")
    print(f"Accuracy: {val_accuracy:.2f}")
    print(f"Precision: {val_precision:.2f}")
    print(f"Recall: {val_recall:.2f}")
    print(f"F1-Score: {val_f1:.2f}")
    print("\nClassification Report (Validation):")
    print(classification_report(Y_val, Y_val_pred))

    # Evaluate the model on the test set
    Y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)
    test_precision = precision_score(Y_test, Y_test_pred)
    test_recall = recall_score(Y_test, Y_test_pred)
    test_f1 = f1_score(Y_test, Y_test_pred)

    print("\nTest Set Evaluation:")
    print(f"Accuracy: {test_accuracy:.2f}")
    print(f"Precision: {test_precision:.2f}")
    print(f"Recall: {test_recall:.2f}")
    print(f"F1-Score: {test_f1:.2f}")
    print("\nClassification Report (Test):")
    print(classification_report(Y_test, Y_test_pred))

    # Confusion Matrix
    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(Y_test, Y_test_pred))

    return model

