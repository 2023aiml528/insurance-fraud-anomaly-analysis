import logging
import numpy as np
import tensorflow as tf
logging.info(f"TensorFlow version: {tf.__version__}")
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from joblib import dump
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def build_and_evaluate_deep_learning_model(X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=10, batch_size=20):
    """
    Builds, trains, and evaluates a deep learning neural network model.
    """
    # Normalize the input data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save the scaler to a file
    dump(scaler, "models/scaler.pkl")

    # Build the model
    model = build_model(input_dim=X_train.shape[1])

    # Train the model
    logging.info("Training the model...")
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, verbose=2)

    # Evaluate the model on the test set
    logging.info("\nEvaluating the model on the test set...")
    Y_test_pred_prob = model.predict(X_test)

    # Handle NaN values in predictions
    if np.isnan(Y_test_pred_prob).any():
        logging.error("Model produced NaN values in prediction probabilities.")
        Y_test_pred_prob = np.zeros_like(Y_test_pred_prob)

    Y_test_pred = (Y_test_pred_prob > 0.5).astype(int)

    test_accuracy = accuracy_score(Y_test, Y_test_pred) * 100
    test_precision = precision_score(Y_test, Y_test_pred, zero_division=0)
    test_recall = recall_score(Y_test, Y_test_pred, zero_division=0)
    test_f1 = f1_score(Y_test, Y_test_pred, zero_division=0)

    logging.info("\nDL Test Set Evaluation:")
    logging.info(f"DL Accuracy: {test_accuracy:.2f}%")
    logging.info(f"DL Precision: {test_precision:.2f}")
    logging.info(f"DL Recall: {test_recall:.2f}")
    logging.info(f"DL F1-Score: {test_f1:.2f}")
    logging.info("\nDL Classification Report (Test):")
    logging.info(classification_report(Y_test, Y_test_pred, zero_division=0))

    # Confusion Matrix
    logging.info("\nDL Confusion Matrix (Test):")
    logging.info(confusion_matrix(Y_test, Y_test_pred))

    return model, history



def build_model(input_dim):
    """
    Builds a deep learning model.

    Parameters:
        input_dim (int): Number of input features.

    Returns:
        tf.keras.Model: The compiled deep learning model.
    """

    logging.info(f"Building the model with input dimension: {input_dim}")
    model = Sequential()
    model.add(Dense(50, activation='relu', input_dim=input_dim))
    for _ in range(7):  # Add 7 more hidden layers
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.2))  # Add dropout to prevent overfitting
    model.add(Dense(1, activation='sigmoid'))  # Output layer
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model