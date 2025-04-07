import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from joblib import dump, load
from imblearn.over_sampling import SMOTE
from tensorflow.keras.metrics import Precision, Recall

# Dynamically handle imports based on execution context
try:
    # When running from `main.py`

    from utils import load_data, encode_categorical, normalize_data, split_data, load_config
except ModuleNotFoundError:
    # When running from FastAPI (e.g., `uvicorn`)
    from src.utils import load_data, encode_categorical, normalize_data, split_data, load_config


# Function to build and evaluate the deep learning model
def build_and_evaluate_deep_learning_model(X, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=20, batch_size=20):
    """
    Builds, trains, and evaluates a deep learning neural network model.
    """



    # Check for NaN values and replace them before scaling
    X_train = np.nan_to_num(X_train)
    X_val = np.nan_to_num(X_val)
    X_test = np.nan_to_num(X_test)

    # Normalize the input data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    scaler_data = {
        "scaler": scaler,
        "feature_names": X.columns
    }

    import pickle


    # Save the dictionary using pickle
    scaler_path = "models/scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler_data, f)

        # Balance dataset using SMOTE (if needed)
    smote = SMOTE(random_state=42)
    X_train_resampled, Y_train_resampled = smote.fit_resample(X_train, Y_train)

    # Build the model
    model = build_model(input_dim=X_train.shape[1])

    # Implement Early Stopping to prevent overfitting
    early_stop = CustomEarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    logging.info("Training the model...")
    history = model.fit(
        X_train_resampled, Y_train_resampled,
        validation_data=(X_val, Y_val),
        epochs=epochs, batch_size=batch_size,
        verbose=2, #callbacks=[early_stop]
    )

    

    # Evaluate the model on the test set
    logging.info("\nEvaluating the model on the test set...")
    Y_test_pred_prob = model.predict(X_test)

    # Handle NaN values in predictions
    if np.isnan(Y_test_pred_prob).any():
        logging.error("Model produced NaN values in prediction probabilities.")
        Y_test_pred_prob = np.nan_to_num(Y_test_pred_prob, nan=0, posinf=1, neginf=0)

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

    # Input layer
    model.add(Dense(39, activation='relu', input_dim=input_dim))  # Match input features
    model.add(BatchNormalization())  # Helps stabilize training

    # Hidden layers with dropout to prevent overfitting
    for _ in range(5):  # Reduce layers for efficiency
        model.add(Dense(64))  
        model.add(BatchNormalization())  # Place before activation  
        model.add(Activation('relu'))  
        model.add(Dropout(0.3))  # Enable dropout 

    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model with optimized settings
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Reduce learning rate for better stability
        loss='binary_crossentropy',
        metrics=['accuracy',Precision(), Recall()]
    )

    return model



class CustomEarlyStopping(EarlyStopping):
    def on_train_begin(self, logs=None):
        """Initialize best metric value at the start of training"""
        self.best = np.inf if self.monitor_op == np.less else -np.inf
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):
        """Ensure weights are stored on epoch 0 and properly track best metric value"""
        if epoch == 0:  # Store initial weights
            self.best_weights = self.model.get_weights()
        super().on_epoch_end(epoch, logs)


