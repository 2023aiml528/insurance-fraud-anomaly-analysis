

import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
from keras import models
from models import Sequential
from tf.models.layers import Dense, Dropout
from optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def build_and_evaluate_deep_learning_model(X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs=50, batch_size=32):
    """
    Builds, trains, and evaluates a deep learning neural network model.

    Parameters:
        X_train (pd.DataFrame): Training feature set.
        Y_train (pd.Series): Training target set.
        X_val (pd.DataFrame): Validation feature set.
        Y_val (pd.Series): Validation target set.
        X_test (pd.DataFrame): Test feature set.
        Y_test (pd.Series): Test target set.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.

    Returns:
        tf.keras.Model: The trained deep learning model.
    """
    # Define the model
    model = Sequential()

    # Input layer and 8 hidden layers with 50 nodes each
    model.add(Dense(50, activation='relu', input_dim=X_train.shape[1]))
    for _ in range(7):  # Add 7 more hidden layers
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.2))  # Add dropout to prevent overfitting

    # Output layer with 1 node (binary classification)
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    print("Training the model...")
    history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=epochs, batch_size=batch_size, verbose=1)

    # Evaluate the model on the test set
    print("\nEvaluating the model on the test set...")
    Y_test_pred_prob = model.predict(X_test)
    Y_test_pred = (Y_test_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

    test_accuracy = accuracy_score(Y_test, Y_test_pred) * 100
    test_precision = precision_score(Y_test, Y_test_pred)
    test_recall = recall_score(Y_test, Y_test_pred)
    test_f1 = f1_score(Y_test, Y_test_pred)

    print("\nTest Set Evaluation:")
    print(f"Accuracy: {test_accuracy:.2f}%")
    print(f"Precision: {test_precision:.2f}")
    print(f"Recall: {test_recall:.2f}")
    print(f"F1-Score: {test_f1:.2f}")
    print("\nClassification Report (Test):")
    print(classification_report(Y_test, Y_test_pred))

    # Confusion Matrix
    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(Y_test, Y_test_pred))

    return model, history