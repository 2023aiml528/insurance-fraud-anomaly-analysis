import matplotlib.pyplot as plt
import shap
import logging
import pandas as pd

def plot_anomaly_percentages(anomaly_percentages):
    plt.figure(figsize=(10, 6))
    plt.bar(anomaly_percentages.keys(), anomaly_percentages.values())
    plt.xlabel('Anomaly Type')
    plt.ylabel('Percentage of Anomalies')
    plt.title('Percentage of Each Anomaly Against the Whole Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

import shap

def perform_shap_analysis(model, X_train, model_name="Model", feature_names=None):
    logging.info(f"Performing SHAP analysis for {model_name}...")
    try:
        # If X_train is a NumPy array, convert it to a DataFrame with column names
        if isinstance(X_train, pd.DataFrame):
            data_for_shap = X_train
        else:
            if feature_names is None:
                raise ValueError("Feature names must be provided if X_train is a NumPy array.")
            data_for_shap = pd.DataFrame(X_train, columns=feature_names)

        # Create a SHAP explainer
        explainer = shap.Explainer(model, data_for_shap)

        # Calculate SHAP values
        shap_values = explainer(data_for_shap)

        # Generate a force plot and save it as an HTML file
        force_plot = shap.plots.force(shap_values[0])  # Visualize the first prediction
        shap.save_html(f"logs/{model_name}_shap_force.html", force_plot)
        logging.info(f"SHAP force plot saved for {model_name}.")

        # Generate a summary plot
        shap.summary_plot(shap_values, data_for_shap, show=False)
        plt.savefig(f"logs/{model_name}_shap_summary.png")
        logging.info(f"SHAP summary plot saved for {model_name}.")
    except Exception as e:
        logging.error(f"Error during SHAP analysis for {model_name}: {str(e)}")


# visualization.py
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_lr_roc(y_true, y_proba, save_path="lr_roc_curve.png"):
    """
    Function to plot ROC curve for Logistic Regression.

    Parameters:
    - y_true: Array of true labels
    - y_proba: Array of predicted probabilities
    - save_path: File path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.title("Logistic Regression ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(save_path)
    plt.close()        



def save_dnn_training_plot(history, output_path="dnn_training_accuracy.png"):
    """
    Save the DNN training accuracy plot.

    Args:
        history: Training history object containing accuracy and loss values.
        output_path: Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('DNN Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("dnn_training_accuracy.png")
    plt.close()      