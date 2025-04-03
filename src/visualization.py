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