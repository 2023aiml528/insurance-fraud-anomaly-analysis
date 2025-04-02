import matplotlib.pyplot as plt
import shap

def plot_anomaly_percentages(anomaly_percentages):
    plt.figure(figsize=(10, 6))
    plt.bar(anomaly_percentages.keys(), anomaly_percentages.values())
    plt.xlabel('Anomaly Type')
    plt.ylabel('Percentage of Anomalies')
    plt.title('Percentage of Each Anomaly Against the Whole Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def perform_shap_analysis(model, X_train, X_test, feature_names):
    """
    Perform SHAP analysis to explain the predictions of a machine learning model.

    Parameters:
    - model: Trained machine learning model (e.g., XGBoost, LightGBM, etc.)
    - X_train: Training dataset used to fit the SHAP explainer.
    - X_test: Test dataset for which SHAP values will be computed.
    - feature_names: List of feature names for better visualization.

    Returns:
    - shap_values: Computed SHAP values for the test dataset.
    """
    # Initialize the SHAP explainer
    explainer = shap.Explainer(model, X_train)

    # Compute SHAP values for the test dataset
    shap_values = explainer(X_test)

    # Summary plot of SHAP values
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)

    # Bar plot of mean absolute SHAP values
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar")

    return shap_values    