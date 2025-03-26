import matplotlib.pyplot as plt

def plot_anomaly_percentages(anomaly_percentages):
    plt.figure(figsize=(10, 6))
    plt.bar(anomaly_percentages.keys(), anomaly_percentages.values())
    plt.xlabel('Anomaly Type')
    plt.ylabel('Percentage of Anomalies')
    plt.title('Percentage of Each Anomaly Against the Whole Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()