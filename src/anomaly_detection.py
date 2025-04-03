import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class AnomalyDetector:
    def __init__(self, contamination=0.1):
        self.contamination = contamination
        self.model = IsolationForest(contamination=self.contamination)

    def fit(self, data):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        self.model.fit(scaled_data)

    def predict(self, data):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return self.model.predict(scaled_data)

    def detect_anomalies(self, data):
        # Step 1: Model-based anomalies using Isolation Forest
        #predictions = self.predict(data)
        #data['model_based_anomaly'] = (predictions == -1).astype(int)

        # Rule 3: Policy Dates anomaly (Start Date after End Date)
        policy_date_anomalies = data[data['start_date'] > data['end_date']].index.tolist()
        data['policy_date_anomaly'] = 0
        data.loc[policy_date_anomalies, 'policy_date_anomaly'] = 1

        # Step 2: Rule-based anomalies
        # Rule 1: Hospitalized Date outside Start Date and End Date
        hospitalized_date_anomalies = data[
            (data['hospitalized_date'] < data['start_date']) | 
            (data['hospitalized_date'] > data['end_date'])
        ].index.tolist()
        data['hospitalized_date_anomaly'] = 0
        data.loc[hospitalized_date_anomalies, 'hospitalized_date_anomaly'] = 1

        # Rule 2: Anomaly detection for Claim Limit lower than Total Amount
        claim_limit_lower_than_total_amount_anomalies = data[data['claim_limits'] < data['total_amount']].index.tolist()
        data['claim_limit_lower_than_total_amount_anomaly'] = 0
        data.loc[claim_limit_lower_than_total_amount_anomalies, 'claim_limit_lower_than_total_amount_anomaly'] = 1

        # Rule 3: Policy Dates anomaly (Start Date after End Date)
        # Document submitted flag (binary column)
        document_submitted_anomalies = data[(data['claim_documents_submitted'] == 'No') & (data['fraud_history_approval_rejection_status'] == 'Approved')].index.tolist()
        data['document_submitted_anomaly'] = 0
        data.loc[document_submitted_anomalies, 'document_submitted_anomaly'] = 1

        # Rule 4: Fraud History anomaly (missing or invalid values)
        covered_anomalies = data[(data['covered'] == 'No') & (data['fraud_history_approval_rejection_status'] == 'Approved')].index.tolist()
        data['covered_anomaly'] = 0
        data.loc[covered_anomalies, 'covered_anomaly'] = 1

        #Rule 5: Detect anomalies based on benefits validity 
        benefits_anomalies = data[~data.apply(AnomalyDetector.check_benefits_validity, axis=1)].index.tolist()
        data['benefits_anomaly'] = 0
        data.loc[benefits_anomalies, 'benefits_anomaly'] = 1

        # Step 3: Combine all anomalies into a single column
        data['any_anomaly'] = data[
            ['hospitalized_date_anomaly', 'claim_limit_lower_than_total_amount_anomaly', 'policy_date_anomaly', 'benefits_anomaly','covered_anomaly','document_submitted_anomaly']
        ].max(axis=1)

        return data
    

    # Function to check if benefits are valid for the given policy name
    @staticmethod
    def check_benefits_validity(row):
        import json
        path = r'data//health_insurance_plans_benefits_mapping.json'
        # Load the mapping from the JSON file
        with open(path) as f:
            health_insurance_plans_benefits_mapping = json.load(f)
        policy_name = row['policy_name']
        benefits = row['benefits'].split(';')
        
        valid_benefits = health_insurance_plans_benefits_mapping.get(policy_name, [])
        
        for benefit in benefits:
            if benefit not in valid_benefits:
                return False
        return True