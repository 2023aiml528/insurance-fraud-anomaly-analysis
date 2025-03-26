import pandas as pd
def load_data(filepath):
    import pandas as pd
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    
    # Encode categorical variables
    df = pd.get_dummies(df, drop_first=True)
    
    # Normalize numerical features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df

def split_data(df, target_column):
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test