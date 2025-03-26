def load_data(filepath):
    import pandas as pd
    return pd.read_csv(filepath)

def save_results(results, filepath):
    import pandas as pd
    results.to_csv(filepath, index=False)

def preprocess_data(df):
    # Example preprocessing steps
    df.fillna(method='ffill', inplace=True)
    return df

def encode_categorical(df, columns):
    """
    Encodes categorical columns in a DataFrame into numerical codes.

    This function converts the specified categorical columns in the given
    DataFrame into numerical codes using pandas' `astype('category').cat.codes`.
    It modifies the DataFrame in place and returns it.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the data.
        columns (list of str): A list of column names in the DataFrame to be encoded.

    Returns:
        pandas.DataFrame: The DataFrame with the specified columns encoded as numerical codes.

    Example:
        >>> import pandas as pd
        >>> data = {'Category': ['A', 'B', 'A', 'C']}
        >>> df = pd.DataFrame(data)
        >>> encode_categorical(df, ['Category'])
        >>> print(df)
           Category
        0         0
        1         1
        2         0
        3         2
    """
    for col in columns:
        df[col] = df[col].astype('category').cat.codes
    return df

def normalize_data(df):
    import pandas as pd  # Add this import
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns)