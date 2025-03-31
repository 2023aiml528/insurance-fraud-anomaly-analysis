from sklearn.model_selection import train_test_split
def load_data(filepath):
    import pandas as pd
    return pd.read_csv(filepath)

def save_results(results, filepath):
    import pandas as pd
    results.to_csv(filepath, index=False)


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

def split_data(X, Y, train_size, val_size, test_size, random_state=42):
    """
    Splits the dataset into training, validation, and test sets.

    Parameters:
        X (pd.DataFrame): The feature matrix.
        Y (pd.Series or np.array): The target variable.
        train_size (int): Number of samples for the training set.
        val_size (int): Number of samples for the validation set.
        test_size (int): Number of samples for the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_val, X_test, Y_train, Y_val, Y_test)
    """
    from sklearn.model_selection import train_test_split

    # Ensure the total number of rows matches the required split sizes
    total_size = train_size + val_size + test_size
    if len(X) < total_size:
        raise ValueError(f"The dataset must have at least {total_size} rows to split into "
                         f"{train_size} training, {val_size} validation, and {test_size} test samples.")

    # Split the data into training and remaining
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, train_size=train_size, random_state=random_state, stratify=Y)

    # Split the remaining data into validation and test
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, train_size=val_size, random_state=random_state, stratify=Y_temp)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test
