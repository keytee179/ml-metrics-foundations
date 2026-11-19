import pandas as pd

def validate_training_dataframe(
    df,
    feature_columns,
    target_column,
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if len(feature_columns) == 0:
        raise ValueError("feature_columns must not be empty")
    if target_column in feature_columns:
        raise ValueError("target_column must not be included in feature_columns")
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("feature_columns must contain unique column names")
    for column in feature_columns:
        if column not in df.columns:
            raise ValueError(f"Missing feature column: {column}")
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")
    if df.empty:
            raise ValueError("df must contain at least one row")
    X_df = df[feature_columns]
    y_series = df[target_column]
    if X_df.isna().any().any():
        raise ValueError("feature columns must not contain missing values")
    if y_series.isna().any():
        raise ValueError("target columns must not contain missing values")
    for column in feature_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError("feature columns must be numeric")
    if not pd.api.types.is_numeric_dtype(y_series):
        raise ValueError("target column must be numeric")
    return X_df, y_series