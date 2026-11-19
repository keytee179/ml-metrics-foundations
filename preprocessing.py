import numpy as np
from data_validation import validate_training_dataframe
from splitting import time_ordered_split
from standardization import fit_standardizer, transform_standardizer
def prepare_regression_data(df, feature_columns, target_column, validation_ratio):
    X_df, y_series = validate_training_dataframe(df, feature_columns, target_column)
    X = np.array(X_df, dtype = float)
    y = np.array(y_series, dtype = float)
    X_train, X_valid, y_train, y_valid = time_ordered_split(X, y, validation_ratio)
    feature_means, feature_stds = fit_standardizer(X_train)
    X_train_scaled = transform_standardizer(X_train, feature_means, feature_stds)
    X_valid_scaled = transform_standardizer(X_valid, feature_means, feature_stds)
    return (
        X_train_scaled,
        X_valid_scaled,
        y_train,
        y_valid,
        feature_means,
        feature_stds,
    )