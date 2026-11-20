from training_table import build_training_table
from preprocessing import prepare_regression_data

def build_and_prepare_regression_data(
    observations_df,
    static_features_df,
    entity_column,
    timestamp_column,
    feature_columns,
    target_column,
    validation_ratio,
):
    training_df = build_training_table(
        observations_df,
        static_features_df,
        entity_column,
        timestamp_column,
        feature_columns,
        target_column,
    )
    (
        X_train_scaled,
        X_valid_scaled,
        y_train,
        y_valid,
        feature_means,
        feature_stds,
    ) = prepare_regression_data(
        training_df,
        feature_columns,
        target_column,
        validation_ratio,
    )
    return (
        training_df,
        X_train_scaled,
        X_valid_scaled,
        y_train,
        y_valid,
        feature_means,
        feature_stds,
    )