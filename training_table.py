from feature_merging import merge_entity_features
from data_validation import validate_training_dataframe
from time_ordering import prepare_time_ordered_dataframe

def build_training_table(
    observations_df,
    static_features_df,
    entity_column,
    timestamp_column,
    feature_columns,
    target_column,
):
    ordered_df = prepare_time_ordered_dataframe(
        observations_df,
        timestamp_column,
        entity_column,
    )
    merged_df = merge_entity_features(
        ordered_df,
        static_features_df,
        entity_column,
    )
    X_df, y_series = validate_training_dataframe(
        merged_df,
        feature_columns,
        target_column,
    )
    output_columns = (
        [timestamp_column, entity_column]
        + feature_columns
        + [target_column]
    )
    training_df = merged_df.loc[:, output_columns].copy()
    training_df = training_df.reset_index(drop=True)
    return training_df