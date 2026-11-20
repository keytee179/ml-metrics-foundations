import pandas as pd

def add_historical_mean_feature(
    df,
    entity_column,
    timestamp_column,
    value_column,
    output_column,
):
    required_columns = [
        entity_column,
        timestamp_column,
        value_column,
    ]
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    for required_column in required_columns:
        if required_column not in df.columns:
            raise ValueError(
                f"Missing required column: {required_column}"
            )
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        raise ValueError("value column must be numeric")
    result_df = df.copy()
    result_df[timestamp_column] = pd.to_datetime(
        result_df[timestamp_column],
        errors="raise",
    )
    result_df = (
        result_df
        .sort_values(
            by=[timestamp_column, entity_column],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    result_df[output_column] = (
        result_df
        .groupby(entity_column)[value_column]
        .transform(
            lambda series: series.shift(1).expanding().mean()
        )
    )
    return result_df