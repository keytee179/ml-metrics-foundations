import pandas as pd

def prepare_time_ordered_dataframe(
    df,
    timestamp_column,
    entity_column
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if timestamp_column not in df.columns:
        raise ValueError(f"Missing timestamp column: {timestamp_column}")
    if entity_column not in df.columns:
        raise ValueError(f"Missing entity column: {entity_column}")
    result_df = df.copy()
    result_df[timestamp_column] = pd.to_datetime(result_df[timestamp_column],errors="raise")
    result_df = (
        result_df
        .sort_values(
            by=[timestamp_column, entity_column],
            ascending=True,
            kind="stable",
        )
        .drop_duplicates(
            subset=[timestamp_column, entity_column],
            keep="last",
        )
        .reset_index(drop=True)
    )
    if result_df.empty:
        raise ValueError("df must contain at least one row")
    return result_df