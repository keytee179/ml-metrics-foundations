import pandas as pd

def aggregate_entity_features(
    df,
    entity_column,
    value_column
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if entity_column not in df.columns:
        raise ValueError(f"Missing entity column: {entity_column}")
    if value_column not in df.columns:
        raise ValueError(f"Missing value column: {value_column}")
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        raise ValueError("value column must be numeric")
    if df[entity_column].isna().any():
        raise ValueError(
            "entity column must not contain missing values"
        )
    if df[value_column].isna().any():
        raise ValueError(
            "value column must not contain missing values"
        )
    summary_df  = df.copy()
    summary_df  = (
        summary_df .groupby(entity_column, as_index=False)
        .agg(
            value_mean = (value_column, "mean"),
            value_max = (value_column, "max"),
            observation_count = (value_column, "size"),
        )
        .sort_values(entity_column, ascending=True)
        .reset_index(drop=True)
    )
    return summary_df