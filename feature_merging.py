import pandas as pd

def merge_entity_features(
    observations_df,
    entity_features_df,
    entity_column
):
    if not isinstance(observations_df, pd.DataFrame):
        raise TypeError("observations_df must be a pandas DataFrame")
    if not isinstance(entity_features_df, pd.DataFrame):
        raise TypeError("entity_features_df must be a pandas DataFrame")
    if entity_column not in observations_df.columns:
        raise ValueError(f"Missing entity column in observations_df: {entity_column}")
    if entity_column not in entity_features_df.columns:
        raise ValueError(f"Missing entity column in entity_features_df: {entity_column}")
    df_merge = observations_df.merge(
        entity_features_df,
        on=entity_column,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if (df_merge["_merge"] == "left_only").any():
        raise ValueError("some observations have no matching entity features")
    df_merge = df_merge.drop(columns=["_merge"])
    if observations_df.shape[0] != df_merge.shape[0]:
        raise ValueError("merge changed the number of observations")
    df_merge = df_merge.reset_index(drop=True)
    return df_merge