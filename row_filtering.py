import pandas as pd

def filter_training_rows(df, minimum_age):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    required_columns = ["age", "income", "house_price"]
    for required_column in required_columns:
        if required_column not in df.columns:
            raise ValueError(
                f"Missing required column: {required_column}"
            )
    mask = (
        (df["age"] >= minimum_age)
        & (df["income"].notna())
        & (df["house_price"].notna())
    )
    filtered_df = df.loc[mask].reset_index(drop=True)
    if filtered_df.empty:
        raise ValueError("no rows remain after filtering")
    return filtered_df