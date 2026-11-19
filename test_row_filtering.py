import pandas as pd
import pytest

from row_filtering import filter_training_rows

def test_filter_training_rows_applies_shared_mask_and_resets_index():
    df = pd.DataFrame({
        "age": [18, 25, 30, 40, 50],
        "income": [8.0, 15.0, None, 35.0, 50.0],
        "house_price": [0.8, 1.2, 1.8, None, 4.0],
    })
    minimum_age = 25
    filtered_df = filter_training_rows(df, minimum_age)
    assert filtered_df.shape == (2, 3)
    assert filtered_df.index.tolist() == [0, 1]
    assert filtered_df["age"].tolist() == [25, 50]
    assert filtered_df["house_price"].tolist() == [1.2, 4.0]

def test_non_dataframe_input_raises_type_error():
    df = [
        [25, 15.0, 1.2],
        [50, 50.0, 4.0],
    ]
    minimum_age = 25
    with pytest.raises(TypeError, match = "df must be a pandas DataFrame"):
        filter_training_rows(df, minimum_age)

def test_missing_required_column_raises_value_error():
    df = pd.DataFrame({
        "age": [18, 25, 30, 40, 50],
        "house_price": [0.8, 1.2, 1.8, None, 4.0],
    })
    minimum_age = 25
    with pytest.raises(ValueError, match = "Missing required column: income"):
        filter_training_rows(df, minimum_age)

def test_no_rows_remain_raises_value_error():
    df = pd.DataFrame({
        "age": [18, 20],
        "income": [8.0, 10.0],
        "house_price": [0.8, 1.0],
    })
    minimum_age = 25
    with pytest.raises(ValueError, match = "no rows remain after filtering"):
        filter_training_rows(df, minimum_age)
