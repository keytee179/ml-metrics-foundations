import pandas as pd
import pytest
from data_validation import validate_training_dataframe

def test_valid_dataframe_returns_features_and_target():
    df = pd.DataFrame({
        "age": [25, 30, 40],
        "income": [15.0, 20.0, 35.0],
        "house_price": [1.2, 1.8, 3.0],
    })
    feature_columns = ["age", "income"]
    target_column = "house_price"
    X_df, y_series = validate_training_dataframe(df, feature_columns, target_column)
    assert isinstance(X_df, pd.DataFrame)
    assert isinstance(y_series, pd.Series)
    assert X_df.columns.tolist() == ["age", "income"]
    assert X_df.shape == (3, 2)
    assert y_series.shape == (3,)
    assert y_series.tolist() == [1.2, 1.8, 3.0]

def test_non_dataframe_input_raises_type_error():
    df = [
        [25, 15.0, 1.2],
        [30, 20.0, 1.8],
    ]
    feature_columns = ["age", "income"]
    target_column = "house_price"
    with pytest.raises(TypeError):
        validate_training_dataframe(df, feature_columns, target_column)

def test_empty_feature_columns_raises_value_error():
    df = pd.DataFrame({
        "age": [25, 30, 40],
        "income": [15.0, 20.0, 35.0],
        "house_price": [1.2, 1.8, 3.0],
    })
    feature_columns = []
    target_column = "house_price"
    with pytest.raises(ValueError, match = "feature_columns must not be empty"):
        validate_training_dataframe(df, feature_columns, target_column)

def test_missing_feature_column_raises_value_error():
    df = pd.DataFrame({
        "age": [25, 30, 40],
        "house_price": [1.2, 1.8, 3.0],
    })
    feature_columns = ["age", "income"]
    target_column = "house_price"
    with pytest.raises(ValueError, match = "Missing feature column: income"):
        validate_training_dataframe(df, feature_columns, target_column)

def test_missing_target_column_raises_value_error():
    df = pd.DataFrame({
        "age": [25, 30, 40],
        "income": [15.0, 20.0, 35.0],
    })
    feature_columns = ["age", "income"]
    target_column = "house_price"
    with pytest.raises(ValueError, match = "Missing target column: house_price"):
        validate_training_dataframe(df, feature_columns, target_column)

def test_target_in_feature_columns_raises_value_error():
    df = pd.DataFrame({
        "age": [25, 30, 40],
        "income": [15.0, 20.0, 35.0],
        "house_price": [1.2, 1.8, 3.0],
    })
    feature_columns = ["age", "income", "house_price"]
    target_column = "house_price"
    with pytest.raises(ValueError, match="target_column must not be included in feature_columns"):
        validate_training_dataframe(df, feature_columns, target_column)

def test_duplicate_feature_columns_raises_value_error():
    df = pd.DataFrame({
        "age": [25, 30, 40],
        "income": [15.0, 20.0, 35.0],
        "house_price": [1.2, 1.8, 3.0],
    })
    feature_columns = ["age", "income", "age"]
    target_column = "house_price"
    with pytest.raises(ValueError, match = "feature_columns must contain unique column names"):
        validate_training_dataframe(df, feature_columns, target_column)

def test_empty_dataframe_raises_value_error():
    df = pd.DataFrame({
        "age": [],
        "income": [],
        "house_price": [],
    })
    feature_columns = ["age", "income"]
    target_column = "house_price"
    with pytest.raises(ValueError, match="df must contain at least one row"):
        validate_training_dataframe(df, feature_columns, target_column)

def test_missing_value_in_features_raises_value_error():
    df = pd.DataFrame({
        "age": [25.0, None, 40.0],
        "income": [15.0, 20.0, 35.0],
        "house_price": [1.2, 1.8, 3.0],
    })
    feature_columns = ["age", "income"]
    target_column = "house_price"
    with pytest.raises(ValueError, match = "feature columns must not contain missing values"):
        validate_training_dataframe(df, feature_columns, target_column)

def test_missing_value_in_target_raises_value_error():
    df = pd.DataFrame({
        "age": [25.0, 30.0, 40.0],
        "income": [15.0, 20.0, 35.0],
        "house_price": [1.2, None, 3.0],
    })
    feature_columns = ["age", "income"]
    target_column = "house_price"
    with pytest.raises(ValueError, match = "target columns must not contain missing values"):
        validate_training_dataframe(df, feature_columns, target_column)

def test_non_numeric_feature_raises_value_error():
    df = pd.DataFrame({
        "age": [25, 30, 40],
        "city": ["Hanoi", "Hai Phong", "Hanoi"],
        "house_price": [1.2, 1.8, 3.0],
    })
    feature_columns = ["age", "city"]
    target_column = "house_price"
    with pytest.raises(ValueError, match = "feature columns must be numeric"):
        validate_training_dataframe(df, feature_columns, target_column)

def test_non_numeric_target_raises_value_error():
    df = pd.DataFrame({
        "age": [25, 30, 40],
        "income": [15.0, 20.0, 35.0],
        "house_price": ["low", "medium", "high"],
    })
    feature_columns = ["age", "income"]
    target_column = "house_price"
    with pytest.raises(ValueError, match="target column must be numeric"):
        validate_training_dataframe(df, feature_columns, target_column)