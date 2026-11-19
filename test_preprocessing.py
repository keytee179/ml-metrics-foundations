import pandas as pd
import numpy as np
from preprocessing import prepare_regression_data

df = pd.DataFrame({
        "age": [10.0, 20.0, 30.0, 100.0],
        "income": [100.0, 200.0, 300.0, 1000.0],
        "house_price": [1.0, 2.0, 3.0, 10.0],
    })
feature_columns = ["age", "income"]
target_column = "house_price"
validation_ratio = 0.25
def test_prepare_regression_data_returns_correct_outputs():
    (
        X_train_scaled,
        X_valid_scaled,
        y_train,
        y_valid,
        feature_means,
        feature_stds,
    ) = prepare_regression_data(df, feature_columns, target_column, validation_ratio)
    expected_X_train_scaled_mean = [0.0, 0.0]
    expected_X_train_scaled_std = [1.0, 1.0]
    expected_feature_stds = np.array([
        np.std([10.0, 20.0, 30.0]),
        np.std([100.0, 200.0, 300.0]),
    ])
    expected_X_valid_scaled = (np.array([[100.0, 1000.0]]) - feature_means) / feature_stds
    assert np.allclose(
        X_valid_scaled,
        expected_X_valid_scaled,
    )
    assert X_train_scaled.shape == (3, 2)
    assert X_valid_scaled.shape == (1, 2)
    assert (y_train == np.array([1.0, 2.0, 3.0])).all()
    assert (y_valid == np.array([10.0])).all()
    assert np.allclose(feature_means, [20.0, 200.0])
    assert np.allclose(X_train_scaled.mean(axis = 0), expected_X_train_scaled_mean)
    assert np.allclose(X_train_scaled.std(axis = 0), expected_X_train_scaled_std)
    assert np.allclose(feature_stds, expected_feature_stds)

