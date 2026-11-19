from splitting import time_ordered_split
from standardization import transform_standardizer, fit_standardizer
import numpy as np
import pytest

def prepare_regression_data(
    X,
    y,
    validation_ratio,
):
    X_train, X_valid, y_train, y_valid = time_ordered_split(
        X, y, validation_ratio
    )
    feature_means, feature_stds = fit_standardizer(X_train)
    X_train_scaled = transform_standardizer(
        X_train,
        feature_means,
        feature_stds,
    )
    X_valid_scaled = transform_standardizer(
        X_valid,
        feature_means,
        feature_stds,
    )
    return (
        X_train_scaled,
        X_valid_scaled,
        y_train,
        y_valid,
        feature_means,
        feature_stds,
    )

def test_prepare_regression_data_uses_training_statistics():
    X = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [100.0, 1000.0],
    ])
    y = np.array([10.0, 20.0, 30.0, 40.0, 1000.0])
    validation_ratio = 0.2
    (
        X_train_scaled,
        X_valid_scaled,
        y_train,
        y_valid,
        feature_means,
        feature_stds,
    ) = prepare_regression_data(
        X,
        y,
        validation_ratio
    )
    expected_means = np.array([2.5, 25.0])
    expected_stds = np.array([
        np.sqrt(1.25),
        np.sqrt(125.0),
    ])
    expected_y_train = np.array([10.0, 20.0, 30.0, 40.0])
    expected_y_valid = np.array([1000.0])
    feature_avg = X_train_scaled.mean(axis=0)
    assert X_train_scaled.shape == (4, 2)
    assert X_valid_scaled.shape == (1, 2)
    assert np.allclose(feature_avg, 0.0)
    assert np.allclose(feature_means, expected_means)
    assert np.allclose(feature_stds, expected_stds)
    assert np.allclose(y_train, expected_y_train)
    assert np.allclose(y_valid, expected_y_valid)