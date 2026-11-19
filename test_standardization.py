import pytest

import numpy as np
from standardization import (
    fit_standardizer,
    transform_standardizer,
)

def test_fit_standardizer_returns_feature_statistics():
    X_train = np.array([
        [10.0, 100.0],
        [20.0, 200.0],
        [30.0, 300.0],
    ])
    feature_means, feature_stds = (
        fit_standardizer(
            X_train
        )
    )
    expected_means = np.array([
        20.0,
        200.0,
    ])
    expected_stds = np.array([
        8.16496581,
        81.64965809,
    ])
    assert np.allclose(
        feature_means,
        expected_means,
    )
    assert np.allclose(
        feature_stds,
        expected_stds,
    )

def test_transform_training_has_zero_mean_and_unit_std():
    X_train = np.array([
        [10.0, 100.0],
        [20.0, 200.0],
        [30.0, 300.0],
    ])
    feature_means, feature_stds = (
        fit_standardizer(
            X_train
        )
    )
    X_train_scaled = transform_standardizer(
        X_train,
        feature_means,
        feature_stds,
    )
    expected_means = np.array([
        0.0,
        0.0,
    ])
    expected_stds = np.array([
        1.0,
        1.0,
    ])
    scaled_means = X_train_scaled.mean(
        axis=0
    )
    scaled_stds = X_train_scaled.std(
        axis=0
    )
    assert np.allclose(
        scaled_means,
        expected_means,
    )
    assert np.allclose(
        scaled_stds,
        expected_stds,
    )

def test_transform_validation_uses_training_statistics():
    X_train = np.array([
        [10.0, 100.0],
        [20.0, 200.0],
        [30.0, 300.0],
    ])
    X_valid = np.array([
        [40.0, 400.0],
        [50.0, 500.0],
    ])
    feature_means, feature_stds = (
        fit_standardizer(
            X_train
        )
    )
    transformed_X_valid = transform_standardizer(
        X_valid,
        feature_means,
        feature_stds,
    )
    expected_X_valid_scaled = np.array([
        [2.44948974, 2.44948974],
        [3.67423461, 3.67423461],
    ])
    assert np.allclose(
        expected_X_valid_scaled,
        transformed_X_valid,
    )

def test_fit_rejects_zero_variance_feature():
    X_train = np.array([
        [10.0, 5.0],
        [20.0, 5.0],
        [30.0, 5.0],
    ])
    with pytest.raises(ValueError):
        fit_standardizer(
            X_train
        )

def test_transform_rejects_statistics_with_wrong_feature_count():
    X = np.array([
        [10.0, 100.0],
        [20.0, 200.0],
    ])
    feature_means = np.array([
        15.0,
    ])
    feature_stds = np.array([
        5.0,
    ])
    with pytest.raises(ValueError):
        transform_standardizer(
            X,
            feature_means,
            feature_stds,
        )

def test_transform_rejects_two_dimensional_statistics():
    X = np.array([
        [10.0, 100.0],
        [20.0, 200.0],
    ])
    feature_means = np.array([
        [15.0, 150.0],
    ])
    feature_stds = np.array([
        [5.0, 50.0],
    ])
    with pytest.raises(ValueError):
        transform_standardizer(
            X,
            feature_means,
            feature_stds,
        )