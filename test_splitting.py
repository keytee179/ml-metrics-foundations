from splitting import time_ordered_split
import pytest
import numpy as np
def test_normal_split_shapes():
    X = np.array([
        [10, 1],
        [20, 2],
        [30, 3],
        [40, 4],
        [50, 5],
    ])
    y = np.array([
        100,
        200,
        300,
        400,
        500,
    ])
    (
        X_train,
        X_valid,
        y_train,
        y_valid,
    ) = time_ordered_split(
        X,
        y,
        validation_ratio=0.4,
    )
    assert X_train.shape == (3, 2)
    assert X_valid.shape == (2, 2)
    assert y_train.shape == (3,)
    assert y_valid.shape == (2,)

def test_split_preserves_order():
    X = np.array([
        [10, 1],
        [20, 2],
        [30, 3],
        [40, 4],
        [50, 5],
    ])
    y = np.array([
        100,
        200,
        300,
        400,
        500,
    ])
    (
        X_train,
        X_valid,
        y_train,
        y_valid,
    ) = time_ordered_split(
        X,
        y,
        validation_ratio=0.4,
    )
    expected_X_valid = np.array([
        [40, 4],
        [50, 5],
    ])

    expected_y_valid = np.array([
        400,
        500,
    ])
    assert np.array_equal(X_valid, expected_X_valid)
    assert np.array_equal(y_valid, expected_y_valid)

def test_invalid_ratio_zero():
    X = np.array([
        [10],
        [20],
        [30],
    ])
    y = np.array([
        100,
        200,
        300,
    ])
    with pytest.raises(
            ValueError
    ):
        time_ordered_split(X, y, validation_ratio=0)

def test_invalid_ratio_one():
    X = np.array([
        [10],
        [20],
        [30],
    ])
    y = np.array([
        100,
        200,
        300,
    ])
    with pytest.raises(
            ValueError
    ):
        time_ordered_split(X, y, validation_ratio=1)

def test_mismatched_sample_count():
    X = np.array([
        [10],
        [20],
        [30],
    ])
    y = np.array([
        100,
        200,
    ])
    with pytest.raises(ValueError):
        time_ordered_split(X, y, validation_ratio=0.4)

def test_empty_input():
    X = np.array([])
    y = np.array([])
    with pytest.raises(ValueError):
        time_ordered_split(X, y, validation_ratio=0.4)

def test_ratio_creates_empty_validation():
    X = np.array([
        [1],
        [2],
        [3],
    ])
    y = np.array([
        10,
        20,
        30,
    ])
    validation_ratio = 0.1
    with pytest.raises(ValueError):
        time_ordered_split(X, y, validation_ratio=validation_ratio)