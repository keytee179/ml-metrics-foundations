import numpy as np
import pytest

from metrics import (
    calculate_mae_numpy,
    calculate_mse_numpy,
    calculate_rmse_numpy,
    calculate_r2_numpy,
)

def test_mae_normal_case():
    result = calculate_mae_numpy(
        [10, 20, 30],
        [12, 18, 35],
    )
    assert result == pytest.approx(
        3.0
    )

def test_mse_normal_case():
    result = calculate_mse_numpy(
        [10, 20, 30],
        [12, 18, 35],
    )
    assert result == pytest.approx(
        11.0
    )

def test_rmse_normal_case():
    result = calculate_rmse_numpy(
        [10, 20, 30],
        [12, 18, 35],
    )
    assert result == pytest.approx(
        11.0 ** 0.5
    )

def test_r2_perfect_prediction():
    result = calculate_r2_numpy(
        [10, 20, 30],
        [10, 20, 30],
    )
    assert result == pytest.approx(
        1.0
    )

def test_r2_baseline_prediction():
    result = calculate_r2_numpy(
        [10, 20, 30],
        [20, 20, 20],
    )
    assert result == pytest.approx(
        0.0
    )

def test_empty_input_raises_value_error():
    with pytest.raises(ValueError):
        calculate_mae_numpy(
            [],
            [],
        )

def test_different_shapes_raise_value_error():
    with pytest.raises(ValueError):
        calculate_mae_numpy(
            [10, 20, 30],
            [12, 18],
        )

def test_two_dimensional_input_raises_value_error():
    with pytest.raises(ValueError):
        calculate_mae_numpy(
            [
                [10],
                [20],
                [30],
            ],
            [
                [12],
                [18],
                [35],
            ],
        )

def test_r2_zero_variance_raises_value_error():
    with pytest.raises(
            ValueError
    ):
        calculate_r2_numpy(
            [5, 5, 5],
            [5, 5, 5],
        )

def test_r2_worse_than_baseline():
    result = calculate_r2_numpy(
        [1, 2, 3],
        [3, 3, 1],
    )
    assert result == pytest.approx(
        -3.5
    )