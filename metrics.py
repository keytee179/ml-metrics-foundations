import numpy as np
def validate_regression_inputs(
    y_true,
    y_pred,
):
    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must "
            "have the same length."
        )

    if len(y_true) == 0:
        raise ValueError(
            "Inputs must not be empty."
        )

def calculate_mae(
    y_true,
    y_pred,
):
    validate_regression_inputs(
        y_true,
        y_pred,
    )
    absolute_total = 0
    for i in range(
        len(y_true)
    ):
        error = (
            y_true[i]
            - y_pred[i]
        )
        absolute_total += abs(
            error
        )
    return (
        absolute_total
        / len(y_true)
    )
def calculate_mse(
    y_true,
    y_pred,
):
    validate_regression_inputs(
        y_true,
        y_pred,
    )
    squared_total = 0
    for i in range(
        len(y_true)
    ):
        error = (
            y_true[i]
            - y_pred[i]
        )
        squared_total += (
            error ** 2
        )
    return (
        squared_total
        / len(y_true)
    )
def calculate_rmse(
    y_true,
    y_pred,
):
    mse = calculate_mse(
        y_true,
        y_pred,
    )
    return mse ** 0.5
def calculate_mae_numpy(
    y_true,
    y_pred,
):
    y_true_array = np.asarray(
        y_true,
        dtype=float,
    )
    y_pred_array = np.asarray(
        y_pred,
        dtype=float,
    )
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(
            "y_true_array and y_pred_array must be the same shape."
        )
    if y_true_array.ndim != 1 or y_pred_array.ndim != 1:
        raise ValueError(
            "Inputs must be one-dimensional."
        )
    if y_true_array.size == 0:
        raise ValueError(
            "Inputs must not be empty."
        )
    errors = (
            y_true_array
            - y_pred_array
    )
    absolute_errors = np.abs(
        errors
    )
    mae = np.mean(
        absolute_errors
    )
    return float(mae)
def validate_regression_arrays(
    y_true,
    y_pred,
):
    y_true_array = np.asarray(
        y_true,
        dtype=float,
    )
    y_pred_array = np.asarray(
        y_pred,
        dtype=float,
    )
    if y_true_array.ndim != 1 or y_pred_array.ndim != 1:
        raise ValueError(
            "Inputs must be one-dimensional."
        )
    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(
            "Inputs must have the same shape."
        )
    if y_pred_array.size == 0:
        raise ValueError(
            "Inputs must not be empty."
        )
    return (
        y_true_array,
        y_pred_array,
    )
def calculate_mse_numpy(
    y_true,
    y_pred,
):
    (
        y_true_array,
        y_pred_array,
    ) = validate_regression_arrays(
        y_true,
        y_pred,
    )
    errors = y_true_array - y_pred_array
    squared_errors = errors**2
    mse = np.mean(squared_errors)
    return float(mse)
def calculate_rmse_numpy(
    y_true,
    y_pred,
):
    mse = calculate_mse_numpy(y_true, y_pred)
    return float(mse**0.5)
def calculate_r2_numpy(
    y_true,
    y_pred,
):
    (
        y_true_array,
        y_pred_array,
    ) = validate_regression_arrays(
        y_true,
        y_pred,
    )
    residuals = y_true_array - y_pred_array
    ss_res = np.sum(
        residuals**2
    )
    y_true_mean = np.mean(
        y_true_array
    )
    baseline_errors = y_true_array - y_true_mean
    ss_tot = np.sum(
        baseline_errors**2
    )
    if ss_tot == 0:
        raise ValueError(
            "R2 is undefined when "
            "y_true has zero variance."
        )

    r2 = 1 - ss_res/ss_tot
    return float(r2)
