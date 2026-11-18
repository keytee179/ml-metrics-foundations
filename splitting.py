import numpy as np

def time_ordered_split(
    X,
    y,
    validation_ratio,
):
    X_array = np.asarray(
        X
    )
    y_array = np.asarray(
        y
    )
    if X_array.ndim < 1:
        raise ValueError(
            "X must contain a sample axis."
        )
    if y_array.ndim != 1:
        raise ValueError(
            "y must be one-dimensional."
        )
    if (
        validation_ratio <= 0
        or validation_ratio >= 1
    ):
        raise ValueError(
            "validation_ratio must be "
            "between 0 and 1."
        )
    if (
        X_array.shape[0]
        != y_array.shape[0]
    ):
        raise ValueError(
            "X and y must contain "
            "the same number of samples."
        )
    if (
        X_array.size == 0
        or y_array.size == 0
    ):
        raise ValueError(
            "Inputs must not be empty."
        )
    n_samples = X_array.shape[0]
    validation_size = int(
        n_samples * validation_ratio
    )
    if validation_size == 0:
        raise ValueError(
            "validation_ratio creates "
            "an empty validation set."
        )
    split_index = (
        n_samples
        - validation_size
    )
    X_train = X_array[:split_index]
    X_valid = X_array[split_index:]
    y_train = y_array[:split_index]
    y_valid = y_array[split_index:]
    return (
        X_train,
        X_valid,
        y_train,
        y_valid,
    )