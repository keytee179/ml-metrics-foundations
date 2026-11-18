import numpy as np
X_train = np.array([
    [10.0, 100.0],
    [20.0, 200.0],
    [30.0, 300.0],
])
X_valid = np.array([
    [40.0, 400.0],
    [50.0, 500.0],
])

def fit_standardizer(
    X_train,
):
    X_train_array = np.asarray(
        X_train,
        dtype=float,
    )
    if X_train_array.ndim != 2:
        raise ValueError(
            "X_train must be two-dimensional."
        )
    if X_train_array.shape[0] == 0:
        raise ValueError(
            "X_train must not be empty."
        )
    feature_means = X_train_array.mean(
        axis=0
    )
    feature_stds = X_train_array.std(
        axis=0
    )
    if np.any(
        feature_stds == 0
    ):
        raise ValueError(
            "Features must have non-zero "
            "standard deviation."
        )
    return (
        feature_means,
        feature_stds,
    )

def transform_standardizer(
    X,
    feature_means,
    feature_stds,
):
    X_array = np.asarray(
        X,
        dtype=float,
    )
    means_array = np.asarray(
        feature_means,
        dtype=float,
    )
    stds_array = np.asarray(
        feature_stds,
        dtype=float,
    )
    if X_array.ndim != 2:
        raise ValueError(
            "X must be two-dimensional."
        )
    if means_array.ndim != 1 or stds_array.ndim != 1:
        raise ValueError(
            "Statistics must be one-dimensional."
        )
    if means_array.shape[0] != X_array.shape[1] or stds_array.shape[0] != X_array.shape[1]:
        raise ValueError(
            "Statistics must match the number of features."
        )
    if np.any(stds_array == 0):
        raise ValueError(
            "Standard deviations must be non-zero."
        )
    X_scaled = (
        X_array
        - means_array
    ) / stds_array
    return X_scaled

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

X_valid_scaled = transform_standardizer(
    X_valid,
    feature_means,
    feature_stds,
)