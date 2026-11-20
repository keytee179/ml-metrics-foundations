from end_to_end_preparation import build_and_prepare_regression_data
import pandas as pd
import numpy as np

def test_end_to_end_preparation():
    observations_df = pd.DataFrame({
        "event_time": [
            "2026-01-04 09:00",
            "2026-01-01 09:00",
            "2026-01-03 09:00",
            "2026-01-02 09:00",
        ],
        "machine_id": ["B", "A", "B", "A"],
        "temperature": [40.0, 10.0, 30.0, 20.0],
        "failure_score": [4.0, 1.0, 3.0, 2.0],
    })
    static_features_df = pd.DataFrame({
        "machine_id": ["A", "B"],
        "machine_age": [5.0, 8.0],
    })
    feature_columns = ["temperature", "machine_age"]
    target_column = "failure_score"
    validation_ratio = 0.25
    (
        training_df,
        X_train_scaled,
        X_valid_scaled,
        y_train,
        y_valid,
        feature_means,
        feature_stds,
    ) = build_and_prepare_regression_data(
        observations_df=observations_df,
        static_features_df=static_features_df,
        entity_column="machine_id",
        timestamp_column="event_time",
        feature_columns=feature_columns,
        target_column=target_column,
        validation_ratio=validation_ratio,
    )
    assert training_df["temperature"].tolist() == [10.0, 20.0, 30.0, 40.0]
    assert X_train_scaled.shape == (3, 2)
    assert X_valid_scaled.shape == (1, 2)
    assert y_train.shape == (3,)
    assert y_valid.shape == (1,)
    assert np.allclose(y_train, [1.0, 2.0, 3.0])
    assert np.allclose(y_valid, [4.0])
    assert np.allclose(X_train_scaled.mean(axis=0), [0.0, 0.0])
    assert np.allclose(X_train_scaled.std(axis=0), [1.0, 1.0])
    expected_feature_means = [20.0, 6.0]
    assert np.allclose(feature_means, expected_feature_means)
    expected_feature_stds = [8.16496580927726, 1.4142135623730951]
    assert np.allclose(feature_stds, expected_feature_stds)
    expected_X_valid_scaled = (np.array([[40.0, 8.0]]) - feature_means) / feature_stds
    assert np.allclose(X_valid_scaled,expected_X_valid_scaled)