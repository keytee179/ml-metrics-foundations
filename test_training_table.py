import pandas as pd
from training_table import build_training_table

def test_build_training_table_integrates_modules_correctly():
    observations_df = pd.DataFrame({
        "event_time": [
            "2026-01-03 09:00",
            "2026-01-01 09:00",
            "2026-01-02 09:00",
            "2026-01-02 09:00",
        ],
        "machine_id": ["B", "A", "A", "A"],
        "temperature": [30.0, 10.0, 20.0, 22.0],
        "failure_score": [3.0, 1.0, 2.0, 2.2],
    })
    static_features_df = pd.DataFrame({
        "machine_id": ["A", "B"],
        "machine_age": [5.0, 2.0],
    })
    training_df = build_training_table(
        observations_df,
        static_features_df,
        entity_column="machine_id",
        timestamp_column="event_time",
        feature_columns=["temperature", "machine_age"],
        target_column="failure_score",
    )
    assert training_df.shape == (3, 5)
    assert training_df.columns.tolist() == [
        "event_time",
        "machine_id",
        "temperature",
        "machine_age",
        "failure_score",
    ]
    assert training_df["machine_id"].tolist() == ["A", "A", "B"]
    assert training_df["temperature"].tolist() == [10.0, 22.0, 30.0]
    assert training_df["machine_age"].tolist() == [5.0, 5.0, 2.0]
    assert training_df["failure_score"].tolist() == [1.0, 2.2, 3.0]