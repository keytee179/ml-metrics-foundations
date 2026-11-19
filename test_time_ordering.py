from time_ordering import prepare_time_ordered_dataframe
import pandas as pd
import numpy as np
import pytest

def test_prepare_time_ordered_dataframe_sorts_and_deduplicates():
    df = pd.DataFrame({
        "event_time": [
            "2026-01-03 09:00",
            "2026-01-01 09:00",
            "2026-01-02 09:00",
            "2026-01-02 09:00",
        ],
        "sensor_id": ["A", "A", "A", "A"],
        "temperature": [30.0, 10.0, 20.0, 22.0],
    })
    timestamp_column = "event_time"
    entity_column = "sensor_id"
    result_df = prepare_time_ordered_dataframe(df, timestamp_column, entity_column)
    assert result_df.shape == (3, 3)
    assert result_df.index.tolist() == [0, 1, 2]
    assert result_df["temperature"].tolist() == [10.0, 22.0, 30.0]
    assert pd.api.types.is_datetime64_any_dtype(result_df["event_time"])
    assert df["event_time"].tolist() == [
        "2026-01-03 09:00",
        "2026-01-01 09:00",
        "2026-01-02 09:00",
        "2026-01-02 09:00",
    ]

def test_non_dataframe_input_raises_type_error():
    df = [
        ["2026-01-03 09:00", "A", 30.0],
        ["2026-01-01 09:00", "A", 10.0],
    ]
    with pytest.raises(TypeError, match="df must be a pandas DataFrame"):
        prepare_time_ordered_dataframe(df, "event_time", "sensor_id")

def test_missing_timestamp_column_raises_value_error():
    df = pd.DataFrame({
        "sensor_id": ["A", "A"],
        "temperature": [10.0, 20.0],
    })
    with pytest.raises(ValueError, match="Missing timestamp column: event_time"):
        prepare_time_ordered_dataframe(df, "event_time", "sensor_id")

def test_invalid_timestamp_raises_error():
    df = pd.DataFrame({
        "event_time": ["2026-01-01 09:00", "not-a-date"],
        "sensor_id": ["A", "A"],
        "temperature": [10.0, 20.0],
    })
    with pytest.raises(ValueError):
        prepare_time_ordered_dataframe(df, "event_time", "sensor_id")
