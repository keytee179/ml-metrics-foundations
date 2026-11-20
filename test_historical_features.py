import numpy as np
import pandas as pd
from historical_features import add_historical_mean_feature

def test_add_historical_mean_feature_uses_only_prior_entity_rows():
    df = pd.DataFrame({
        "event_time": [
            "2026-01-03",
            "2026-01-01",
            "2026-01-02",
            "2026-01-02",
            "2026-01-04",
        ],
        "machine_id": ["A", "A", "B", "A", "B"],
        "temperature": [30.0, 10.0, 40.0, 20.0, 60.0],
    })
    result_df = add_historical_mean_feature(
        df,
        entity_column="machine_id",
        timestamp_column="event_time",
        value_column="temperature",
        output_column="historical_mean_temperature",
    )
    assert result_df["machine_id"].tolist() == ["A", "A", "B", "A", "B"]
    assert result_df["temperature"].tolist() == [10.0, 20.0, 40.0, 30.0, 60.0]
    expected = np.array([
        np.nan,
        10.0,
        np.nan,
        15.0,
        40.0,
    ])
    assert np.allclose(
        result_df["historical_mean_temperature"],
        expected,
        equal_nan=True,
    )
    assert df["event_time"].tolist() == [
        "2026-01-03",
        "2026-01-01",
        "2026-01-02",
        "2026-01-02",
        "2026-01-04",
    ]