import pandas as pd
from entity_aggregation import aggregate_entity_features


def test_aggregate_entity_features_returns_correct_summary():
    df = pd.DataFrame({
        "machine_id": ["B", "A", "B", "A", "B"],
        "pressure": [200.0, 100.0, 220.0, 110.0, 240.0],
    })
    summary_df = aggregate_entity_features(
        df,
        entity_column="machine_id",
        value_column="pressure",
    )
    assert summary_df.shape == (2, 4)
    assert summary_df.columns.tolist() == ["machine_id", "value_mean", "value_max", "observation_count"]
    assert summary_df["machine_id"].tolist() == ["A", "B"]
    assert summary_df["value_mean"].tolist() == [105, 220]
    assert summary_df["value_max"].tolist() == [110, 240]
    assert summary_df["observation_count"].tolist() == [2, 3]