import pytest
from feature_merging import merge_entity_features
import pandas as pd
from pandas.errors import MergeError

def test_merge_entity_features_preserves_observations():
    observations_df = pd.DataFrame({
        "machine_id": ["A", "A", "B"],
        "target": [1.2, 1.5, 2.0],
    })
    entity_features_df = pd.DataFrame({
        "machine_id": ["A", "B"],
        "value_mean": [105.0, 220.0],
        "observation_count": [2, 3],
    })
    merged_df = merge_entity_features(
        observations_df,
        entity_features_df,
        entity_column="machine_id",
    )
    assert merged_df.shape == (3, 4)
    assert merged_df["machine_id"].tolist() == ["A", "A", "B"]
    assert merged_df["value_mean"].tolist() == [105.0, 105.0, 220.0]
    assert merged_df["target"].tolist() == [1.2, 1.5, 2.0]
    assert "_merge" not in merged_df.columns

def test_duplicate_entity_features_raise_merge_error():
    observations_df = pd.DataFrame({
        "machine_id": ["A", "A", "B"],
        "target": [1.2, 1.5, 2.0],
    })
    entity_features_df = pd.DataFrame({
        "machine_id": ["A", "A", "B"],
        "value_mean": [105.0, 110.0, 220.0],
    })
    with pytest.raises(MergeError):
        merge_entity_features(
            observations_df,
            entity_features_df,
            entity_column="machine_id",
        )

