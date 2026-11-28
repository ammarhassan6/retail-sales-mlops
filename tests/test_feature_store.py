"""FEATURE STORE TESTS (Feast).

1. The source table: schema, entity keys, timestamps, freshness, nulls, valid values.
2. Training-serving consistency: the table matches the training features and the API snapshot.
3. A real Feast store in a temp folder: apply -> materialize -> online + offline retrieval.
"""

import numpy as np
import pandas as pd
import pytest

from feature_store.client import (
    get_online_history_features,
    get_training_features,
    load_definitions,
)
from src.features.preprocessing import (
    HISTORY_COLUMNS,
    build_feature_store_table,
    build_features,
    latest_store_item_stats,
)


@pytest.fixture
def feature_table(sample_sales):
    return build_feature_store_table(sample_sales)


# ---------- 1. the source table ----------

def test_schema_and_types(feature_table):
    assert list(feature_table.columns) == ["store", "item", "event_timestamp"] + HISTORY_COLUMNS
    assert pd.api.types.is_integer_dtype(feature_table["store"])
    assert pd.api.types.is_integer_dtype(feature_table["item"])
    for col in HISTORY_COLUMNS:
        assert pd.api.types.is_float_dtype(feature_table[col])


def test_entity_key_and_timestamp_are_unique(feature_table):
    assert not feature_table.duplicated(["store", "item", "event_timestamp"]).any()


def test_timestamps_are_utc(feature_table):
    assert str(feature_table["event_timestamp"].dt.tz) == "UTC"


def test_freshness_newest_row_is_day_after_last_sale(sample_sales, feature_table):
    expected = (sample_sales["date"].max() + pd.Timedelta(days=1)).tz_localize("UTC")
    newest = feature_table.groupby(["store", "item"])["event_timestamp"].max()
    assert (newest == expected).all()  # every store-item has a fresh value


def test_no_nulls_and_valid_values(feature_table):
    assert feature_table.isna().sum().sum() == 0
    assert (feature_table[HISTORY_COLUMNS] >= 0).all().all()  # means and std can't be negative


# ---------- 2. training-serving consistency ----------

def test_table_matches_training_features(sample_sales, feature_table):
    training = build_features(sample_sales)
    table = feature_table.assign(date=feature_table["event_timestamp"].dt.tz_localize(None))
    merged = training.merge(table, on=["store", "item", "date"], suffixes=("", "_fs"))
    assert len(merged) == len(training)
    for col in HISTORY_COLUMNS:
        assert np.allclose(merged[col], merged[f"{col}_fs"])


def test_table_matches_api_snapshot(sample_sales, feature_table):
    snapshot = latest_store_item_stats(sample_sales)
    newest = feature_table[feature_table["event_timestamp"] == feature_table["event_timestamp"].max()]
    merged = snapshot.merge(newest, on=["store", "item"], suffixes=("", "_fs"))
    assert len(merged) == len(snapshot)
    for col in HISTORY_COLUMNS:
        assert np.allclose(merged[col], merged[f"{col}_fs"])


def test_definitions_match_pipeline_feature_names():
    entities, feature_views = load_definitions()
    view = feature_views.store_item_sales_stats
    assert [f.name for f in view.features] == HISTORY_COLUMNS
    assert entities.store.join_key == "store" and entities.item.join_key == "item"


# ---------- 3. a real Feast store in a temporary folder ----------

# `temp_store` comes from tests/conftest.py (shared with the API tests)


def test_online_retrieval_returns_latest_values(sample_sales, temp_store):
    snapshot = latest_store_item_stats(sample_sales)
    online = get_online_history_features(temp_store, snapshot[["store", "item"]])
    merged = snapshot.merge(online, on=["store", "item"], suffixes=("", "_online"))
    for col in HISTORY_COLUMNS:
        assert np.allclose(merged[col], merged[f"{col}_online"])


def test_online_unknown_store_item_is_missing(temp_store):
    online = get_online_history_features(temp_store, pd.DataFrame({"store": [999], "item": [1]}))
    assert online[HISTORY_COLUMNS].isna().all().all()


def test_offline_retrieval_is_point_in_time_correct(sample_sales, temp_store):
    training = build_features(sample_sales).sample(50, random_state=1)
    entity_df = training[["store", "item"]].assign(event_timestamp=training["date"].dt.tz_localize("UTC"))
    offline = get_training_features(temp_store, entity_df)
    merged = training.assign(event_timestamp=entity_df["event_timestamp"]).merge(
        offline, on=["store", "item", "event_timestamp"], suffixes=("", "_offline"))
    assert len(merged) == 50
    for col in HISTORY_COLUMNS:
        assert np.allclose(merged[col], merged[f"{col}_offline"])
