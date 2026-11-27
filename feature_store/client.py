"""Small helper functions to read features from Feast.

Used by: feature_store/materialize.py, the tests, and the Flask API (online lookup).
"""

import os
import sys
from pathlib import Path

import pandas as pd

from src.features.preprocessing import HISTORY_COLUMNS

os.environ.setdefault("FEAST_USAGE", "False")  # don't send anonymous usage stats to Feast (privacy)

from feast import FeatureStore  # noqa: E402  (must come after FEAST_USAGE is set)

REPO_PATH = Path(__file__).resolve().parent / "feature_repo"
FEATURE_VIEW_NAME = "store_item_sales_stats"
FEATURE_REFS = [f"{FEATURE_VIEW_NAME}:{name}" for name in HISTORY_COLUMNS]


def get_store(repo_path: Path = REPO_PATH) -> FeatureStore:
    return FeatureStore(repo_path=str(repo_path))


def load_definitions():
    """Import entities + feature view the same way `feast apply` does
    (the repo folder is added to the Python path)."""
    if str(REPO_PATH) not in sys.path:
        sys.path.insert(0, str(REPO_PATH))
    import entities
    import feature_views
    return entities, feature_views


def get_online_history_features(store: FeatureStore, pairs: pd.DataFrame) -> pd.DataFrame:
    """ONLINE lookup (API): latest history features for each [store, item] row.
    Unknown store-items come back with NaN values."""
    entity_rows = [{"store": int(s), "item": int(i)} for s, i in zip(pairs["store"], pairs["item"])]
    result = store.get_online_features(features=FEATURE_REFS, entity_rows=entity_rows).to_df()
    return result[["store", "item"] + HISTORY_COLUMNS]


def get_training_features(store: FeatureStore, entity_df: pd.DataFrame) -> pd.DataFrame:
    """OFFLINE lookup (training): point-in-time correct features.
    entity_df needs columns store, item, event_timestamp (UTC). For each row Feast
    returns the feature values that were valid AT that timestamp - never future values."""
    return store.get_historical_features(entity_df=entity_df, features=FEATURE_REFS).to_df()
