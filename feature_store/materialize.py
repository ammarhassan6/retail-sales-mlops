"""STEP 3b - Feast feature store: register features, fill the online store, verify.

  1. APPLY        register entities + feature view in the Feast registry
  2. MATERIALIZE  copy the newest feature values from the offline store
                  (Parquet) into the online store (SQLite)
  3. VERIFY       offline (point-in-time) and online lookups must return
                  exactly the features our pipeline computed -> no skew.
                  Any mismatch -> exit code 1 (pipeline stops).

Run:  python -m feature_store.materialize
"""

import sys
from datetime import timedelta

import numpy as np
import pandas as pd

from feature_store.client import (
    get_online_history_features,
    get_store,
    get_training_features,
    load_definitions,
)
from src.features.preprocessing import HISTORY_COLUMNS
from src.utils.config import load_config, project_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def apply_definitions(store) -> None:
    entities, feature_views = load_definitions()
    store.apply([entities.store, entities.item, feature_views.store_item_sales_stats])
    logger.info("Applied entities [store, item] and feature view '%s'", feature_views.FEATURE_VIEW_NAME)


def materialize_latest(store, feature_table: pd.DataFrame) -> None:
    newest = feature_table["event_timestamp"].max()
    # Only the last few days are needed: the online store keeps just the newest value per store-item
    store.materialize(start_date=newest - timedelta(days=7), end_date=newest + timedelta(seconds=1))
    logger.info("Materialized online store up to %s", newest)


def verify_offline(store, test: pd.DataFrame, sample_size: int = 500) -> bool:
    """Point-in-time features from Feast must equal the features in test.parquet."""
    sample = test.sample(min(sample_size, len(test)), random_state=42)
    entity_df = sample[["store", "item"]].copy()
    entity_df["event_timestamp"] = pd.to_datetime(sample["date"]).dt.tz_localize("UTC")

    from_feast = get_training_features(store, entity_df).sort_values(["store", "item", "event_timestamp"])
    expected = sample.sort_values(["store", "item", "date"])
    ok = np.allclose(from_feast[HISTORY_COLUMNS].to_numpy(), expected[HISTORY_COLUMNS].to_numpy())
    logger.info("Offline check (%d point-in-time rows): %s", len(sample), "MATCH" if ok else "MISMATCH")
    return ok


def verify_online(store, latest_stats: pd.DataFrame) -> bool:
    """Online values must equal the API's snapshot file latest_store_item_stats.parquet."""
    from_feast = get_online_history_features(store, latest_stats[["store", "item"]])
    merged = latest_stats.merge(from_feast, on=["store", "item"], suffixes=("", "_feast"))
    ok = len(merged) == len(latest_stats) and all(
        np.allclose(merged[col], merged[f"{col}_feast"]) for col in HISTORY_COLUMNS
    )
    logger.info("Online check (%d store-items): %s", len(latest_stats), "MATCH" if ok else "MISMATCH")
    logger.info("Example online lookup (store 1, item 1):\n%s",
                from_feast[(from_feast.store == 1) & (from_feast.item == 1)].to_string(index=False))
    return ok


def main() -> None:
    processed = project_path(load_config()["data"]["processed_dir"])
    feature_table = pd.read_parquet(processed / "store_item_features.parquet")

    store = get_store()
    apply_definitions(store)
    materialize_latest(store, feature_table)

    offline_ok = verify_offline(store, pd.read_parquet(processed / "test.parquet"))
    online_ok = verify_online(store, pd.read_parquet(processed / "latest_store_item_stats.parquet"))
    if not (offline_ok and online_ok):
        logger.error("Feature store verification FAILED")
        sys.exit(1)
    logger.info("Feature store ready and verified")


if __name__ == "__main__":
    main()
