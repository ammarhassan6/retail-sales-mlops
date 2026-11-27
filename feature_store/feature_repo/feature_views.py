"""Feature VIEW = a group of features + its entities + its source + how fresh it must be.

Note: Feast loads the files in this folder as top-level modules, so we import
`entities` and `features` directly (not as `feature_store.feature_repo.xxx`).
"""

from datetime import timedelta

from feast import FeatureView

from entities import item, store  # isort: skip  (local files of this feature repo)
from features import ENTITY_FIELDS, FEATURE_TABLE_PATH, SALES_HISTORY_FIELDS, make_source  # isort: skip

FEATURE_VIEW_NAME = "store_item_sales_stats"


def make_feature_view(parquet_path=FEATURE_TABLE_PATH) -> FeatureView:
    return FeatureView(
        name=FEATURE_VIEW_NAME,
        entities=[store, item],
        # Freshness: a feature value older than 30 days is treated as stale/missing
        ttl=timedelta(days=30),
        schema=ENTITY_FIELDS + SALES_HISTORY_FIELDS,
        source=make_source(parquet_path),
        online=True,
        owner="person-a-data-ml",
        description="Rolling sales statistics per store-item, computed from previous days only",
        tags={"team": "retail-sales-mlops", "updated_by": "dvc preprocess stage"},
    )


# Module-level object so `feast apply` (CLI) can find it
store_item_sales_stats = make_feature_view()
