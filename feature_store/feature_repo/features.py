"""Feature DEFINITIONS: names, data types and the data source they come from.

The source file is created by the DVC 'preprocess' stage
(src/features/preprocessing.py -> build_feature_store_table).
"""

from pathlib import Path

from feast import Field, FileSource
from feast.types import Float64, Int64

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_TABLE_PATH = PROJECT_ROOT / "data" / "processed" / "store_item_features.parquet"

# The 3 history features - same names as HISTORY_COLUMNS in preprocessing.py
SALES_HISTORY_FIELDS = [
    Field(name="sales_mean_7d", dtype=Float64, description="Average daily sales over the previous 7 days"),
    Field(name="sales_mean_28d", dtype=Float64, description="Average daily sales over the previous 28 days"),
    Field(name="sales_std_28d", dtype=Float64,
          description="Std. deviation of daily sales over the previous 28 days (used for safety stock)"),
]

ENTITY_FIELDS = [
    Field(name="store", dtype=Int64),
    Field(name="item", dtype=Int64),
]


def make_source(parquet_path: Path | str = FEATURE_TABLE_PATH) -> FileSource:
    """Offline data source. A function so tests can point it at a temporary file."""
    return FileSource(
        name="store_item_features_source",
        path=str(parquet_path),
        timestamp_field="event_timestamp",
        description="Daily store-item history features built by the DVC preprocess stage",
    )
