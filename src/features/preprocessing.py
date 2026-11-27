"""STEP 3 - Feature engineering + train/val/test split.

Features used by the model (FEATURE_COLUMNS):
  * store, item                        -> which shop / which product
  * calendar features                  -> day of week, month, year, ... (seasonality)
  * sales_mean_7d / sales_mean_28d     -> average sales of the previous 7 / 28 days
  * sales_std_28d                      -> how much sales jump around (also used for safety stock)

IMPORTANT (no data leakage): the history features for day D use ONLY days
before D (we shift by 1 day before computing the rolling average).

The SAME functions are used by training and by the API, so training and
serving features are always computed identically (no training-serving skew).

Run:  python -m src.features.preprocessing
"""

import pandas as pd

from src.utils.config import load_config, project_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET = "sales"
CALENDAR_COLUMNS = ["dayofweek", "month", "year", "dayofmonth", "weekofyear", "is_weekend"]
HISTORY_COLUMNS = ["sales_mean_7d", "sales_mean_28d", "sales_std_28d"]
FEATURE_COLUMNS = ["store", "item"] + CALENDAR_COLUMNS + HISTORY_COLUMNS


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["date"]
    df["dayofweek"] = d.dt.dayofweek          # Monday=0 ... Sunday=6
    df["month"] = d.dt.month
    df["year"] = d.dt.year
    df["dayofmonth"] = d.dt.day
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling statistics per store-item, using only PAST days (shift(1))."""
    df = df.sort_values(["store", "item", "date"]).copy()
    sales_by_group = df.groupby(["store", "item"])["sales"]

    df["sales_mean_7d"] = sales_by_group.transform(lambda s: s.shift(1).rolling(7).mean())
    df["sales_mean_28d"] = sales_by_group.transform(lambda s: s.shift(1).rolling(28).mean())
    df["sales_std_28d"] = sales_by_group.transform(lambda s: s.shift(1).rolling(28).std())
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature table. The first 28 days of each store-item have no
    history yet, so those rows are dropped."""
    df = add_history_features(add_calendar_features(df))
    return df.dropna(subset=HISTORY_COLUMNS).reset_index(drop=True)


def time_split(df: pd.DataFrame, val_days: int, test_days: int):
    """Split by DATE (not randomly): train = oldest, val = middle, test = newest."""
    last_date = df["date"].max()
    test_start = last_date - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)

    train = df[df["date"] < val_start]
    val = df[(df["date"] >= val_start) & (df["date"] < test_start)]
    test = df[df["date"] >= test_start]
    return train, val, test


def latest_store_item_stats(sales: pd.DataFrame) -> pd.DataFrame:
    """History features for the day AFTER the last date in the data.

    This snapshot is what the API (and later the Feast online store) uses at
    prediction time. We reuse add_history_features() on a fake 'next day' row
    so the logic is exactly the same as in training.
    """
    next_day = sales["date"].max() + pd.Timedelta(days=1)
    future_rows = sales[["store", "item"]].drop_duplicates().assign(date=next_day, sales=float("nan"))
    with_future = add_history_features(pd.concat([sales, future_rows], ignore_index=True))

    snapshot = with_future[with_future["date"] == next_day]
    snapshot = snapshot[["store", "item"] + HISTORY_COLUMNS]
    snapshot.insert(2, "as_of_date", next_day)
    return snapshot.reset_index(drop=True)


def build_feature_store_table(sales: pd.DataFrame) -> pd.DataFrame:
    """Source table for the Feast feature store (offline store).

    One row per store-item-day with the history features "as of" that day,
    PLUS one extra row per store-item for the day after the last data day
    (that newest row is what the online store will serve to the API).
    Feast needs a timezone-aware timestamp column -> event_timestamp (UTC).
    """
    next_day = sales["date"].max() + pd.Timedelta(days=1)
    future_rows = sales[["store", "item"]].drop_duplicates().assign(date=next_day, sales=float("nan"))
    with_history = add_history_features(pd.concat([sales, future_rows], ignore_index=True))

    table = with_history.dropna(subset=HISTORY_COLUMNS)
    table = table[["store", "item", "date"] + HISTORY_COLUMNS].rename(columns={"date": "event_timestamp"})
    table["event_timestamp"] = table["event_timestamp"].dt.tz_localize("UTC")
    return table.reset_index(drop=True)


def build_inference_frame(requests: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Used by the API: requests has columns [store, item, date].
    Joins the latest history stats and adds calendar features."""
    requests = requests.copy()
    requests["date"] = pd.to_datetime(requests["date"])
    merged = requests.merge(stats, on=["store", "item"], how="left")
    return add_calendar_features(merged)


def main() -> None:
    cfg = load_config()
    sales = pd.read_parquet(project_path(cfg["data"]["validated_path"]))

    features = build_features(sales)
    train, val, test = time_split(features, cfg["split"]["val_days"], cfg["split"]["test_days"])

    out_dir = project_path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    latest_store_item_stats(sales).to_parquet(out_dir / "latest_store_item_stats.parquet", index=False)
    build_feature_store_table(sales).to_parquet(out_dir / "store_item_features.parquet", index=False)

    logger.info("Features built: train=%d, val=%d, test=%d rows", len(train), len(val), len(test))


if __name__ == "__main__":
    main()
