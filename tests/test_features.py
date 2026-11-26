"""UNIT TESTS for feature engineering."""

import pandas as pd
import pytest

from src.features.preprocessing import (
    FEATURE_COLUMNS,
    add_calendar_features,
    build_features,
    build_inference_frame,
    latest_store_item_stats,
    time_split,
)


def test_calendar_features():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-06", "2024-01-08"])})  # Saturday, Monday
    out = add_calendar_features(df)
    assert out["dayofweek"].tolist() == [5, 0]
    assert out["is_weekend"].tolist() == [1, 0]
    assert out["month"].tolist() == [1, 1]


def test_no_missing_features_after_build(sample_sales):
    features = build_features(sample_sales)
    assert features[FEATURE_COLUMNS].isna().sum().sum() == 0


def test_history_uses_only_past_days(sample_sales):
    """No data leakage: mean_7d on a day must equal the mean of the 7 days BEFORE it."""
    one = sample_sales[(sample_sales.store == 1) & (sample_sales.item == 1)].reset_index(drop=True)
    features = build_features(one)
    row = features.iloc[10]
    previous_7 = one[(one.date < row.date) & (one.date >= row.date - pd.Timedelta(days=7))]
    assert row["sales_mean_7d"] == pytest.approx(previous_7["sales"].mean())


def test_time_split_is_ordered_and_non_overlapping(sample_sales):
    features = build_features(sample_sales)
    train, val, test = time_split(features, val_days=30, test_days=30)
    assert train["date"].max() < val["date"].min()
    assert val["date"].max() < test["date"].min()
    assert len(train) + len(val) + len(test) == len(features)


def test_serving_features_match_training_features(sample_sales):
    """Training-serving skew test: features built for the API for day D must be
    identical to the features the training code builds for day D."""
    last_day = sample_sales["date"].max()
    history = sample_sales[sample_sales["date"] < last_day]

    stats = latest_store_item_stats(history)  # snapshot "as of" last_day
    requests = pd.DataFrame({"store": [1, 2], "item": [1, 3], "date": [last_day, last_day]})
    serving = build_inference_frame(requests, stats)

    training = build_features(sample_sales)
    training = training[training["date"] == last_day].merge(requests[["store", "item"]])

    pd.testing.assert_frame_equal(
        serving.sort_values(["store", "item"])[FEATURE_COLUMNS].reset_index(drop=True),
        training.sort_values(["store", "item"])[FEATURE_COLUMNS].reset_index(drop=True),
        check_dtype=False,
    )
