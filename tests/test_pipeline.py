"""INTEGRATION TEST - validation -> features -> training -> prediction -> inventory
all working together on sample data."""

import pandas as pd

from src.data.validation import validate_sales
from src.features.preprocessing import (
    FEATURE_COLUMNS,
    TARGET,
    build_features,
    build_inference_frame,
    latest_store_item_stats,
)
from src.inventory.optimizer import recommend_order
from src.models.tune import build_model


def test_end_to_end_on_sample_data(sample_sales):
    passed, errors = validate_sales(sample_sales)
    assert passed, errors

    features = build_features(sample_sales)
    model = build_model("hist_gradient_boosting", {"max_iter": 50})
    model.fit(features[FEATURE_COLUMNS], features[TARGET])

    # forecast the next 7 days for store 1 / item 1, like the API will do
    stats = latest_store_item_stats(sample_sales)
    start = sample_sales["date"].max()
    requests = pd.DataFrame({"store": 1, "item": 1,
                             "date": pd.date_range(start + pd.Timedelta(days=1), periods=7)})
    X = build_inference_frame(requests, stats)[FEATURE_COLUMNS]
    forecasts = model.predict(X).clip(min=0).tolist()

    result = recommend_order(forecasts, daily_std=float(X["sales_std_28d"].iloc[0]),
                             current_stock=10, lead_time_days=7)
    assert len(forecasts) == 7
    assert result["recommended_order_quantity"] >= 0
