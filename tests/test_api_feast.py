"""API TESTS for the Feast path of get_history_features().

The API must give the SAME answer whether history features come from the
Feast online store or from the parquet snapshot, and must fall back to the
snapshot if Feast fails.
"""

import pytest

from src.api.app import create_app
from src.features.preprocessing import FEATURE_COLUMNS, TARGET, build_features, latest_store_item_stats
from src.models.tune import build_model

BODY = {"store": 1, "item": 1, "start_date": "2017-02-04", "horizon_days": 7,
        "current_stock": 120, "lead_time_days": 7}


@pytest.fixture
def artifacts(sample_sales):
    features = build_features(sample_sales)
    model = build_model("hist_gradient_boosting", {"max_iter": 50})
    model.fit(features[FEATURE_COLUMNS], features[TARGET])
    return {
        "model": model,
        "stats": latest_store_item_stats(sample_sales),
        "metadata": {"model_type": "hist_gradient_boosting"},
        "metrics": {"registered_version": "7"},
        "feast_store": None,
    }


class BrokenStore:
    """Pretends to be a Feast store whose online database is unreachable."""

    def get_online_features(self, *args, **kwargs):
        raise ConnectionError("online store down")


def test_model_info_shows_feature_source(artifacts, temp_store):
    parquet_info = create_app(artifacts).test_client().get("/model-info").get_json()
    feast_info = create_app({**artifacts, "feast_store": temp_store}).test_client().get("/model-info").get_json()
    assert parquet_info["feature_source"] == "parquet_snapshot"
    assert feast_info["feature_source"] == "feast_online"


def test_feast_and_parquet_give_same_prediction(artifacts, temp_store):
    from_parquet = create_app(artifacts).test_client().post("/predict", json=BODY)
    from_feast = create_app({**artifacts, "feast_store": temp_store}).test_client().post("/predict", json=BODY)
    assert from_feast.status_code == 200
    assert from_feast.get_json() == from_parquet.get_json()


def test_unknown_store_item_via_feast_is_400(artifacts, temp_store):
    client = create_app({**artifacts, "feast_store": temp_store}).test_client()
    response = client.post("/predict", json={**BODY, "store": 999})
    assert response.status_code == 400


def test_falls_back_to_parquet_when_feast_fails(artifacts):
    client = create_app({**artifacts, "feast_store": BrokenStore()}).test_client()
    response = client.post("/predict", json=BODY)
    assert response.status_code == 200
    assert len(response.get_json()["forecast"]) == 7
