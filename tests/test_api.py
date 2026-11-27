"""API TESTS for the Flask REST API (src/api/app.py).

We train a tiny model on FAKE sample data, so these tests run in seconds
(and in GitHub Actions) without the real model file or DagsHub access.
Flask's test_client() sends requests without starting a real web server.
"""

import pandas as pd
import pytest

from src.api.app import create_app
from src.data.sample_data import generate_sample_sales
from src.features.preprocessing import FEATURE_COLUMNS, TARGET, build_features, latest_store_item_stats
from src.models.tune import build_model


@pytest.fixture(scope="module")
def artifacts():
    """Same kind of objects load_artifacts() returns, but small and fake.
    scope="module" -> the model is trained once for all tests in this file."""
    sales = generate_sample_sales(n_stores=2, n_items=3, days=400)  # ends 2017-02-03
    sales["date"] = pd.to_datetime(sales["date"])
    features = build_features(sales)
    model = build_model("hist_gradient_boosting", {"max_iter": 50})
    model.fit(features[FEATURE_COLUMNS], features[TARGET])
    return {
        "model": model,
        "stats": latest_store_item_stats(sales),
        "metadata": {"model_type": "hist_gradient_boosting", "params": {"max_iter": 50},
                     "feature_columns": FEATURE_COLUMNS},
        "metrics": {"registered_version": "7", "test": {"smape": 12.0}, "quality_gate_passed": True},
    }


@pytest.fixture
def client(artifacts):
    return create_app(artifacts).test_client()


def valid_body(**changes) -> dict:
    body = {"store": 1, "item": 1, "start_date": "2017-02-04", "horizon_days": 7,
            "current_stock": 120, "lead_time_days": 7}
    body.update(changes)
    return body


# ---------------- /health and /model-info ----------------

def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    response = client.get("/model-info")
    info = response.get_json()
    assert response.status_code == 200
    assert info["model_version"] == "7"
    assert info["model_type"] == "hist_gradient_boosting"
    assert info["history_as_of_date"] == "2017-02-04"
    assert info["store_item_count"] == 6


def test_not_ready_when_model_missing():
    # create_app(None) = model files were missing at startup
    client = create_app(None).test_client()
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json=valid_body()).status_code == 503


# ---------------- /predict: valid requests ----------------

def test_predict_valid_request(client):
    response = client.post("/predict", json=valid_body())
    result = response.get_json()

    assert response.status_code == 200
    assert result["model_version"] == "7"
    assert len(result["forecast"]) == 7
    assert result["forecast"][0]["date"] == "2017-02-04"
    assert result["forecast"][6]["date"] == "2017-02-10"
    for day in result["forecast"]:
        assert day["predicted_sales"] >= 0
    for key in ["lead_time_demand", "safety_stock", "reorder_point", "should_reorder",
                "recommended_order_quantity"]:
        assert key in result["inventory"]


def test_predict_uses_defaults_for_optional_fields(client):
    body = valid_body()
    del body["horizon_days"]
    del body["lead_time_days"]
    response = client.post("/predict", json=body)
    assert response.status_code == 200
    assert len(response.get_json()["forecast"]) == 7   # default_horizon_days in api_config.yaml


def test_reorder_decision_depends_on_stock(client):
    empty = client.post("/predict", json=valid_body(current_stock=0)).get_json()["inventory"]
    full = client.post("/predict", json=valid_body(current_stock=100000)).get_json()["inventory"]
    assert empty["should_reorder"] is True
    assert empty["recommended_order_quantity"] > 0
    assert full["should_reorder"] is False
    assert full["recommended_order_quantity"] == 0


# ---------------- /predict: bad input -> 400 ----------------

def test_malformed_json(client):
    response = client.post("/predict", data="{not valid json", content_type="application/json")
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_body_must_be_an_object(client):
    response = client.post("/predict", json=[1, 2, 3])
    assert response.status_code == 400


def test_missing_field(client):
    body = valid_body()
    del body["store"]
    response = client.post("/predict", json=body)
    assert response.status_code == 400
    assert response.get_json()["error"] == "missing required field: store"


@pytest.mark.parametrize("changes", [
    {"store": "one"},                 # wrong type
    {"store": True},                  # bool is not an int for us
    {"item": -3},                     # must be positive
    {"start_date": "2017-13-45"},     # impossible date
    {"start_date": 20170204},         # not a string
    {"current_stock": -5},            # negative stock
    {"current_stock": float("nan")},  # NaN would break the maths
    {"horizon_days": 0},
    {"horizon_days": 2.5},
    {"lead_time_days": 0},
    {"lead_time_days": 10, "horizon_days": 7},  # need a forecast for every lead-time day
])
def test_invalid_fields(client, changes):
    response = client.post("/predict", json=valid_body(**changes))
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_unknown_store_item(client):
    response = client.post("/predict", json=valid_body(store=99, item=1))
    assert response.status_code == 400
    assert "unknown store/item" in response.get_json()["error"]


def test_horizon_too_large(client):
    response = client.post("/predict", json=valid_body(horizon_days=29))
    assert response.status_code == 400
    assert "between 1 and 28" in response.get_json()["error"]


# ---------------- other error responses ----------------

def test_unknown_url_returns_json_404(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.get_json() == {"error": "Not Found"}


def test_wrong_method_returns_405(client):
    assert client.get("/predict").status_code == 405


def test_internal_error_hides_details(artifacts):
    class BrokenModel:
        def predict(self, features):
            raise RuntimeError("secret internal detail")

    broken = dict(artifacts, model=BrokenModel())
    response = create_app(broken).test_client().post("/predict", json=valid_body())

    assert response.status_code == 500
    assert response.get_json() == {"error": "internal server error"}
    assert "Traceback" not in response.get_data(as_text=True)
    assert "secret internal detail" not in response.get_data(as_text=True)
