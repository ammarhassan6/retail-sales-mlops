"""Flask REST API: daily sales forecast + inventory (reorder) recommendation.

Endpoints:
  GET  /health      -> is the API up and is the model loaded? (used later by Docker/Kubernetes probes)
  GET  /model-info  -> which model is served: type, registry version, metrics, data + code version
  POST /predict     -> forecast for one store-item for the next N days + reorder recommendation

The API only READS files made by the ML pipeline (paths come from configs/*.yaml):
  models/model.joblib, models/metadata.json, reports/metrics.json,
  data/processed/latest_store_item_stats.parquet
Features are built with the SAME function as training (build_inference_frame),
so there is no training-serving skew.

Run:  python -m src.api.app        -> http://127.0.0.1:8000
"""

import json

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from src.api.validation import validate_predict_request
from src.features.preprocessing import FEATURE_COLUMNS, build_inference_frame
from src.inventory.optimizer import recommend_order
from src.utils.config import load_config, project_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_artifacts() -> dict | None:
    """Load the model and the files it needs.

    Returns None when a file is missing: the API still starts, and /health
    answers 503 "not ready" instead of the whole process crashing.
    """
    pipeline_cfg = load_config()
    api_cfg = load_config("api_config.yaml")
    model_path = project_path(pipeline_cfg["model"]["model_path"])
    metadata_path = project_path(pipeline_cfg["model"]["metadata_path"])
    metrics_path = project_path(pipeline_cfg["reports"]["metrics_file"])
    stats_path = project_path(api_cfg["stats_path"])

    for path in [model_path, metadata_path, metrics_path, stats_path]:
        if not path.exists():
            logger.error("Missing file %s - run `dvc pull` (or `dvc repro`) first", path)
            return None

    logger.info("Loading model from %s", model_path)
    return {
        "model": joblib.load(model_path),
        "stats": pd.read_parquet(stats_path),
        "metadata": json.loads(metadata_path.read_text()),
        "metrics": json.loads(metrics_path.read_text()),
    }


def error_response(message: str, status_code: int):
    # Every error has the same simple JSON shape - never a stack trace
    return jsonify({"error": message}), status_code


def model_version(artifacts: dict) -> str:
    # The MLflow Model Registry version written by src/models/evaluate.py
    return str(artifacts["metrics"].get("registered_version", "unknown"))


def is_known_store_item(stats: pd.DataFrame, store: int, item: int) -> bool:
    match = (stats["store"] == store) & (stats["item"] == item)
    return bool(match.any())


def build_model_info(artifacts: dict) -> dict:
    metadata = artifacts["metadata"]
    metrics = artifacts["metrics"]
    stats = artifacts["stats"]
    return {
        "model_type": metadata.get("model_type"),
        "model_version": model_version(artifacts),
        "params": metadata.get("params"),
        "val_metrics": metadata.get("val_metrics"),
        "test_metrics": metrics.get("test"),
        "baseline_test_metrics": metrics.get("baseline_test"),
        "quality_gate_passed": metrics.get("quality_gate_passed"),
        "feature_columns": metadata.get("feature_columns"),
        "trained_at": metadata.get("trained_at"),
        "git_commit": metadata.get("git_commit"),
        "data_version_md5": metadata.get("data_version_md5"),
        "mlflow_run_id": metadata.get("mlflow_run_id"),
        "history_as_of_date": str(pd.Timestamp(stats["as_of_date"].max()).date()),
        "store_item_count": len(stats),
    }


def make_prediction(params: dict, artifacts: dict, service_level: float) -> dict:
    """Forecast every day of the horizon, then turn it into a reorder decision."""
    dates = pd.date_range(params["start_date"], periods=params["horizon_days"], freq="D")
    requests_df = pd.DataFrame({"store": params["store"], "item": params["item"], "date": dates})

    # Same feature code as training -> same columns, same order
    features = build_inference_frame(requests_df, artifacts["stats"])[FEATURE_COLUMNS]
    # Sales can never be negative, so clip small negative model outputs to 0
    predictions = artifacts["model"].predict(features).clip(min=0)

    daily_std = float(features["sales_std_28d"].iloc[0])
    inventory = recommend_order(
        daily_forecasts=predictions.tolist(),
        daily_std=daily_std,
        current_stock=params["current_stock"],
        lead_time_days=params["lead_time_days"],
        service_level=service_level,
    )

    forecast = []
    for day, value in zip(dates, predictions):
        forecast.append({"date": day.strftime("%Y-%m-%d"), "predicted_sales": round(float(value), 2)})

    return {
        "model_version": model_version(artifacts),
        "store": params["store"],
        "item": params["item"],
        "forecast": forecast,
        "inventory": inventory,
    }


def create_app(artifacts: dict | None) -> Flask:
    """Build the Flask app. Tests pass small fake artifacts; main() passes the real ones."""
    app = Flask(__name__)
    api_cfg = load_config("api_config.yaml")
    inventory_cfg = load_config()["inventory"]
    # Bodies larger than this get "413 Request Entity Too Large"
    app.config["MAX_CONTENT_LENGTH"] = api_cfg["max_request_bytes"]

    @app.get("/health")
    def health():
        if artifacts is None:
            return jsonify({"status": "unavailable", "model_loaded": False}), 503
        return jsonify({"status": "ok", "model_loaded": True}), 200

    @app.get("/model-info")
    def model_info():
        if artifacts is None:
            return error_response("model is not loaded", 503)
        return jsonify(build_model_info(artifacts)), 200

    @app.post("/predict")
    def predict():
        if artifacts is None:
            return error_response("model is not loaded", 503)

        # silent=True -> returns None for broken JSON instead of raising an error
        body = request.get_json(silent=True)
        if body is None:
            return error_response("request body must be valid JSON (Content-Type: application/json)", 400)

        params, error = validate_predict_request(
            body,
            default_horizon_days=api_cfg["default_horizon_days"],
            max_horizon_days=api_cfg["max_horizon_days"],
            default_lead_time_days=inventory_cfg["default_lead_time_days"],
        )
        if error:
            return error_response(error, 400)
        if not is_known_store_item(artifacts["stats"], params["store"], params["item"]):
            return error_response(f"unknown store/item combination: {params['store']}/{params['item']}", 400)

        result = make_prediction(params, artifacts, inventory_cfg["service_level"])
        logger.info("Prediction store=%s item=%s days=%s reorder=%s", params["store"], params["item"],
                    params["horizon_days"], result["inventory"]["should_reorder"])
        return jsonify(result), 200

    @app.errorhandler(HTTPException)
    def handle_http_error(err):
        # Flask's own errors: 404 unknown URL, 405 wrong method, 413 body too large, ...
        return error_response(err.name, err.code)

    @app.errorhandler(Exception)
    def handle_unexpected_error(err):
        # Full details go to the server LOG only; the client gets a generic message
        logger.exception("Unexpected error on %s", request.path)
        return error_response("internal server error", 500)

    return app


def main() -> None:
    api_cfg = load_config("api_config.yaml")
    app = create_app(load_artifacts())
    # debug=False: never show the interactive debugger or stack traces to clients
    app.run(host=api_cfg["host"], port=api_cfg["port"], debug=False)


if __name__ == "__main__":
    main()
