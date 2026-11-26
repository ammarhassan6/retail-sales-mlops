"""MODEL TESTS + REGRESSION TESTS."""

import json

import numpy as np
import pytest

from src.features.preprocessing import FEATURE_COLUMNS, TARGET, build_features, time_split
from src.models.evaluate import check_quality_gate
from src.models.metrics import regression_metrics, smape
from src.models.tune import build_model, param_combinations
from src.utils.config import load_config, project_path


@pytest.fixture
def trained(sample_sales):
    train, _, test = time_split(build_features(sample_sales), val_days=30, test_days=30)
    model = build_model("hist_gradient_boosting", {"max_iter": 100})
    model.fit(train[FEATURE_COLUMNS], train[TARGET])
    return model, test


def test_smape_known_values():
    assert smape([100, 0], [100, 0]) == 0.0
    assert smape([100], [50]) == pytest.approx(200 * 50 / 150)


def test_param_combinations():
    combos = param_combinations({"a": [1, 2], "b": [3]})
    assert combos == [{"a": 1, "b": 3}, {"a": 2, "b": 3}]


def test_prediction_shape_and_values(trained):
    model, test = trained
    preds = model.predict(test[FEATURE_COLUMNS])
    assert preds.shape == (len(test),)
    assert np.isfinite(preds).all()


def test_model_beats_naive_baseline(trained):
    model, test = trained
    preds = model.predict(test[FEATURE_COLUMNS]).clip(min=0)
    model_smape = regression_metrics(test[TARGET], preds)["smape"]
    baseline_smape = regression_metrics(test[TARGET], test["sales_mean_7d"])["smape"]
    assert model_smape < baseline_smape


def test_quality_gate_blocks_bad_model():
    gate = {"max_smape": 20.0, "must_beat_baseline": True}
    passed, reasons = check_quality_gate({"smape": 35.0}, {"smape": 30.0}, gate)
    assert not passed
    assert len(reasons) == 2


def test_quality_gate_accepts_good_model():
    gate = {"max_smape": 20.0, "must_beat_baseline": True}
    passed, _ = check_quality_gate({"smape": 12.0}, {"smape": 18.0}, gate)
    assert passed


def test_no_regression_against_last_report():
    """REGRESSION TEST: if a real evaluation report exists, its quality must
    still be within the allowed limit (detects model quality deterioration)."""
    metrics_file = project_path(load_config()["reports"]["metrics_file"])
    if not metrics_file.exists():
        pytest.skip("no metrics report yet - run the pipeline first")
    report = json.loads(metrics_file.read_text())
    assert report["test"]["smape"] <= load_config()["quality_gate"]["max_smape"]
