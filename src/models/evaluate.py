"""STEP 5 - Evaluate the best model on the TEST set + QUALITY GATE.

Quality gate (configs/pipeline_config.yaml -> quality_gate):
  * test SMAPE must be <= max_smape
  * the model must beat a naive baseline (prediction = last 7-day average)
If the gate FAILS the script exits with code 1 -> the pipeline stops and the
model is NOT registered / deployed.
If it PASSES the model is registered in the MLflow Model Registry with the
alias "candidate" (promotion to production is a separate, controlled step).

Run:  python -m src.models.evaluate
"""

import json
import sys

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src.features.preprocessing import FEATURE_COLUMNS, TARGET
from src.models.metrics import regression_metrics
from src.models.mlflow_utils import setup_mlflow
from src.utils.config import load_config, project_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def check_quality_gate(model_metrics: dict, baseline_metrics: dict, gate_cfg: dict) -> tuple[bool, list[str]]:
    reasons = []
    if model_metrics["smape"] > gate_cfg["max_smape"]:
        reasons.append(f"SMAPE {model_metrics['smape']:.2f} > limit {gate_cfg['max_smape']}")
    if gate_cfg["must_beat_baseline"] and model_metrics["smape"] >= baseline_metrics["smape"]:
        reasons.append("model does not beat the naive baseline")
    return len(reasons) == 0, reasons


def plot_predictions(test: pd.DataFrame, preds, out_path) -> None:
    """Actual vs predicted sales for one store-item (easy to show in the demo)."""
    plt.switch_backend("Agg")  # draw plots without a screen (servers / CI)
    sample = test.assign(pred=preds)
    first = sample[(sample["store"] == sample["store"].min()) & (sample["item"] == sample["item"].min())]
    plt.figure(figsize=(10, 4))
    plt.plot(first["date"], first[TARGET], label="actual")
    plt.plot(first["date"], first["pred"], label="predicted")
    plt.title(f"Store {first['store'].iloc[0]} / Item {first['item'].iloc[0]} - test period")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def register_candidate(cfg: dict, run_id: str) -> str:
    name = cfg["mlflow"]["registered_model_name"]
    version = mlflow.register_model(f"runs:/{run_id}/model", name).version
    mlflow.MlflowClient().set_registered_model_alias(name, "candidate", version)
    return version


def main() -> None:
    cfg = load_config()
    model = joblib.load(project_path(cfg["model"]["model_path"]))
    metadata = json.loads(project_path(cfg["model"]["metadata_path"]).read_text())
    test = pd.read_parquet(project_path(cfg["data"]["processed_dir"]) / "test.parquet")

    preds = model.predict(test[FEATURE_COLUMNS]).clip(min=0)
    model_metrics = regression_metrics(test[TARGET], preds)
    baseline_metrics = regression_metrics(test[TARGET], test["sales_mean_7d"])

    passed, reasons = check_quality_gate(model_metrics, baseline_metrics, cfg["quality_gate"])

    plot_path = project_path(cfg["reports"]["plot_file"])
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_predictions(test, preds, plot_path)

    result = {"model_type": metadata["model_type"], "test": model_metrics,
              "baseline_test": baseline_metrics, "quality_gate_passed": passed,
              "failure_reasons": reasons}

    # Attach test results to the SAME MLflow run that trained the model
    setup_mlflow(cfg)
    with mlflow.start_run(run_id=metadata["mlflow_run_id"]):
        mlflow.log_metrics({f"test_{k}": v for k, v in model_metrics.items()})
        mlflow.log_metrics({f"baseline_test_{k}": v for k, v in baseline_metrics.items()})
        mlflow.set_tag("quality_gate", "passed" if passed else "failed")
        mlflow.log_artifact(str(plot_path))

    if passed and cfg["mlflow"]["register_model"]:
        result["registered_version"] = register_candidate(cfg, metadata["mlflow_run_id"])

    project_path(cfg["reports"]["metrics_file"]).write_text(json.dumps(result, indent=2))
    logger.info("Test metrics: %s | baseline: %s", model_metrics, baseline_metrics)

    if not passed:
        logger.error("QUALITY GATE FAILED: %s", reasons)
        sys.exit(1)
    logger.info("Quality gate PASSED")


if __name__ == "__main__":
    main()
