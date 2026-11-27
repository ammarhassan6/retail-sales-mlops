"""STEP 4 - Train + tune candidate models and track everything in MLflow.

For every model type and every hyperparameter combination:
  1. train on the TRAIN set
  2. score on the VALIDATION set
  3. log params, metrics, duration, data version and git commit to MLflow
The best run (lowest validation SMAPE) is saved to models/model.joblib.

Run:  python -m src.models.train
"""

import json
import time
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from src.features.preprocessing import FEATURE_COLUMNS, TARGET
from src.models.metrics import regression_metrics
from src.models.mlflow_utils import setup_mlflow
from src.models.tune import build_model, param_combinations
from src.utils.config import load_config, project_path
from src.utils.logger import get_logger
from src.utils.versioning import file_md5, git_commit

logger = get_logger(__name__)


def main() -> None:
    cfg = load_config()
    model_grid = load_config("model_params.yaml")["models"]
    seed = cfg["project"]["random_seed"]

    processed = project_path(cfg["data"]["processed_dir"])
    train = pd.read_parquet(processed / "train.parquet")
    val = pd.read_parquet(processed / "val.parquet")
    X_train, y_train = train[FEATURE_COLUMNS], train[TARGET]
    X_val, y_val = val[FEATURE_COLUMNS], val[TARGET]

    # Versions recorded with every run -> any model can be traced back to its code + data
    data_version = file_md5(project_path(cfg["data"]["raw_path"]))
    code_version = git_commit()

    setup_mlflow(cfg)
    best = None

    for model_type, grid in model_grid.items():
        for params in param_combinations(grid):
            with mlflow.start_run(run_name=model_type) as run:
                model = build_model(model_type, params, seed)

                start = time.time()
                model.fit(X_train, y_train)
                duration = time.time() - start

                preds = model.predict(X_val).clip(min=0)   # sales can't be negative
                metrics = regression_metrics(y_val, preds)

                mlflow.set_tags({"model_type": model_type, "git_commit": code_version,
                                 "data_version_md5": data_version})
                mlflow.log_params({"model_type": model_type, **params,
                                   "n_train_rows": len(train), "features": ",".join(FEATURE_COLUMNS)})
                mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
                mlflow.log_metric("training_duration_sec", duration)

                logger.info("%s %s -> val SMAPE %.2f (%.1fs)", model_type, params, metrics["smape"], duration)

                if best is None or metrics["smape"] < best["val_metrics"]["smape"]:
                    best = {"model": model, "model_type": model_type, "params": params,
                            "val_metrics": metrics, "run_id": run.info.run_id}

    # Save the model artifact only for the best run (keeps storage small)
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.sklearn.log_model(best["model"], artifact_path="model")
        mlflow.set_tag("selected_as_best", "true")

    model_path = project_path(cfg["model"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], model_path)

    metadata = {
        "model_type": best["model_type"],
        "params": best["params"],
        "val_metrics": best["val_metrics"],
        "mlflow_run_id": best["run_id"],
        "feature_columns": FEATURE_COLUMNS,
        "data_version_md5": data_version,
        "git_commit": code_version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    project_path(cfg["model"]["metadata_path"]).write_text(json.dumps(metadata, indent=2), newline="\n")
    logger.info("Best model: %s (val SMAPE %.2f) saved to %s",
                best["model_type"], best["val_metrics"]["smape"], model_path)


if __name__ == "__main__":
    main()
