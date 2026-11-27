"""Connect to MLflow.

Locally we use a SQLite file (mlflow.db). On DagsHub we set these
environment variables instead (never hard-code them!):
  MLFLOW_TRACKING_URI      = https://dagshub.com/<user>/<repo>.mlflow
  MLFLOW_TRACKING_USERNAME = <dagshub username>
  MLFLOW_TRACKING_PASSWORD = <dagshub token>
"""

import os

import mlflow

from src.utils.config import project_path


def setup_mlflow(cfg: dict) -> str:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        local = cfg["mlflow"]["local_tracking_uri"]          # sqlite:///mlflow.db
        uri = local.replace("sqlite:///", f"sqlite:///{project_path('').as_posix()}/")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    return uri
