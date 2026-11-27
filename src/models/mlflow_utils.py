"""Connect to MLflow.

Locally we use a SQLite file (mlflow.db). For DagsHub we put these values in
a .env file (git-ignored, never committed) - see .env.example:
  MLFLOW_TRACKING_URI      = https://dagshub.com/<user>/<repo>.mlflow
  MLFLOW_TRACKING_USERNAME = <dagshub username>
  MLFLOW_TRACKING_PASSWORD = <dagshub token>
"""

import os

import mlflow
from dotenv import load_dotenv

from src.utils.config import project_path


def setup_mlflow(cfg: dict) -> str:
    load_dotenv(project_path(".env"))  # reads .env if it exists; does nothing otherwise

    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        local = cfg["mlflow"]["local_tracking_uri"]          # sqlite:///mlflow.db
        uri = local.replace("sqlite:///", f"sqlite:///{project_path('').as_posix()}/")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    return uri
