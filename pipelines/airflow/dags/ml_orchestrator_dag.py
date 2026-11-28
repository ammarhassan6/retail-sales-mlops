"""Airflow DAG: retail sales forecasting pipeline.

    fetch_raw_data -> ingest -> validate -> preprocess -> feature_store -> train -> evaluate -> report

* Every task runs ONE stage of our DVC pipeline (`dvc repro -s <stage>`):
  Airflow = WHEN and in WHICH ORDER things run (schedule, retries, logs, UI),
  DVC     = WHAT changed and which versions were used (dvc.lock).
  If the inputs of a stage did not change, DVC skips it -> re-running the DAG is cheap.
* fetch_raw_data downloads the raw dataset from DagsHub storage (`dvc pull`).
* validate / evaluate exit with code 1 on bad data / a model that fails the quality
  gate -> the task fails -> all later tasks are skipped (nothing gets registered).
* Trigger "with config" {"force": true} to re-run every stage even without changes.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

PROJECT_DIR = os.environ.get("PROJECT_DIR", "/opt/project")
BIN = os.environ.get("PROJECT_PYTHON_BIN", "/opt/airflow/project-venv/bin")  # the project's own venv

default_args = {
    "owner": "person-a",
    "retries": 2,                          # e.g. a network hiccup while talking to DagsHub
    "retry_delay": timedelta(minutes=1),
}


# DVC runs each stage as `python -m ...`. Putting the project venv FIRST on the PATH makes that
# `python` our project's Python (with pandera, feast, mlflow), not Airflow's own Python.
IN_PROJECT = f"export PATH={BIN}:$PATH && cd {PROJECT_DIR}"


def dvc_stage(stage: str) -> str:
    """Bash command that runs one DVC stage. '--force' is added when the run asks for it."""
    force = "{{ '--force' if params.force else '' }}"
    return f"{IN_PROJECT} && dvc repro --single-item {stage} {force}"


def report_results() -> None:
    """Print the evaluation result into the Airflow log (visible in the UI)."""
    metrics = json.loads(Path(PROJECT_DIR, "reports", "metrics.json").read_text())
    print(f"Model: {metrics['model_type']}")
    print(f"Test SMAPE: {metrics['test']['smape']:.2f}%  (baseline {metrics['baseline_test']['smape']:.2f}%)")
    print(f"Quality gate passed: {metrics['quality_gate_passed']}")
    print(f"Registered model version: {metrics.get('registered_version', 'not registered')}")


with DAG(
    dag_id="retail_sales_pipeline",
    description="Fetch, validate, build features, train and evaluate the sales forecasting model",
    schedule="@weekly",                    # new sales data is assumed to arrive weekly
    start_date=datetime(2026, 11, 1),
    catchup=False,                         # don't run all the missed weeks since start_date
    max_active_runs=1,                     # never two pipeline runs writing the same files
    default_args=default_args,
    params={"force": Param(False, type="boolean", description="Re-run every stage even if nothing changed")},
    tags=["retail", "mlops"],
) as dag:

    fetch_raw_data = BashOperator(
        task_id="fetch_raw_data",
        bash_command=f"{IN_PROJECT} && dvc pull data/raw/train.csv.dvc",
    )

    stages = ["ingest", "validate", "preprocess", "feature_store", "train", "evaluate"]
    # Bad data or a model below the quality gate won't get better by retrying -> fail at once
    no_retry = {"validate", "evaluate"}
    stage_tasks = [
        BashOperator(task_id=stage, bash_command=dvc_stage(stage),
                     retries=0 if stage in no_retry else 2,
                     execution_timeout=timedelta(minutes=30))
        for stage in stages
    ]

    report = PythonOperator(task_id="report", python_callable=report_results, retries=0)

    # fetch_raw_data >> ingest >> validate >> ... >> evaluate >> report
    previous = fetch_raw_data
    for task in stage_tasks:
        previous >> task
        previous = task
    previous >> report
