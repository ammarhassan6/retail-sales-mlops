# Airflow orchestration

DAG `retail_sales_pipeline` ([dags/ml_orchestrator_dag.py](dags/ml_orchestrator_dag.py)):

```
fetch_raw_data → ingest → validate → preprocess → feature_store → train → evaluate → report
```

| Task | What it runs | Retries |
|---|---|---|
| fetch_raw_data | `dvc pull data/raw/train.csv.dvc` (raw data from DagsHub storage) | 2 |
| ingest … train | `dvc repro --single-item <stage>` | 2 |
| validate, evaluate | same, but **no retries**: bad data / a weak model won't improve by retrying | 0 |
| report | prints test SMAPE, baseline, quality gate, registered version | 0 |

- **Schedule:** `@weekly`, `catchup=False`, `max_active_runs=1`. Manual runs: ▶ in the UI.
- **Idempotent:** DVC skips stages whose inputs didn't change. Trigger with config `{"force": true}`
  to re-run everything.
- **Failure handling:** a failed task stops the run; downstream tasks are marked `upstream_failed`,
  so a model that fails validation or the quality gate is never registered.
- **Two Python environments:** Airflow's own + `/opt/airflow/project-venv` for our code
  (Airflow 2.x needs SQLAlchemy < 2.0, MLflow 2.22 needs 2.0.x).

## Run locally
```bash
cd pipelines/airflow
cp .env.example .env            # then set your own admin password + secret key
docker compose up -d --build    # first build takes a few minutes
# open http://localhost:8080 -> unpause "retail_sales_pipeline" -> ▶ Trigger DAG
docker compose down             # stop
```
Needs the project's `.dvc/config.local` (DagsHub token) and `.env` (MLflow on DagsHub) on the host;
they are mounted into the containers, never copied into the image.
