# Retail Sales Forecasting & Inventory Optimization — MLOps Project

Forecast daily sales for every **store × item** and turn the forecast into an **inventory decision**
(reorder point, safety stock, order quantity), wrapped in a reproducible, automated, tested,
deployed and monitored ML system.

**Dataset:** [Store Item Demand Forecasting Challenge (Kaggle)](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data)
— 5 years (2013–2017) of daily sales, 10 stores, 50 items (`date, store, item, sales`).
Used under the Kaggle competition rules for educational, non-commercial purposes.

## Pipeline

```
data/raw/train.csv
   │  1. ingestion      src/data/ingestion.py         (local or S3)
   ▼
data/interim/sales.parquet
   │  2. validation     src/data/validation.py        (Pandera schema + quality checks, fails the pipeline on bad data)
   ▼
data/interim/validated.parquet
   │  3. preprocessing  src/features/preprocessing.py (calendar + rolling-history features, time-based split)
   ▼
data/processed/{train,val,test}.parquet + latest_store_item_stats.parquet + store_item_features.parquet
   │  3b. feature store feature_store/materialize.py   (Feast: register features, fill online store, verify)
   │  4. training       src/models/train.py           (grid search over 3 model types, every run tracked in MLflow)
   ▼
models/model.joblib + models/metadata.json
   │  5. evaluation     src/models/evaluate.py        (test metrics + QUALITY GATE → register as "candidate")
   ▼
reports/metrics.json, reports/pred_vs_actual.png
```

Inventory logic: `src/inventory/optimizer.py`
(`reorder_point = forecast demand during lead time + z × σ × √lead_time`).

## Quick start (Windows PowerShell, Python 3.12)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

# put Kaggle train.csv in data/raw/   (or create fake data: python -m src.data.sample_data)
dvc repro                       # runs all 5 stages (or run each: python -m src.data.ingestion ...)
python -m pytest -v             # unit, data, model, integration, regression tests
ruff check .                    # code quality
mlflow ui --backend-store-uri sqlite:///mlflow.db   # open http://127.0.0.1:5000
```

## Configuration
- `configs/pipeline_config.yaml` — paths, split sizes, MLflow names, quality gate, inventory settings
- `configs/model_params.yaml` — candidate models and hyperparameter grids

## Environment variables (names only — never commit values)
| Name | Used for |
|---|---|
| `MLFLOW_TRACKING_URI` | remote MLflow server (DagsHub); local SQLite is used if not set |
| `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` | DagsHub credentials |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` | S3 access (or an IAM role) |

## Repository structure
```
configs/        YAML configuration
src/data/       ingestion, validation, sample data generator
src/features/   feature engineering (shared by training and serving)
src/models/     training, tuning, evaluation, metrics, MLflow helpers
src/inventory/  reorder-point / safety-stock optimizer
feature_store/  Feast feature repo (entities, feature views) - see feature_store/README.md
tests/          pytest suite (runs on small generated data)
dvc.yaml        DVC pipeline definition
```
*(More sections — API, Docker, CI/CD, AWS, Kubernetes, Airflow, Kubeflow, Feast, monitoring — are added as the project grows.)*
