# Airflow image + a SEPARATE Python environment for our ML project.
#
# Why two environments? Airflow 2.x needs SQLAlchemy < 2.0, our MLflow needs 2.0.40.
# Installing both into one environment would break one of them, so the project's
# libraries live in /opt/project-venv and Airflow tasks call /opt/project-venv/bin/...
FROM apache/airflow:2.10.5-python3.12

USER root
# DVC and our versioning code need git (not included in the Airflow image).
# The project folder is mounted from Windows (different owner), so git must be told to trust it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

USER airflow
COPY requirements.txt requirements-api.txt /tmp/req/
RUN python -m venv /opt/airflow/project-venv \
    && /opt/airflow/project-venv/bin/pip install --no-cache-dir -r /tmp/req/requirements.txt "dvc[s3]==3.59.2"
