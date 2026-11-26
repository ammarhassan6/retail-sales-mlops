"""STEP 1 - Data ingestion.

Reads the raw sales CSV (from local disk or from AWS S3), fixes data types,
sorts it, and saves it as a Parquet file for the next step.

Run:  python -m src.data.ingestion
"""

import pandas as pd

from src.utils.config import load_config, project_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def download_from_s3(bucket: str, key: str, local_path) -> None:
    """Download the raw file from S3. Credentials come from the environment /
    IAM role - they are NEVER written in the code."""
    import boto3  # imported here so local runs do not need boto3

    local_path.parent.mkdir(parents=True, exist_ok=True)
    boto3.client("s3").download_file(bucket, key, str(local_path))
    logger.info("Downloaded s3://%s/%s", bucket, key)


def load_raw_data(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")  # bad dates -> NaT (caught by validation)
    return df.sort_values(["store", "item", "date"]).reset_index(drop=True)


def main() -> None:
    cfg = load_config()["data"]
    raw_path = project_path(cfg["raw_path"])

    if cfg["source"] == "s3":
        download_from_s3(cfg["s3_bucket"], cfg["s3_key"], raw_path)

    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. Download the Kaggle dataset or run "
            "'python -m src.data.sample_data' and point raw_path to the sample file."
        )

    df = load_raw_data(raw_path)
    out_path = project_path(cfg["interim_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info("Ingested %d rows (%s to %s) -> %s",
                len(df), df["date"].min().date(), df["date"].max().date(), out_path)


if __name__ == "__main__":
    main()
