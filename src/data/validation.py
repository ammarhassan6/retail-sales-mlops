"""STEP 2 - Data validation (schema + quality checks) using Pandera.

If ANY check fails, the script exits with an error, so the pipeline STOPS
and bad data never reaches model training.

Run:  python -m src.data.validation
"""

import json
import sys

import pandas as pd
import pandera.pandas as pa

from src.utils.config import load_config, project_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# The "contract" our data must follow
SALES_SCHEMA = pa.DataFrameSchema(
    columns={
        "date": pa.Column("datetime64[ns]", nullable=False),
        "store": pa.Column(int, checks=pa.Check.ge(1), nullable=False),
        "item": pa.Column(int, checks=pa.Check.ge(1), nullable=False),
        "sales": pa.Column(int, checks=pa.Check.ge(0), nullable=False),  # no negative sales
    },
    unique=["date", "store", "item"],  # one row per store-item-day
    strict=True,                        # no unexpected extra columns
)


def validate_sales(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Return (passed, list_of_error_messages)."""
    try:
        SALES_SCHEMA.validate(df, lazy=True)  # lazy=True -> collect ALL errors, not just the first
        return True, []
    except pa.errors.SchemaErrors as err:
        cases = err.failure_cases[["column", "check"]].drop_duplicates()
        messages = [f"{row.column}: {row.check}" for row in cases.itertuples()]
        return False, messages


def main() -> None:
    cfg = load_config()
    df = pd.read_parquet(project_path(cfg["data"]["interim_path"]))

    passed, errors = validate_sales(df)
    report = {
        "passed": passed,
        "n_rows": int(len(df)),
        "n_stores": int(df["store"].nunique()),
        "n_items": int(df["item"].nunique()),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "missing_values": int(df.isna().sum().sum()),
        "errors": errors,
    }

    report_path = project_path(cfg["reports"]["validation_report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))

    if not passed:
        logger.error("Data validation FAILED: %s", errors)
        sys.exit(1)  # non-zero exit code stops DVC / Airflow / CI

    out_path = project_path(cfg["data"]["validated_path"])
    df.to_parquet(out_path, index=False)
    logger.info("Data validation passed (%d rows) -> %s", len(df), out_path)


if __name__ == "__main__":
    main()
