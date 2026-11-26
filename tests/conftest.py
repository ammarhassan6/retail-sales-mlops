"""Shared test fixtures. Tests use small FAKE data so they run in seconds
(and in GitHub Actions) without the real Kaggle dataset."""

import pandas as pd
import pytest

from src.data.sample_data import generate_sample_sales


@pytest.fixture
def sample_sales() -> pd.DataFrame:
    df = generate_sample_sales(n_stores=2, n_items=3, days=400)
    df["date"] = pd.to_datetime(df["date"])
    return df
