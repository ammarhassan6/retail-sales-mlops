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


@pytest.fixture
def temp_store(tmp_path, sample_sales):
    """A REAL Feast store in a temporary folder, filled with the sample data:
    apply definitions -> materialize the online store. Used by the feature store
    tests and the API tests (Feast path)."""
    from feast import FeatureStore
    from feast.repo_config import RepoConfig

    from feature_store.client import load_definitions
    from src.features.preprocessing import build_feature_store_table

    feature_table = build_feature_store_table(sample_sales)
    parquet = tmp_path / "features.parquet"
    feature_table.to_parquet(parquet, index=False)
    config = RepoConfig(
        project="test_retail",
        registry=str(tmp_path / "registry.db"),
        provider="local",
        online_store={"type": "sqlite", "path": str(tmp_path / "online.db")},
        offline_store={"type": "file"},
        entity_key_serialization_version=3,
        repo_path=str(tmp_path),
    )
    store = FeatureStore(config=config)
    entities, feature_views = load_definitions()
    store.apply([entities.store, entities.item, feature_views.make_feature_view(parquet)])
    newest = feature_table["event_timestamp"].max()
    store.materialize(start_date=newest - pd.Timedelta(days=7), end_date=newest + pd.Timedelta(seconds=1))
    return store
