"""DATA TESTS - the validation step must accept good data and reject bad data."""

import pandas as pd

from src.data.validation import validate_sales


def test_valid_data_passes(sample_sales):
    passed, errors = validate_sales(sample_sales)
    assert passed, errors


def test_negative_sales_fail(sample_sales):
    sample_sales.loc[0, "sales"] = -5
    passed, errors = validate_sales(sample_sales)
    assert not passed
    assert any("sales" in e for e in errors)


def test_missing_values_fail(sample_sales):
    sample_sales["sales"] = sample_sales["sales"].astype(float)
    sample_sales.loc[3, "sales"] = None
    passed, _ = validate_sales(sample_sales)
    assert not passed


def test_duplicate_rows_fail(sample_sales):
    duplicated = pd.concat([sample_sales, sample_sales.head(1)], ignore_index=True)
    passed, _ = validate_sales(duplicated)
    assert not passed


def test_extra_column_fails(sample_sales):
    sample_sales["unexpected"] = 1
    passed, _ = validate_sales(sample_sales)
    assert not passed


def test_wrong_type_fails(sample_sales):
    sample_sales["store"] = "store_A"
    passed, _ = validate_sales(sample_sales)
    assert not passed
