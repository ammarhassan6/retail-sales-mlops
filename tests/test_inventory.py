"""UNIT TESTS for the inventory recommendation logic."""

import pytest

from src.inventory.optimizer import recommend_order, z_score


def test_z_score_95_percent():
    assert z_score(0.95) == pytest.approx(1.645, abs=0.001)


def test_order_needed_when_stock_is_low():
    result = recommend_order([10] * 7, daily_std=2, current_stock=20, lead_time_days=7)
    assert result["lead_time_demand"] == 70
    assert result["should_reorder"] is True
    assert result["recommended_order_quantity"] > 50


def test_no_order_when_stock_is_high():
    result = recommend_order([10] * 7, daily_std=2, current_stock=500, lead_time_days=7)
    assert result["should_reorder"] is False
    assert result["recommended_order_quantity"] == 0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        recommend_order([10] * 7, daily_std=2, current_stock=-1, lead_time_days=7)
    with pytest.raises(ValueError):
        recommend_order([10] * 3, daily_std=2, current_stock=10, lead_time_days=7)
