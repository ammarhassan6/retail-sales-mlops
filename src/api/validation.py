"""Input validation for POST /predict.

Every field is checked BEFORE the model is called, so bad input gets a clear
400 error message instead of crashing the API (500). We use plain Python
checks: no extra library is needed and each rule is easy to read.

Used by src/api/app.py - there is nothing to run directly.
"""

import math
from datetime import date

REQUIRED_FIELDS = ["store", "item", "start_date", "current_stock"]


def is_int(value) -> bool:
    # In Python True/False are also ints (True == 1), so we reject bools explicitly
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    # Python's JSON parser accepts NaN and Infinity, which would break the maths
    return math.isfinite(value)


def parse_date(value) -> date | None:
    """Return a date for a 'YYYY-MM-DD' string, or None if it is not a valid date."""
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def validate_predict_request(body, default_horizon_days: int, max_horizon_days: int,
                             default_lead_time_days: int) -> tuple[dict | None, str | None]:
    """Return (params, None) when the request is valid, or (None, error_message) when not."""
    if not isinstance(body, dict):
        return None, "request body must be a JSON object"

    for field in REQUIRED_FIELDS:
        if field not in body:
            return None, f"missing required field: {field}"

    store = body["store"]
    item = body["item"]
    start_date = parse_date(body["start_date"])
    current_stock = body["current_stock"]
    horizon_days = body.get("horizon_days", default_horizon_days)
    lead_time_days = body.get("lead_time_days", default_lead_time_days)

    if not is_int(store) or store < 1:
        return None, "store must be a positive integer"
    if not is_int(item) or item < 1:
        return None, "item must be a positive integer"
    if start_date is None:
        return None, "start_date must be a date in YYYY-MM-DD format"
    if not is_number(current_stock) or current_stock < 0:
        return None, "current_stock must be a number >= 0"
    if not is_int(horizon_days) or not 1 <= horizon_days <= max_horizon_days:
        return None, f"horizon_days must be an integer between 1 and {max_horizon_days}"
    if not is_int(lead_time_days) or lead_time_days < 1:
        return None, "lead_time_days must be a positive integer"
    if lead_time_days > horizon_days:
        # the reorder point needs a forecast for every day of the lead time
        return None, "lead_time_days cannot be larger than horizon_days"

    params = {
        "store": store,
        "item": item,
        "start_date": start_date,
        "horizon_days": horizon_days,
        "lead_time_days": lead_time_days,
        "current_stock": current_stock,
    }
    return params, None
