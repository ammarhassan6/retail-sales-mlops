"""Inventory optimization: turn a sales FORECAST into an ORDER decision.

Classic "reorder point" formula used in supply-chain management:

  lead_time_demand = sum of forecasted sales during the lead time
  safety_stock     = z * daily_std * sqrt(lead_time_days)
  reorder_point    = lead_time_demand + safety_stock
  order_quantity   = max(0, reorder_point - current_stock)

  z = how many standard deviations of extra stock we keep. For a 95% service
      level z = 1.645 (we will NOT run out of stock on ~95% of cycles).
"""

import math
from statistics import NormalDist


def z_score(service_level: float) -> float:
    if not 0.5 <= service_level < 1:
        raise ValueError("service_level must be between 0.5 and 1 (e.g. 0.95)")
    return NormalDist().inv_cdf(service_level)


def recommend_order(daily_forecasts: list[float], daily_std: float, current_stock: float,
                    lead_time_days: int, service_level: float = 0.95) -> dict:
    if lead_time_days < 1:
        raise ValueError("lead_time_days must be at least 1")
    if current_stock < 0:
        raise ValueError("current_stock cannot be negative")
    if len(daily_forecasts) < lead_time_days:
        raise ValueError("need one forecast per lead-time day")

    lead_time_demand = sum(daily_forecasts[:lead_time_days])
    safety_stock = z_score(service_level) * daily_std * math.sqrt(lead_time_days)
    reorder_point = lead_time_demand + safety_stock
    order_quantity = max(0.0, reorder_point - current_stock)

    return {
        "lead_time_demand": round(lead_time_demand, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "current_stock": current_stock,
        "should_reorder": current_stock < reorder_point,
        "recommended_order_quantity": math.ceil(order_quantity),  # can't order half an item
    }
