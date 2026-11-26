"""Generate a small FAKE sales dataset with the same columns as the Kaggle data.

Why? Unit tests and CI (GitHub Actions) must run without downloading the
real 900k-row dataset. The fake data has weekly + yearly seasonality so a
model can actually learn something from it.

Run:  python -m src.data.sample_data      -> writes data/raw/sample_train.csv
"""

import numpy as np
import pandas as pd

from src.utils.config import project_path


def generate_sample_sales(n_stores: int = 2, n_items: int = 3, days: int = 400,
                          start: str = "2016-01-01", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    day_index = np.arange(days)

    weekly = 1 + 0.3 * (dates.dayofweek >= 5)                      # weekend boost
    yearly = 1 + 0.2 * np.sin(2 * np.pi * dates.dayofyear / 365.25)  # seasons
    trend = 1 + 0.0005 * day_index                                  # slow growth

    frames = []
    for store in range(1, n_stores + 1):
        for item in range(1, n_items + 1):
            base = rng.uniform(15, 60)
            expected = base * weekly * yearly * trend
            frames.append(pd.DataFrame({
                "date": dates.strftime("%Y-%m-%d"),
                "store": store,
                "item": item,
                "sales": rng.poisson(expected),
            }))
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    out = project_path("data/raw/sample_train.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    generate_sample_sales(n_stores=3, n_items=5, days=730).to_csv(out, index=False)
    print(f"Sample data written to {out}")
