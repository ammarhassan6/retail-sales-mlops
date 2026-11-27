"""Hyperparameter tuning helpers (grid search over configs/model_params.yaml).

Example: {"max_iter": [200, 400], "learning_rate": [0.05, 0.1]}
  -> 4 combinations, each one becomes a separate MLflow run.
"""

from itertools import product

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def param_combinations(grid: dict) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in product(*grid.values())]


def build_model(model_type: str, params: dict, seed: int = 42):
    if model_type == "ridge":
        # linear models need scaled inputs
        return make_pipeline(StandardScaler(), Ridge(**params))
    if model_type == "random_forest":
        return RandomForestRegressor(random_state=seed, **params)
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=seed, **params)
    raise ValueError(f"Unknown model type: {model_type}")
