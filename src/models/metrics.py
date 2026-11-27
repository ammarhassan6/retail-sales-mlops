"""Regression metrics used everywhere (training, evaluation, tests, monitoring).

MAE   = average absolute error in units sold           (easy to explain)
RMSE  = like MAE but punishes big mistakes more
SMAPE = percentage error between 0 and 200%, works even when sales = 0
        (it was the official metric of the Kaggle competition)
"""

import numpy as np


def smape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    denominator = np.abs(y_true) + np.abs(y_pred)
    safe_denominator = np.where(denominator == 0, 1, denominator)    # avoid dividing by 0
    ratio = 2 * np.abs(y_pred - y_true) / safe_denominator
    ratio = np.where(denominator == 0, 0.0, ratio)                   # actual=0 and pred=0 -> perfect
    return float(100 * ratio.mean())


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "smape": smape(y_true, y_pred),
    }
