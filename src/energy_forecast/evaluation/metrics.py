import numpy as np
import pandas as pd

# Scalar metrics returned by evaluate(), logged to MLflow and shown in the comparison table
SCALAR_METRICS = ["mae_test", "rmse_test", "mape_test", "nrmse_test"]
# The same metrics on the training data (evaluate(..., split="train"))
TRAIN_METRICS = [m.replace("_test", "_train") for m in SCALAR_METRICS]


def evaluate(y_true, y_pred, index, split: str = "test", min_week_count: int = 300) -> dict:
    """
    Score predictions over a whole split; returns metrics plus timestamped errors.

    Metric keys are suffixed with split (e.g. rmse_test, or rmse_train for split="train").

    index must be a DatetimeIndex aligned with y_true/y_pred. Weekly RMSE is computed per
    calendar week, keeping only weeks with at least min_week_count observations: a full week
    is 336 half-hours, and the default leaves room for clock-change weeks and small gaps
    while dropping the partial weeks at the start/end of the test period.
    """
    y_true = pd.Series(np.asarray(y_true).flatten(), index=index)
    errors = y_true - np.asarray(y_pred).flatten()
    rmse = float(np.sqrt((errors ** 2).mean()))

    weekly = (errors ** 2).resample("W").agg(["mean", "count"])
    weekly = weekly[weekly["count"] >= min_week_count]

    return {
        "errors": errors,
        f"mae_{split}": float(errors.abs().mean()),
        f"rmse_{split}": rmse,
        # Demand never approaches zero, so percentage errors are well defined
        f"mape_{split}": float((errors.abs() / y_true.abs()).mean() * 100),
        f"nrmse_{split}": rmse / float(y_true.mean()),
        "weekly_rmse": np.sqrt(weekly["mean"]),
    }
