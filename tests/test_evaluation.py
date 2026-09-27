import numpy as np
import pandas as pd
import pytest

from energy_forecast.evaluation.metrics import evaluate
from energy_forecast.evaluation.significance import diebold_mariano_test


def half_hourly_index(n, start="2024-01-01"):  # 2024-01-01 is a Monday
    return pd.date_range(start, periods=n, freq="30min")


# ---------- evaluate ----------

def test_evaluate_known_values():
    index = half_hourly_index(336 * 2)
    y_true = np.full(len(index), 100.0)
    y_pred = np.full(len(index), 90.0)

    result = evaluate(y_true, y_pred, index)

    assert result["mae_test"] == pytest.approx(10)
    assert result["rmse_test"] == pytest.approx(10)
    assert result["mape_test"] == pytest.approx(10)
    assert result["nrmse_test"] == pytest.approx(0.1)


def test_evaluate_split_sets_metric_names():
    index = half_hourly_index(10)
    result = evaluate(np.full(10, 100.0), np.full(10, 90.0), index, split="train")

    assert {"mae_train", "rmse_train", "mape_train", "nrmse_train"} <= result.keys()
    assert "rmse_test" not in result


def test_evaluate_errors_are_actual_minus_predicted_on_given_index():
    index = half_hourly_index(5)
    result = evaluate([10, 20, 30, 40, 50], [11, 18, 30, 45, 50], index)

    assert result["errors"].index.equals(index)
    assert result["errors"].tolist() == [-1, 2, 0, -5, 0]


def test_evaluate_weekly_rmse_drops_partial_weeks():
    # Two full Monday-Sunday weeks, then 10 half-hours of a third week
    index = half_hourly_index(336 * 2 + 10)
    errors = np.concatenate([np.full(336, 1.0), np.full(336, 3.0), np.full(10, 100.0)])

    result = evaluate(errors, np.zeros(len(index)), index)

    assert result["weekly_rmse"].tolist() == pytest.approx([1.0, 3.0])


# ---------- diebold_mariano_test ----------

@pytest.fixture
def two_forecasts():
    rng = np.random.default_rng(0)
    errors_a = rng.normal(0, 1.0, 500)
    errors_b = rng.normal(0, 1.5, 500)
    return errors_a, errors_b


def test_dm_without_lags_matches_closed_form(two_forecasts):
    errors_a, errors_b = two_forecasts
    d = errors_a ** 2 - errors_b ** 2
    expected = d.mean() / np.sqrt(d.var(ddof=0) / len(d))

    stat, _ = diebold_mariano_test(errors_a, errors_b, lags=0)

    assert stat == pytest.approx(expected)


def test_dm_bartlett_long_run_variance(two_forecasts):
    errors_a, errors_b = two_forecasts
    d = errors_a ** 2 - errors_b ** 2
    T, c = len(d), d - d.mean()
    gamma = [np.sum(c[k:] * c[: T - k]) / T for k in range(3)]
    # lags=2: Bartlett weights 2/3 and 1/3
    long_run_var = gamma[0] + 2 * (2 / 3 * gamma[1] + 1 / 3 * gamma[2])
    expected = d.mean() / np.sqrt(long_run_var / T)

    stat, _ = diebold_mariano_test(errors_a, errors_b, lags=2)

    assert stat == pytest.approx(expected)


def test_dm_detects_more_accurate_forecast(two_forecasts):
    stat, p_value = diebold_mariano_test(*two_forecasts, h=1)

    assert stat < 0  # A has the lower loss
    assert p_value < 0.01


def test_dm_is_antisymmetric(two_forecasts):
    errors_a, errors_b = two_forecasts

    stat_ab, p_ab = diebold_mariano_test(errors_a, errors_b, h=48)
    stat_ba, p_ba = diebold_mariano_test(errors_b, errors_a, h=48)

    assert stat_ab == pytest.approx(-stat_ba)
    assert p_ab == pytest.approx(p_ba)


def test_dm_lags_default_to_horizon_minus_one(two_forecasts):
    assert diebold_mariano_test(*two_forecasts, h=5) == pytest.approx(
        diebold_mariano_test(*two_forecasts, h=1, lags=4)
    )
