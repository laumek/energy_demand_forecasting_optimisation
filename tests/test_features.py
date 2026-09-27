import numpy as np
import pandas as pd
import pytest

from energy_forecast.features.lags import create_day_ahead_features, create_rolling_features

HORIZON = 48


def test_rolling_features_end_shift_steps_back():
    df = pd.DataFrame({"nd": np.arange(100, dtype=float)})

    result = create_rolling_features(df, windows=[3], shift=HORIZON)

    # Row 60 averages rows 10, 11, 12 (the 3 values ending 48 steps before row 60)
    assert result.loc[60, "nd_rolling_mean_3"] == pytest.approx(np.mean([10, 11, 12]))


def build(df):
    return create_day_ahead_features(
        df, horizon=HORIZON, lags=[48, 336], windows=[48, 336], lagged_columns=["ifa2_flow"]
    )


def test_day_ahead_features_ignore_the_last_horizon_steps():
    """Changing demand/grid values at time t0 must not change any feature before t0 + HORIZON."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"nd": rng.normal(size=800), "ifa2_flow": rng.normal(size=800)})
    t0 = 600

    perturbed = df.copy()
    perturbed.loc[t0, ["nd", "ifa2_flow"]] += 1000

    features = build(df).drop(columns="nd")
    features_perturbed = build(perturbed).drop(columns="nd")

    pd.testing.assert_frame_equal(features.loc[: t0 + HORIZON - 1], features_perturbed.loc[: t0 + HORIZON - 1])
    # ...and the change does show up exactly HORIZON steps later
    assert not features.loc[t0 + HORIZON].equals(features_perturbed.loc[t0 + HORIZON])


def test_day_ahead_features_replace_lagged_columns():
    df = pd.DataFrame({"nd": np.arange(400.0), "ifa2_flow": np.arange(400.0)})

    result = build(df)

    assert "ifa2_flow" not in result.columns
    assert result.loc[100, "ifa2_flow_lag_48"] == 52


def test_day_ahead_features_reject_lags_shorter_than_horizon():
    df = pd.DataFrame({"nd": np.arange(400.0)})

    with pytest.raises(ValueError):
        create_day_ahead_features(df, horizon=HORIZON, lags=[1, 48], windows=[48], lagged_columns=[])
