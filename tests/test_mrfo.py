import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from energy_forecast.features.mrfo import manta_ray_foraging_optimization, random_search

N_FEATURES = 20
INFORMATIVE = [0, 1, 2]


@pytest.fixture
def toy_data():
    """y depends on the first three features only; the other 17 are pure noise."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(400, N_FEATURES)))
    y = pd.Series(3 * X[0] + 2 * X[1] - 2 * X[2] + rng.normal(scale=0.5, size=400))
    return X, y


def linear_fitness(features, X, y):
    """Fast stand-in for the XGBoost/LSTM fitness functions: normalised validation MSE."""
    X_selected = X.iloc[:, np.where(features == 1)[0]]
    split = int(0.8 * len(X))
    model = LinearRegression().fit(X_selected[:split], y[:split])
    preds = model.predict(X_selected[split:])
    return mean_squared_error(y[split:], preds) / np.var(y[split:])


def run_mrfo(X, y, **kwargs):
    params = dict(
        fitness_function=linear_fitness,
        num_iterations=30,
        num_manta_rays=20,
        random_state=42,
        verbose=False,
    )
    params.update(kwargs)
    return manta_ray_foraging_optimization(X, y, **params)


def test_mrfo_selects_informative_features(toy_data):
    result = run_mrfo(*toy_data)

    assert result.best_mask[INFORMATIVE].all()
    assert result.best_mask[3:].sum() <= 2


def test_mrfo_beats_random_search_on_same_budget(toy_data):
    mrfo = run_mrfo(*toy_data)
    baseline = random_search(
        *toy_data,
        fitness_function=linear_fitness,
        num_evaluations=len(mrfo.history),
        random_state=42,
        verbose=False,
    )

    assert mrfo.best_objective <= baseline.best_objective


def test_history_matches_budget_and_never_increases(toy_data):
    num_rays, num_iterations = 20, 30
    result = run_mrfo(*toy_data, num_manta_rays=num_rays, num_iterations=num_iterations)

    # One call per ray at initialisation, then two per ray per iteration (after each phase)
    assert len(result.history) == num_rays * (2 * num_iterations + 1)
    assert all(b <= a for a, b in zip(result.history, result.history[1:]))
    assert result.history[-1] == result.best_objective


def test_same_random_state_gives_same_result(toy_data):
    first = run_mrfo(*toy_data)
    second = run_mrfo(*toy_data)

    np.testing.assert_array_equal(first.best_mask, second.best_mask)
    assert first.history == second.history


def test_each_subset_is_trained_once(toy_data):
    calls = []

    def counting_fitness(features, X, y):
        calls.append(features.tobytes())
        return linear_fitness(features, X, y)

    result = run_mrfo(*toy_data, fitness_function=counting_fitness)

    assert len(calls) == len(set(calls))
    assert len(calls) <= result.n_unique_evaluations < len(result.history)


def test_objective_splits_into_error_and_penalty_terms(toy_data):
    alpha = 0.99
    result = run_mrfo(*toy_data, alpha=alpha)

    assert result.best_error_term + result.best_penalty_term == pytest.approx(result.best_objective)
    assert result.best_penalty_term == pytest.approx((1 - alpha) * result.best_mask.sum() / N_FEATURES)
