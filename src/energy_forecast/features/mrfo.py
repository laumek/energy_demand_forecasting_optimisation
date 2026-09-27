"""
Wrapper feature selection with binary Manta Ray Foraging Optimization (MRFO).

MRFO follows Zhao, Zhang & Wang (2020), "Manta ray foraging optimization: An effective
bio-inspired optimizer for engineering applications", Engineering Applications of
Artificial Intelligence 87, 103300. Each manta ray has a continuous position in
[0, 1]^n_features; a feature is selected when its coordinate is above 0.5.

Candidate feature subsets are scored by an objective to minimise:

    objective = alpha * normalised_validation_error + (1 - alpha) * n_selected / n_features

where normalised_validation_error = validation MSE / variance of the validation target,
so it is dimensionless and comparable across fitness functions.
"""

from typing import NamedTuple

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from energy_forecast.models.lstm_model import (
    DEVICE,
    LSTM,
    predict_test_set,
    prepare_lstm_data,
    train_model,
)
import torch
import torch.nn as nn


class SearchResult(NamedTuple):
    best_mask: np.ndarray  # binary vector, 1 = feature selected
    best_objective: float
    history: list[float]  # best objective so far after each objective call
    n_unique_evaluations: int  # number of distinct feature subsets actually trained
    # The two parts of best_objective. If the penalty is more than a small tie-breaker next to
    # the error term, alpha is too low and MRFO may be dropping genuinely useful features
    best_error_term: float  # alpha * normalised validation error
    best_penalty_term: float  # (1 - alpha) * n_selected / n_features


def xgboost_fitness_function(features: np.ndarray, X, y) -> float:
    """
    Evaluate a candidate feature subset by training a fast XGBoost model and
    returning its normalised validation MSE (lower is better).
    """
    selected_indices = np.where(features == 1)[0]
    if len(selected_indices) == 0:
        return np.inf

    X_selected = X.iloc[:, selected_indices]

    X_train, X_val, y_train, y_val = train_test_split(
        X_selected, y, test_size=0.2, shuffle=False
    )

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    return mean_squared_error(y_val, preds) / np.var(y_val)


def lstm_fitness_function(
    features: np.ndarray,
    X,
    y,
    seq_length: int = 48,
    batch_size: int = 336,
    hidden_size: int = 64,
    num_layers: int = 2,
    num_epochs: int = 5,
    learning_rate: float = 0.001,
    seed: int = 42,
) -> float:
    """
    Evaluate a candidate feature subset by training a small LSTM and returning its
    normalised validation MSE (lower is better). Slower than the XGBoost fitness
    function, intended to validate MRFO's selection against the original thesis's
    target model. Seeded so the same subset always gets the same score.
    """
    selected_indices = np.where(features == 1)[0]
    if len(selected_indices) == 0:
        return np.inf

    X_selected = X.iloc[:, selected_indices]

    X_train, X_val, y_train, y_val = train_test_split(
        X_selected, y, test_size=0.2, shuffle=False
    )

    train_loader, val_loader, x_scaler, y_scaler, y_train_arr, y_val_arr = prepare_lstm_data(
        X_train, X_val, y_train, y_val, seq_length, batch_size
    )

    torch.manual_seed(seed)
    model = LSTM(X_selected.shape[1], hidden_size, num_layers, output_size=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model, _ = train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

    val_results = predict_test_set(model, val_loader, y_val_arr, y_scaler)
    return val_results["rmse"] ** 2 / np.var(y_val_arr)


def _make_objective(fitness_function, X, y, alpha, fitness_kwargs):
    """
    Build the penalised objective, cached by feature mask (many continuous positions
    map to the same subset, and each subset only needs training once).
    """
    n_features = X.shape[1]
    cache: dict[bytes, float] = {}
    history: list[float] = []

    def objective(mask: np.ndarray) -> float:
        key = mask.tobytes()
        if key not in cache:
            if not mask.any():
                cache[key] = np.inf
            else:
                error = fitness_function(mask, X, y, **fitness_kwargs)
                cache[key] = alpha * error + (1 - alpha) * mask.sum() / n_features
        score = cache[key]
        history.append(min(score, history[-1]) if history else score)
        return score

    return objective, cache, history


def _search_result(best_mask, best_score, history, cache, alpha) -> SearchResult:
    penalty_term = (1 - alpha) * best_mask.sum() / len(best_mask)
    return SearchResult(
        best_mask, float(best_score), history, len(cache),
        best_error_term=float(best_score - penalty_term), best_penalty_term=float(penalty_term),
    )


def _to_mask(position: np.ndarray) -> np.ndarray:
    return (position > 0.5).astype(int)


def manta_ray_foraging_optimization(
    X,
    y,
    fitness_function=xgboost_fitness_function,
    num_iterations: int = 10,
    num_manta_rays: int = 10,
    alpha: float = 0.99,
    somersault_factor: float = 2.0,
    random_state: int | None = None,
    verbose: bool = True,
    **fitness_kwargs,
) -> SearchResult:
    """
    Run binary MRFO feature selection over the columns of X.

    Each iteration applies chain or cyclone foraging (50/50 per ray), then somersault
    foraging, updating the best-so-far position after each phase. fitness_function
    determines which model scores candidate subsets (xgboost_fitness_function by
    default, or lstm_fitness_function for a slower, more thorough search).
    """
    rng = np.random.default_rng(random_state)
    objective, cache, history = _make_objective(fitness_function, X, y, alpha, fitness_kwargs)

    n_features = X.shape[1]
    T = num_iterations

    positions = rng.random((num_manta_rays, n_features))
    scores = np.array([objective(_to_mask(p)) for p in positions])
    best_idx = int(np.argmin(scores))
    best_position, best_score = positions[best_idx].copy(), scores[best_idx]

    def update_best():
        nonlocal best_position, best_score
        idx = int(np.argmin(scores))
        if scores[idx] < best_score:
            best_position, best_score = positions[idx].copy(), scores[idx]

    for t in range(1, T + 1):
        # Phase 1: chain or cyclone foraging, using the positions from the start of the iteration
        new_positions = np.empty_like(positions)
        for i in range(num_manta_rays):
            r = rng.random(n_features)

            if rng.random() < 0.5:
                # Cyclone foraging: spiral around a reference point. Early on (t/T small) the
                # reference is usually a random point (exploration), later the best (exploitation)
                r1 = rng.random(n_features)
                beta = 2 * np.exp(r1 * (T - t + 1) / T) * np.sin(2 * np.pi * r1)
                reference = rng.random(n_features) if t / T < rng.random() else best_position
                previous = reference if i == 0 else positions[i - 1]
                new_positions[i] = (
                    reference + r * (previous - positions[i]) + beta * (reference - positions[i])
                )
            else:
                # Chain foraging: move towards the ray in front and towards the best position
                r_alpha = 1 - rng.random(n_features)  # in (0, 1], so log is finite
                chain_alpha = 2 * r_alpha * np.sqrt(np.abs(np.log(r_alpha)))
                previous = best_position if i == 0 else positions[i - 1]
                new_positions[i] = (
                    positions[i]
                    + r * (previous - positions[i])
                    + chain_alpha * (best_position - positions[i])
                )

        positions = np.clip(new_positions, 0, 1)
        scores = np.array([objective(_to_mask(p)) for p in positions])
        update_best()

        # Phase 2: somersault foraging around the best position
        r2 = rng.random(positions.shape)
        r3 = rng.random(positions.shape)
        positions = np.clip(
            positions + somersault_factor * (r2 * best_position - r3 * positions), 0, 1
        )
        scores = np.array([objective(_to_mask(p)) for p in positions])
        update_best()

        if verbose:
            print(
                f"Iteration {t}/{T} — best objective: {best_score:.5f}, "
                f"features: {_to_mask(best_position).sum()}/{n_features}, "
                f"unique subsets evaluated: {len(cache)}"
            )

    return _search_result(_to_mask(best_position), best_score, history, cache, alpha)


def random_search(
    X,
    y,
    fitness_function=xgboost_fitness_function,
    num_evaluations: int = 100,
    alpha: float = 0.99,
    random_state: int | None = None,
    verbose: bool = True,
    **fitness_kwargs,
) -> SearchResult:
    """
    Baseline: score uniformly random feature subsets with the same objective as MRFO.
    Pass num_evaluations=len(mrfo_result.history) to give it MRFO's evaluation budget.
    """
    rng = np.random.default_rng(random_state)
    objective, cache, history = _make_objective(fitness_function, X, y, alpha, fitness_kwargs)

    n_features = X.shape[1]
    best_mask, best_score = np.zeros(n_features, dtype=int), np.inf

    for _ in range(num_evaluations):
        mask = rng.integers(0, 2, size=n_features)
        score = objective(mask)
        if score < best_score:
            best_mask, best_score = mask, score

    if verbose:
        print(
            f"Random search — best objective: {best_score:.5f}, "
            f"features: {best_mask.sum()}/{n_features}, unique subsets evaluated: {len(cache)}"
        )

    return _search_result(best_mask, best_score, history, cache, alpha)
