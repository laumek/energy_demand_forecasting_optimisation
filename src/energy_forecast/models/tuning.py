"""Random-search hyperparameter tuning with a fixed trial budget, shared by every model."""

import itertools

import numpy as np
import pandas as pd


def sample_configurations(search_space: dict[str, list], n_trials: int, random_state=None) -> list[dict]:
    """Distinct random configurations from the grid (the whole grid if it has fewer than n_trials)."""
    names = list(search_space)
    grid = [dict(zip(names, values)) for values in itertools.product(*search_space.values())]
    rng = np.random.default_rng(random_state)
    chosen = rng.choice(len(grid), size=min(n_trials, len(grid)), replace=False)
    return [grid[i] for i in chosen]


def tune(search_space: dict[str, list], score_fn, n_trials: int, random_state=None, verbose=True):
    """
    Evaluate n_trials random configurations with score_fn(config), which returns a dict with
    a "score" (lower is better) plus any extra values to keep (e.g. the best epoch count).
    Returns (best, trials): the best configuration merged with its outcome, and a table of
    every trial sorted by score.
    """
    rows = []
    for i, config in enumerate(sample_configurations(search_space, n_trials, random_state), start=1):
        outcome = score_fn(config)
        rows.append({**config, **outcome})
        if verbose:
            print(f"Trial {i}/{n_trials}: {config} -> score {outcome['score']:.4f}")

    best = min(rows, key=lambda row: row["score"])
    return best, pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
