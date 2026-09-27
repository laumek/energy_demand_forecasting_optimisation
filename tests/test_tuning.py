from energy_forecast.models.tuning import sample_configurations, tune

SPACE = {"a": [1, 2, 3], "b": ["x", "y"]}


def test_sample_configurations_are_distinct_and_capped_by_grid_size():
    configs = sample_configurations(SPACE, n_trials=4, random_state=0)
    assert len(configs) == 4
    assert len({tuple(c.items()) for c in configs}) == 4

    assert len(sample_configurations(SPACE, n_trials=100, random_state=0)) == 6


def test_sample_configurations_are_reproducible():
    assert sample_configurations(SPACE, 3, random_state=1) == sample_configurations(SPACE, 3, random_state=1)


def test_tune_returns_lowest_score_with_extra_outcomes():
    best, trials = tune(
        SPACE,
        lambda config: {"score": config["a"] + (config["b"] == "y"), "extra": config["a"] * 10},
        n_trials=6,
        verbose=False,
    )

    assert best == {"a": 1, "b": "x", "score": 1, "extra": 10}
    assert trials["score"].is_monotonic_increasing
    assert len(trials) == 6
