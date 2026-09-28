from energy_forecast.cache import cached


def test_cached_computes_once_then_loads(tmp_path):
    calls = []

    def compute():
        calls.append(1)
        return {"answer": 42}

    path = tmp_path / "sub" / "result.pkl"
    assert cached(path, compute) == {"answer": 42}
    assert cached(path, compute) == {"answer": 42}
    assert len(calls) == 1


def test_cached_recompute_overwrites(tmp_path):
    path = tmp_path / "result.pkl"
    cached(path, lambda: 1)

    assert cached(path, lambda: 2, recompute=True) == 2
    assert cached(path, lambda: 3) == 2
