import pickle
from pathlib import Path


def cached(path, compute, recompute: bool = False):
    """
    Return the result saved at `path` if it exists, otherwise run compute(), save its result
    there and return it. Used for the slow steps (MRFO searches, tuning), so a lost session or
    kernel restart doesn't mean redoing hours of work. Delete the file, or pass recompute=True,
    to recompute (e.g. after changing settings that aren't part of the file name).
    """
    path = Path(path)
    if path.exists() and not recompute:
        print(f"Loaded saved result: {path}")
        with path.open("rb") as f:
            return pickle.load(f)

    result = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(result, f)
    print(f"Saved result: {path}")
    return result
