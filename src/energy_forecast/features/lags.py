# lags.py
import pandas as pd

def create_lag_features(df: pd.DataFrame, lags: list[int], column: str = "nd") -> pd.DataFrame:
    """Create lag features for the given column at the specified lags (in steps)."""
    df = df.copy()
    for lag in lags:
        df[f"{column}_lag_{lag}"] = df[column].shift(lag)
    return df


def create_rolling_features(
    df: pd.DataFrame, windows: list[int], column: str = "nd", shift: int = 1
) -> pd.DataFrame:
    """
    Create rolling mean/std features for the given column over the specified windows (in steps).
    Each window ends `shift` steps before the current row, so for a forecast horizon h,
    pass shift=h to use only values available h steps ahead of the target.
    """
    df = df.copy()
    for window in windows:
        shifted = df[column].shift(shift)
        df[f"{column}_rolling_mean_{window}"] = shifted.rolling(window).mean()
        df[f"{column}_rolling_std_{window}"] = shifted.rolling(window).std()
    return df


def create_day_ahead_features(
    df: pd.DataFrame,
    horizon: int,
    lags: list[int],
    windows: list[int],
    lagged_columns: list[str],
    column: str = "nd",
) -> pd.DataFrame:
    """
    Build features that only use information available `horizon` steps before each target:
    lags of `column` (every lag must be >= horizon), rolling mean/std of `column` ending
    `horizon` steps back, and `lagged_columns` replaced by their values `horizon` steps
    earlier (e.g. ifa2_flow -> ifa2_flow_lag_48; the originals are dropped).
    """
    if min(lags) < horizon:
        raise ValueError(f"every lag must be >= horizon ({horizon}), got {lags}")

    df = create_lag_features(df, lags=lags, column=column)
    df = create_rolling_features(df, windows=windows, column=column, shift=horizon)
    for lagged_column in lagged_columns:
        df = create_lag_features(df, lags=[horizon], column=lagged_column)
    return df.drop(columns=lagged_columns)
