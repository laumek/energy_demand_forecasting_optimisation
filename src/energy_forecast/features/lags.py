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