import pandas as pd

# Interconnectors with 70-90% missing values (only recorded from 2019-2023 onwards)
MOSTLY_MISSING_COLUMNS = ["eleclink_flow", "nsl_flow", "scottish_transfer", "viking_flow"]
# Demand measures that are near-duplicates of the target (nd) by definition
TARGET_DUPLICATE_COLUMNS = ["tsd", "england_wales_demand"]

LOCAL_TZ = "Europe/London"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw demand data and index it by a tz-aware Europe/London timestamp.

    Settlement periods count half-hours from local midnight, so clock-change days have 46
    (spring) or 50 (autumn) periods. Each timestamp is local midnight plus the elapsed
    half-hours, which places every period correctly on those days too. Periods that don't
    fall on their own settlement date (logging errors) are dropped.
    """
    df = df.drop(columns=MOSTLY_MISSING_COLUMNS + TARGET_DUPLICATE_COLUMNS)

    settlement_day = pd.to_datetime(df["settlement_date"])
    local_midnight = settlement_day.dt.tz_localize(LOCAL_TZ)
    # Adding a Timedelta to a tz-aware timestamp adds elapsed time, so clock changes are handled
    timestamps = local_midnight + pd.to_timedelta((df["settlement_period"] - 1) * 30, unit="min")

    on_own_date = (timestamps.dt.date == settlement_day.dt.date).to_numpy()
    df = df[on_own_date].drop(columns=["settlement_date"])
    df.index = pd.DatetimeIndex(timestamps[on_own_date], name="settlement_date")
    df = df.sort_index()

    if df.index.has_duplicates:
        raise ValueError("Duplicate timestamps after cleaning; check the raw settlement periods.")

    return df
