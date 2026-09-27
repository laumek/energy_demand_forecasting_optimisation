import numpy as np
import pandas as pd

from energy_forecast.data.preprocess import (
    MOSTLY_MISSING_COLUMNS,
    TARGET_DUPLICATE_COLUMNS,
    clean_data,
)


def raw_day(date, periods):
    periods = list(periods)
    return pd.DataFrame({
        "settlement_date": date,
        "settlement_period": periods,
        "nd": 1000,
        **{column: 0 for column in MOSTLY_MISSING_COLUMNS + TARGET_DUPLICATE_COLUMNS},
    })


def london(timestamp):
    return pd.Timestamp(timestamp, tz="Europe/London")


def test_clean_data_handles_clock_change_days():
    raw = pd.concat([
        raw_day("2021-03-27", range(1, 49)),  # normal day
        raw_day("2021-03-28", range(1, 47)),  # spring forward: 46 periods
        raw_day("2021-10-31", range(1, 51)),  # fall back: 50 periods
    ], ignore_index=True)

    df = clean_data(raw)

    assert len(df) == 48 + 46 + 50
    assert not df.index.has_duplicates
    # Spring: period 3 is the first half-hour after the clock jumps from 01:00 to 02:00
    assert df.index[48 + 2] == london("2021-03-28 02:00")
    assert df.index[48 + 45] == london("2021-03-28 23:30")
    # Autumn: period 50 ends the day at 23:30 local time
    assert df.index[-1] == london("2021-10-31 23:30")
    # Every half-hour is exactly 30 minutes of elapsed time after the previous one within a day
    for day in ["2021-03-28", "2021-10-31"]:
        day_index = df.index[df.index.strftime("%Y-%m-%d") == day]
        assert (np.diff(day_index.tz_convert("UTC").asi8) == 30 * 60 * 10**9).all()


def test_clean_data_drops_periods_outside_their_date():
    # Period 47 doesn't exist on a 46-period day: it would land on the next day
    raw = raw_day("2021-03-28", list(range(1, 47)) + [47])

    df = clean_data(raw)

    assert len(df) == 46
    assert df.index.max() == london("2021-03-28 23:30")


def test_clean_data_drops_unused_columns_and_indexes_by_timestamp():
    df = clean_data(raw_day("2021-03-27", range(1, 49)))

    assert df.index.name == "settlement_date"
    assert "settlement_date" not in df.columns
    assert not set(MOSTLY_MISSING_COLUMNS + TARGET_DUPLICATE_COLUMNS) & set(df.columns)
