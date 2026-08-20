import pandas as pd
import datetime

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(by= ['settlement_date', 'settlement_period']).reset_index(drop=True)


    # Drop mostly-null columns
    df = df.drop(columns=["eleclink_flow", "nsl_flow", "scottish_transfer", "viking_flow"])
    df = df[df["settlement_period"] <= 48].reset_index(drop=True)

    # Build proper datetime index
    df["period_hour"] = df["settlement_period"].apply(
        lambda x: str(datetime.timedelta(hours=(x - 1) * 0.5))
    )
    df.loc[df["period_hour"] == "1 day, 0:00:00", "period_hour"] = "0:00:00"
    df["settlement_date"] = pd.to_datetime(df["settlement_date"] + " " + df["period_hour"])
    df = df.set_index("settlement_date").sort_index()
    df = df.drop(columns=["period_hour"])

    # Drop only true near-duplicates of the target (multicollinear demand/flow measures)
    df = df.drop(columns=["tsd", "england_wales_demand", "ifa_flow", "moyle_flow"])

    return df