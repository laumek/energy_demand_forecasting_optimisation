from pathlib import Path
import pandas as pd

def load_raw_data(path:str | Path = "data/raw/historic_demande_2009_2024.csv") -> pd.DataFrame:
    """Load the raw UK National Grid energy demand dataset"""
    return pd.read_csv(path, index_col=0)