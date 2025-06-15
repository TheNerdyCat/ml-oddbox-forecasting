import pandas as pd
from pathlib import Path


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw Oddbox data with initial parsing."""
    df = pd.read_csv(path)

    # Convert columns
    df["week"] = pd.to_datetime(df["week"], errors="coerce")
    df["box_orders"] = df["box_orders"].replace("1O0", "100")  # known typo fix
    df["box_orders"] = pd.to_numeric(df["box_orders"], errors="coerce")

    return df
