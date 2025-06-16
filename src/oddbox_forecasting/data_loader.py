import pandas as pd
from pathlib import Path


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """
    Load and preprocess the raw Oddbox dataset from a CSV file.

    This function reads a CSV file, parses the 'week' column as datetime,
    fixes known data entry issues in the 'box_orders' column (e.g., "1O0" → "100"),
    and coerces the 'box_orders' column to numeric.

    Parameters:
        path (str | Path): Path to the raw CSV file.

    Returns:
        pd.DataFrame: A cleaned DataFrame with parsed dates and numeric box order values.
    """
    df = pd.read_csv(path)

    # Convert columns
    df["week"] = pd.to_datetime(df["week"], errors="coerce")
    df["box_orders"] = df["box_orders"].replace("1O0", "100")  # known typo fix
    df["box_orders"] = pd.to_numeric(df["box_orders"], errors="coerce")

    return df
