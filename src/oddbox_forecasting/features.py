import numpy as np
import pandas as pd

from oddbox_forecasting.config import WEEK_CYCLE_PERIOD


def add_lag_features(df: pd.DataFrame, col: str, lags: list[int]) -> pd.DataFrame:
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df.groupby("box_type")[col].shift(lag)
    return df


def add_cyclic_week_features(df: pd.DataFrame) -> pd.DataFrame:
    df["weekofyear"] = df["week"].dt.isocalendar().week
    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / WEEK_CYCLE_PERIOD)
    df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / WEEK_CYCLE_PERIOD)
    return df
