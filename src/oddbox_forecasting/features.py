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


def add_box_order_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    for lag in lags:
        df[f"box_orders_lag_{lag}"] = df.groupby("box_type")["box_orders"].shift(lag)
    return df


def add_rolling_box_stats(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    df["box_orders_rollmean"] = df.groupby("box_type")["box_orders"].transform(
        lambda x: x.rolling(window).mean()
    )
    df["box_orders_rollstd"] = df.groupby("box_type")["box_orders"].transform(
        lambda x: x.rolling(window).std()
    )
    return df


def add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["is_event_week"] = df["is_marketing_week"].astype(int) + df[
        "holiday_week"
    ].astype(int)
    return df


def add_event_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["is_marketing_week"] = df["is_marketing_week"].astype(int)
    df["holiday_week"] = df["holiday_week"].astype(int)

    df["marketing_x_box"] = (
        df["box_type"] + "_MKT_" + df["is_marketing_week"].astype(str)
    )
    df["holiday_x_box"] = df["box_type"] + "_HOL_" + df["holiday_week"].astype(str)

    return df
