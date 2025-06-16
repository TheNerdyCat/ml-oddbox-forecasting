import numpy as np
import pandas as pd

from oddbox_forecasting.config import WEEK_CYCLE_PERIOD


def add_lag_features(df: pd.DataFrame, col: str, lags: list[int]) -> pd.DataFrame:
    """
    Add lagged versions of a specified column per box type.

    For each lag in `lags`, creates a new column containing the lagged values
    of `col`, grouped by 'box_type'.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the target column.
        col (str): Column name to lag.
        lags (list[int]): List of lag values to apply.

    Returns:
        pd.DataFrame: DataFrame with new lagged feature columns added.
    """
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df.groupby("box_type")[col].shift(lag)
    return df


def add_cyclic_week_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclic sine and cosine transformations for the week of year.

    Encodes the seasonal cycle of weeks into two continuous variables,
    useful for capturing seasonality in models.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a 'week' datetime column.

    Returns:
        pd.DataFrame: DataFrame with added 'week_sin' and 'week_cos' columns.
    """
    df["weekofyear"] = df["week"].dt.isocalendar().week
    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / WEEK_CYCLE_PERIOD)
    df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / WEEK_CYCLE_PERIOD)
    return df


def add_box_order_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """
    Add lagged features for 'box_orders' per box type.

    For each lag in `lags`, creates new columns containing previous values
    of 'box_orders', grouped by 'box_type'.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'box_orders' and 'box_type'.
        lags (list[int]): List of lag values to apply.

    Returns:
        pd.DataFrame: DataFrame with new lagged 'box_orders' columns.
    """
    for lag in lags:
        df[f"box_orders_lag_{lag}"] = df.groupby("box_type")["box_orders"].shift(lag)
    return df


def add_rolling_box_stats(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Add rolling mean and standard deviation of 'box_orders' per box type.

    Computes rolling statistics over a specified window to capture recent trends
    and volatility.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'box_orders' and 'box_type'.
        window (int): Rolling window size in weeks (default is 3).

    Returns:
        pd.DataFrame: DataFrame with 'box_orders_rollmean' and 'box_orders_rollstd'.
    """
    df["box_orders_rollmean"] = df.groupby("box_type")["box_orders"].transform(
        lambda x: x.rolling(window).mean()
    )
    df["box_orders_rollstd"] = df.groupby("box_type")["box_orders"].transform(
        lambda x: x.rolling(window).std()
    )
    return df


def add_event_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a combined flag column indicating marketing or holiday event weeks.

    Adds 'is_event_week' as the sum of binary 'is_marketing_week' and 'holiday_week'.

    Parameters:
        df (pd.DataFrame): Input DataFrame with event indicator columns.

    Returns:
        pd.DataFrame: DataFrame with added 'is_event_week' column.
    """
    df["is_event_week"] = df["is_marketing_week"].astype(int) + df[
        "holiday_week"
    ].astype(int)
    return df


def add_event_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between box type and event flags.

    Adds string-based features combining 'box_type' with both marketing and holiday flags.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'box_type', 'is_marketing_week', and 'holiday_week'.

    Returns:
        pd.DataFrame: DataFrame with 'marketing_x_box' and 'holiday_x_box' columns.
    """
    df["is_marketing_week"] = df["is_marketing_week"].astype(int)
    df["holiday_week"] = df["holiday_week"].astype(int)

    df["marketing_x_box"] = (
        df["box_type"] + "_MKT_" + df["is_marketing_week"].astype(str)
    )
    df["holiday_x_box"] = df["box_type"] + "_HOL_" + df["holiday_week"].astype(str)

    return df


def add_rolling_volatility(df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Add rolling standard deviation (volatility) of 'box_orders' per box type.

    Helps quantify variability in recent demand for each box type.

    Parameters:
        df (pd.DataFrame): Input DataFrame with 'box_orders' and 'box_type'.
        window (int): Rolling window size in weeks (default is 3).

    Returns:
        pd.DataFrame: DataFrame with added 'box_orders_volatility' column.
    """
    """Adds a rolling std deviation feature of box_orders per box type."""
    df["box_orders_volatility"] = df.groupby("box_type")["box_orders"].transform(
        lambda x: x.rolling(window).std()
    )
    return df


def get_model_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract model input features (X) and target variable (y) from the DataFrame.

    Verifies that all required features are present before returning them.

    Parameters:
        df (pd.DataFrame): Input DataFrame with all engineered features.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Tuple containing:
            - X: Feature matrix with selected columns.
            - y: Target series ('box_orders').
    """
    feature_cols = [
        "weekly_subscribers",
        "fortnightly_subscribers",
        "weekly_subscribers_lag_1",
        "weekly_subscribers_lag_2",
        "box_orders_lag_1",
        "box_orders_lag_2",
        "box_orders_rollmean",
        "box_orders_rollstd",
        "box_orders_volatility",
        "week_sin",
        "week_cos",
        "is_marketing_week",
        "holiday_week",
        "is_event_week",
        "marketing_x_box",
        "holiday_x_box",
    ]

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected features: {missing}")

    X = df[feature_cols].copy()
    y = df["box_orders"].copy()
    return X, y
