import pandas as pd


def check_missing_and_duplicates(df: pd.DataFrame) -> dict:
    """
    Perform basic data integrity checks.

    Checks for:
      - Missing values.
      - Duplicate rows.
      - Row count distribution across weeks.

    Parameters:
        df (pd.DataFrame): Input dataframe to check.

    Returns:
        dict: Summary containing counts of missing values, duplicates,
              and a frequency distribution of rows per week.
    """
    return {
        "missing": df.isnull().sum(),
        "duplicates": df.duplicated().sum(),
        "rows_per_week": df.groupby("week").size().value_counts().to_dict(),
    }


def impute_missing_fortnightly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing 'fortnightly_subscribers' values using the median for the affected week.

    Parameters:
        df (pd.DataFrame): Input dataframe with possible missing values.

    Returns:
        pd.DataFrame: Dataframe with missing values in 'fortnightly_subscribers' filled.
    """
    missing_mask = df["fortnightly_subscribers"].isnull()
    if missing_mask.any():
        week = df.loc[missing_mask, "week"].iloc[0]
        imputed_val = df[df["week"] == week]["fortnightly_subscribers"].median()
        df.loc[missing_mask, "fortnightly_subscribers"] = imputed_val
    return df


def validate_weekly_structure(df: pd.DataFrame) -> bool:
    """
    Validate that each week has exactly 8 entries (one per box_type).

    Parameters:
        df (pd.DataFrame): Input dataframe.

    Returns:
        bool: True if all weeks contain exactly 8 rows, False otherwise.
    """
    rows_per_week = df.groupby("week").size()
    return all(rows_per_week == 8)


def lagged_correlation(
    df: pd.DataFrame, target_col: str, predictor_col: str, max_lag: int = 2
) -> dict:
    """
    Compute correlation between a target column and a lagged predictor.

    Parameters:
        df (pd.DataFrame): Input dataframe.
        target_col (str): Column name of the dependent variable.
        predictor_col (str): Column name of the predictor to lag.
        max_lag (int): Maximum number of lag steps to compute.

    Returns:
        dict: Lag number mapped to correlation coefficient (e.g., {"lag_0": 0.5, ...}).
    """
    results = {}
    for lag in range(max_lag + 1):
        shifted = df[predictor_col].shift(lag)
        corr = df[target_col].corr(shifted)
        results[f"lag_{lag}"] = corr
    return results


def get_box_type_splits(
    df: pd.DataFrame, feature_cols: list[str], target_col: str
) -> dict:
    """
    Split dataframe into separate DataFrames per box_type, dropping rows with missing values.

    Parameters:
        df (pd.DataFrame): Full dataset.
        feature_cols (list[str]): List of required feature columns.
        target_col (str): Name of the target column.

    Returns:
        dict: Mapping from box_type to filtered DataFrame with no missing values in relevant columns.
    """
    return {
        box: group.dropna(subset=feature_cols + [target_col])
        for box, group in df.groupby("box_type")
    }
