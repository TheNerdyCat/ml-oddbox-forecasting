import pandas as pd


def check_missing_and_duplicates(df: pd.DataFrame) -> dict:
    """Basic integrity checks."""
    return {
        "missing": df.isnull().sum(),
        "duplicates": df.duplicated().sum(),
        "rows_per_week": df.groupby("week").size().value_counts().to_dict(),
    }


def impute_missing_fortnightly(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing fortnightly subscriber count using weekly median."""
    missing_mask = df["fortnightly_subscribers"].isnull()
    if missing_mask.any():
        week = df.loc[missing_mask, "week"].iloc[0]
        imputed_val = df[df["week"] == week]["fortnightly_subscribers"].median()
        df.loc[missing_mask, "fortnightly_subscribers"] = imputed_val
    return df


def validate_weekly_structure(df: pd.DataFrame) -> bool:
    """Ensure each week has exactly 8 rows (1 per box type)."""
    rows_per_week = df.groupby("week").size()
    return all(rows_per_week == 8)


def lagged_correlation(
    df: pd.DataFrame, target_col: str, predictor_col: str, max_lag: int = 2
) -> dict:
    results = {}
    for lag in range(max_lag + 1):
        shifted = df[predictor_col].shift(lag)
        corr = df[target_col].corr(shifted)
        results[f"lag_{lag}"] = corr
    return results


def get_box_type_splits(
    df: pd.DataFrame, feature_cols: list[str], target_col: str
) -> dict:
    """Returns dict of DataFrames split by box_type with only valid rows."""
    return {
        box: group.dropna(subset=feature_cols + [target_col])
        for box, group in df.groupby("box_type")
    }
