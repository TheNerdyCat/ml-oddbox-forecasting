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
