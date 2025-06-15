from oddbox_forecasting.data_loader import load_raw_data
from oddbox_forecasting.utils import (
    check_missing_and_duplicates,
    impute_missing_fortnightly,
    validate_weekly_structure,
)


def load_and_prepare_data(path: str):
    """Complete raw -> clean dataframe step."""
    df = load_raw_data(path)
    df = impute_missing_fortnightly(df)

    if not validate_weekly_structure(df):
        raise ValueError("One or more weeks do not have 8 box type rows.")

    report = check_missing_and_duplicates(df)
    print("Integrity Report:", report)

    return df
