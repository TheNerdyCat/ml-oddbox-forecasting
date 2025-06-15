import pandas as pd
import numpy as np

from oddbox_forecasting.config import FEATURE_LAGS
from oddbox_forecasting.data_loader import load_raw_data
from oddbox_forecasting.utils import (
    check_missing_and_duplicates,
    impute_missing_fortnightly,
    validate_weekly_structure,
)
from oddbox_forecasting.features import add_lag_features, add_cyclic_week_features


def load_and_prepare(path: str) -> pd.DataFrame:
    """Complete raw -> clean dataframe step."""
    df = load_raw_data(path)
    df = impute_missing_fortnightly(df)

    if not validate_weekly_structure(df):
        raise ValueError("One or more weeks do not have 8 box type rows.")

    report = check_missing_and_duplicates(df)
    print("Integrity Report:", report)

    df = add_lag_features(df, "weekly_subscribers", FEATURE_LAGS)
    df = add_cyclic_week_features(df)
    return df
