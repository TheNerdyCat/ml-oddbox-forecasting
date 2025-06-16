import pandas as pd
import numpy as np
from datetime import datetime
import os

from oddbox_forecasting.config import FEATURE_LAGS, ROLLING_WINDOW
from oddbox_forecasting.data_loader import load_raw_data
from oddbox_forecasting.features import (
    add_lag_features,
    add_box_order_lags,
    add_rolling_box_stats,
    add_event_flags,
    add_cyclic_week_features,
    add_event_interaction_features,
    add_rolling_volatility,
)
from oddbox_forecasting.utils import (
    impute_missing_fortnightly,
    validate_weekly_structure,
    check_missing_and_duplicates,
)


def load_and_prepare(path: str) -> pd.DataFrame:
    """Complete raw -> clean dataframe step and export processed data."""
    df = load_raw_data(path)
    df = impute_missing_fortnightly(df)

    if not validate_weekly_structure(df):
        raise ValueError("One or more weeks do not have 8 box type rows.")

    report = check_missing_and_duplicates(df)
    print("Integrity Report:", report)

    # Feature engineering steps
    df = add_lag_features(df, "weekly_subscribers", FEATURE_LAGS)
    df = add_box_order_lags(df, FEATURE_LAGS)
    df = add_rolling_box_stats(df, window=ROLLING_WINDOW)
    df = add_event_flags(df)
    df = add_cyclic_week_features(df)
    df = add_event_interaction_features(df)
    df = add_rolling_volatility(df, window=ROLLING_WINDOW)

    # Output to processed directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"../data/processed/{timestamp}_processed.csv"
    os.makedirs("../data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")

    return df
