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
from oddbox_forecasting.models import (
    train_total_model,
    train_share_models,
    evaluate_share_model,
    compute_metrics,
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


def run_demand_forecast_pipeline(df_model: pd.DataFrame) -> pd.DataFrame:
    df_model["week"] = pd.to_datetime(df_model["week"])
    df_model["total_orders"] = df_model.groupby("week")["box_orders"].transform("sum")
    df_model["box_share"] = df_model["box_orders"] / df_model["total_orders"]

    df_total = (
        df_model.groupby("week")
        .agg(
            total_orders=("box_orders", "sum"),
            weekly_subscribers=("weekly_subscribers", "first"),
            fortnightly_subscribers=("fortnightly_subscribers", "first"),
            weekly_subscribers_lag_1=("weekly_subscribers_lag_1", "first"),
            weekly_subscribers_lag_2=("weekly_subscribers_lag_2", "first"),
            is_marketing_week=("is_marketing_week", "first"),
            holiday_week=("holiday_week", "first"),
            is_event_week=("is_event_week", "first"),
            week_sin=("week_sin", "first"),
            week_cos=("week_cos", "first"),
        )
        .sort_values("week")
        .reset_index()
        .fillna(method="ffill")
        .fillna(method="bfill")
    )

    holdout_weeks = df_total["week"].iloc[-4:]
    df_total["total_pred"] = train_total_model(
        df_total[~df_total["week"].isin(holdout_weeks)]
    ).predict(df_total.drop(columns=["week", "total_orders"]))

    share_features = ["week_sin", "week_cos", "is_marketing_week", "holiday_week"]
    share_models = train_share_models(
        df_model[~df_model["week"].isin(holdout_weeks)], share_features
    )

    df_all_preds = evaluate_share_model(
        df_model, df_total, share_models, share_features, holdout_weeks
    )

    df_train = df_all_preds[~df_all_preds["week"].isin(holdout_weeks)]
    df_test = df_all_preds[df_all_preds["week"].isin(holdout_weeks)]

    return pd.concat(
        [
            compute_metrics(df_train, "train"),
            compute_metrics(df_test, "test"),
        ]
    )
