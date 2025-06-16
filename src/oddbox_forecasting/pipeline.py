import pandas as pd
import numpy as np
from datetime import datetime
import os

from oddbox_forecasting.config import (
    FEATURE_LAGS,
    ROLLING_WINDOW,
    TEST_NAME,
    SHARE_FEATURES,
    FORECAST_HORIZON,
)
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
    calculate_event_uplift,
    apply_adjustment_layer,
    compute_adjusted_metrics,
)


def load_and_prepare(path: str) -> pd.DataFrame:
    """
    Load raw Oddbox data and apply cleaning and feature engineering steps.

    This includes:
      - Parsing raw columns and fixing known typos.
      - Imputing missing fortnightly subscriber values.
      - Validating structure (one row per box type per week).
      - Adding lag features, rolling stats, cyclic encodings, and interaction terms.
      - Saving processed dataset to disk.

    Parameters:
        path (str): File path to the raw data CSV.

    Returns:
        pd.DataFrame: Cleaned and feature-enriched dataset ready for modeling.
    """
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
    output_path = f"data/processed/{TEST_NAME}_processed.csv"
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")

    return df


def run_demand_forecast_pipeline(df_model: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end pipeline for training, predicting, and evaluating Oddbox demand forecasts.

    Steps:
      - Aggregates total orders and computes box share.
      - Trains total volume and share models (per box_type).
      - Makes predictions and computes evaluation metrics.
      - Applies adjustment layer based on event uplift analysis.
      - Returns raw and adjusted forecasts, errors, and feature importances.

    Parameters:
        df_model (pd.DataFrame): Preprocessed input dataframe with features.

    Returns:
        tuple:
            - pd.DataFrame: Raw forecast evaluation metrics (train/test).
            - pd.DataFrame: Adjusted forecast metrics (test only).
            - pd.DataFrame: All predictions, actuals, and error terms.
            - pd.DataFrame: Feature importances from total volume model.
            - pd.DataFrame: Feature importances from share models by box_type.
    """
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
        )
        .sort_values("week")
        .reset_index()
        .fillna(method="ffill")
        .fillna(method="bfill")
    )

    holdout_weeks = df_total["week"].iloc[-FORECAST_HORIZON:]
    total_model, total_importance, total_features = train_total_model(
        df_total[~df_total["week"].isin(holdout_weeks)]
    )
    df_total["total_pred"] = total_model.predict(
        df_total.drop(columns=["week", "total_orders"])
    )

    share_models = train_share_models(
        df_model[~df_model["week"].isin(holdout_weeks)], SHARE_FEATURES
    )

    # Feature importances as DataFrames
    total_feat_df = pd.DataFrame(
        {
            "feature": total_features,
            "importance": total_importance,
        }
    )

    share_feat_dfs = []
    for box, model_info in share_models.items():
        share_feat_dfs.append(
            pd.DataFrame(
                {
                    "box_type": box,
                    "feature": SHARE_FEATURES,
                    "importance": model_info["feature_importance"],
                }
            )
        )
    share_feat_df = pd.concat(share_feat_dfs)

    df_all_preds = evaluate_share_model(
        df_model, df_total, share_models, SHARE_FEATURES, holdout_weeks
    )

    # Apply adjustment layer
    uplift_df = calculate_event_uplift(df_all_preds)
    df_all_preds = apply_adjustment_layer(df_all_preds, uplift_df)

    # Compute adjustment errors
    df_all_preds["adjusted_squared_error"] = (
        df_all_preds["box_orders"] - df_all_preds["adjusted_prediction"]
    ) ** 2
    df_all_preds["adjusted_abs_error"] = (
        df_all_preds["box_orders"] - df_all_preds["adjusted_prediction"]
    ).abs()

    df_train = df_all_preds[~df_all_preds["week"].isin(holdout_weeks)]
    df_test = df_all_preds[df_all_preds["week"].isin(holdout_weeks)]

    # Combine raw and adjusted metrics
    raw_metrics = pd.concat(
        [
            compute_metrics(df_train, "train"),
            compute_metrics(df_test, "test"),
        ]
    )

    adj_metrics = compute_adjusted_metrics(df_test, "test_adjusted")

    return raw_metrics, adj_metrics, df_all_preds, total_feat_df, share_feat_df
