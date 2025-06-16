from oddbox_forecasting.models import (
    run_baseline_forecasts,
    train_total_model,
    train_share_models,
    compute_metrics,
    calculate_event_uplift,
    apply_adjustment_layer,
)
from oddbox_forecasting.config import FORECAST_HORIZON
import pandas as pd


def test_run_baseline_forecasts():
    df = pd.DataFrame(
        {
            "week": pd.date_range("2023-01-01", periods=60, freq="W").repeat(2),
            "box_type": ["A", "B"] * 60,
            "box_orders": [80 + i % 5 for i in range(120)],
        }
    )
    result = run_baseline_forecasts(df, forecast_horizon=FORECAST_HORIZON)
    assert "A" in result
    assert "rolling_forecast" in result["A"]
    assert len(result["A"]["rolling_forecast"]) == FORECAST_HORIZON


def test_train_total_model_runs():
    df = pd.DataFrame(
        {
            "week": pd.date_range("2024-01-01", periods=8),
            "total_orders": range(100, 108),
            "weekly_subscribers": range(200, 208),
            "fortnightly_subscribers": range(50, 58),
            "weekly_subscribers_lag_1": range(195, 203),
            "weekly_subscribers_lag_2": range(190, 198),
            "is_marketing_week": [0, 1] * 4,
            "holiday_week": [0, 0] * 4,
            "is_event_week": [0] * 8,
            "week_sin": [0.1] * 8,
            "week_cos": [0.9] * 8,
        }
    )
    model, _, _ = train_total_model(df)
    assert hasattr(model, "predict")


def test_compute_metrics_shape():
    df = pd.DataFrame(
        {
            "box_type": ["A"] * 4,
            "box_orders": [100, 110, 105, 108],
            "predicted_box_orders": [98, 112, 107, 109],
            "rolling_baseline": [95, 107, 103, 110],
        }
    )
    df["squared_error"] = (df["box_orders"] - df["predicted_box_orders"]) ** 2
    df["abs_error"] = (df["box_orders"] - df["predicted_box_orders"]).abs()
    df["baseline_squared_error"] = (df["box_orders"] - df["rolling_baseline"]) ** 2
    df["baseline_abs_error"] = (df["box_orders"] - df["rolling_baseline"]).abs()

    metrics = compute_metrics(df, split_label="test")
    assert "rmse" in metrics.columns


def test_calculate_event_uplift_shape():
    df = pd.DataFrame(
        {
            "box_type": ["LFV"] * 10 + ["MV"] * 10,
            "box_orders": [100] * 5 + [120] * 5 + [90] * 5 + [85] * 5,
            "is_marketing_week": [0] * 5 + [1] * 5 + [0] * 5 + [1] * 5,
            "holiday_week": [0] * 10 + [0] * 5 + [1] * 5,
        }
    )
    result = calculate_event_uplift(df)
    assert {"box_type", "event_type", "uplift"} <= set(result.columns)
    assert set(result["event_type"]) <= {"marketing", "holiday"}


def test_apply_adjustment_layer_modifies_rows():
    df = pd.DataFrame(
        {
            "box_type": ["A", "A"],
            "is_marketing_week": [1, 0],
            "holiday_week": [0, 0],
            "predicted_box_orders": [100.0, 100.0],
        }
    )
    uplift_df = pd.DataFrame(
        [{"box_type": "A", "event_type": "marketing", "uplift": 0.1}]
    )
    out = apply_adjustment_layer(df, uplift_df)
    assert abs(out.loc[0, "adjusted_prediction"] - 110.0) < 1e-6
    assert out.loc[1, "adjusted_prediction"] == 100.0
