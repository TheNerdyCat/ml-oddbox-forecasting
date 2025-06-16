import pandas as pd
from oddbox_forecasting.features import (
    add_box_order_lags,
    add_event_interaction_features,
    add_rolling_volatility,
    get_model_features,
)


def test_add_box_order_lags():
    df = pd.DataFrame({"box_type": ["A"] * 5, "box_orders": [10, 20, 30, 40, 50]})
    result = add_box_order_lags(df, [1])
    assert "box_orders_lag_1" in result.columns
    assert result["box_orders_lag_1"].isnull().sum() == 1


def test_add_event_interaction_features():
    df = pd.DataFrame(
        {
            "box_type": ["A", "B"],
            "is_marketing_week": [True, False],
            "holiday_week": [False, True],
        }
    )
    df = add_event_interaction_features(df)
    assert "marketing_x_box" in df.columns
    assert "holiday_x_box" in df.columns
    assert df.loc[0, "marketing_x_box"] == "A_MKT_1"


def test_add_rolling_volatility():
    df = pd.DataFrame({"box_type": ["A"] * 5, "box_orders": [10, 20, 30, 40, 50]})
    result = add_rolling_volatility(df, window=3)
    assert "box_orders_volatility" in result.columns
    assert result["box_orders_volatility"].isnull().sum() >= 2  # due to window size


def test_get_model_features_shape():
    df = pd.DataFrame(
        {
            "weekly_subscribers": [100],
            "fortnightly_subscribers": [50],
            "weekly_subscribers_lag_1": [90],
            "weekly_subscribers_lag_2": [80],
            "box_orders_lag_1": [75],
            "box_orders_lag_2": [70],
            "box_orders_rollmean": [72.5],
            "box_orders_rollstd": [2.5],
            "box_orders_volatility": [3.0],
            "week_sin": [0.5],
            "week_cos": [0.86],
            "is_marketing_week": [1],
            "holiday_week": [0],
            "is_event_week": [1],
            "marketing_x_box": ["MV_MKT_1"],
            "holiday_x_box": ["MV_HOL_0"],
            "box_orders": [85],
        }
    )
    X, y = get_model_features(df)
    assert X.shape[1] == 16
    assert y.iloc[0] == 85
