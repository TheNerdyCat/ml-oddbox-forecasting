import pandas as pd
from oddbox_forecasting.features import (
    add_box_order_lags,
    add_event_interaction_features,
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
