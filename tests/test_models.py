from oddbox_forecasting.models import run_baseline_forecasts
import pandas as pd


def test_run_baseline_forecasts():
    df = pd.DataFrame(
        {
            "week": pd.date_range("2023-01-01", periods=60, freq="W").repeat(2),
            "box_type": ["A", "B"] * 60,
            "box_orders": [80 + i % 5 for i in range(120)],
        }
    )
    result = run_baseline_forecasts(df, forecast_horizon=4)
    assert "A" in result
    assert "rolling_forecast" in result["A"]
    assert len(result["A"]["rolling_forecast"]) == 4
