import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def run_baseline_forecasts(
    df: pd.DataFrame, forecast_horizon: int = 4, rolling_window: int = 3
) -> dict:
    """Generate rolling average and seasonal naive forecasts per box_type."""
    results = {}

    for box, group in df.groupby("box_type"):
        group = group.sort_values("week").copy()
        past_orders = group["box_orders"].iloc[:-forecast_horizon]

        # Rolling average forecast
        roll_mean = past_orders.rolling(window=rolling_window).mean().iloc[-1]
        rolling_forecast = [roll_mean] * forecast_horizon

        # Seasonal naive (same week last year)
        last_year_indices = [
            -forecast_horizon - 52 + i for i in range(forecast_horizon)
        ]
        if all(idx >= 0 for idx in last_year_indices):
            seasonal_naive = group["box_orders"].iloc[last_year_indices].tolist()
        else:
            seasonal_naive = rolling_forecast  # fallback

        actual = group["box_orders"].iloc[-forecast_horizon:].tolist()

        # Metrics
        rmse_roll = np.sqrt(mean_squared_error(actual, rolling_forecast))
        mae_roll = mean_absolute_error(actual, rolling_forecast)

        rmse_seasonal = np.sqrt(mean_squared_error(actual, seasonal_naive))
        mae_seasonal = mean_absolute_error(actual, seasonal_naive)

        results[box] = {
            "actual": actual,
            "rolling_forecast": rolling_forecast,
            "seasonal_naive": seasonal_naive,
            "rmse_roll": rmse_roll,
            "mae_roll": mae_roll,
            "rmse_seasonal": rmse_seasonal,
            "mae_seasonal": mae_seasonal,
        }

    return results
