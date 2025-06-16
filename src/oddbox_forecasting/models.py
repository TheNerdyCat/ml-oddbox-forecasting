import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
from oddbox_forecasting.config import BOX_TYPES


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


def train_total_model(
    df_total_train: pd.DataFrame,
) -> tuple[lgb.LGBMRegressor, np.ndarray]:
    X = df_total_train.drop(columns=["week", "total_orders"])
    y = df_total_train["total_orders"]
    model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, model.feature_importances_, X.columns.tolist()


def train_share_models(df_model_train: pd.DataFrame, share_features: list[str]) -> dict:
    models = {}
    for box in BOX_TYPES:
        df_box = df_model_train[df_model_train["box_type"] == box].dropna(
            subset=["box_share"] + share_features
        )
        X = df_box[share_features]
        y = df_box["box_share"]
        model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        models[box] = {
            "model": model,
            "train_rows": len(X),
            "train_weeks": df_box["week"].nunique(),
            "feature_importance": model.feature_importances_,
        }
    return models


def evaluate_share_model(
    df_model: pd.DataFrame,
    df_total: pd.DataFrame,
    share_models: dict,
    share_features: list[str],
    holdout_weeks: pd.Series,
) -> pd.DataFrame:
    df_model = df_model.copy()
    df_model["total_pred"] = df_model["week"].map(
        df_total.set_index("week")["total_pred"]
    )
    all_preds = []
    for box in BOX_TYPES:
        df_box = df_model[df_model["box_type"] == box].dropna(
            subset=share_features + ["box_orders", "total_pred"]
        )
        model = share_models[box]["model"]
        df_box = df_box.sort_values("week")
        X = df_box[share_features]
        df_box["predicted_share"] = model.predict(X)
        df_box["predicted_box_orders"] = (
            df_box["predicted_share"] * df_box["total_pred"]
        )

        # Baseline
        df_box["rolling_baseline"] = (
            df_box["box_orders"].shift(1).rolling(window=3).mean()
        )

        df_box["squared_error"] = (
            df_box["box_orders"] - df_box["predicted_box_orders"]
        ) ** 2
        df_box["abs_error"] = (
            df_box["box_orders"] - df_box["predicted_box_orders"]
        ).abs()
        df_box["baseline_squared_error"] = (
            df_box["box_orders"] - df_box["rolling_baseline"]
        ) ** 2
        df_box["baseline_abs_error"] = (
            df_box["box_orders"] - df_box["rolling_baseline"]
        ).abs()

        all_preds.append(df_box)
    return pd.concat(all_preds)


def compute_metrics(df: pd.DataFrame, split_label: str) -> pd.DataFrame:
    return (
        df.groupby("box_type")
        .agg(
            rmse=("squared_error", lambda x: np.sqrt(np.mean(x))),
            mae=("abs_error", "mean"),
            rmse_baseline=("baseline_squared_error", lambda x: np.sqrt(np.mean(x))),
            mae_baseline=("baseline_abs_error", "mean"),
            mean_actual=("box_orders", "mean"),
        )
        .assign(mape=lambda d: 100 * d["mae"] / d["mean_actual"], split=split_label)
        .reset_index()
    )


def calculate_event_uplift(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate % uplift or dampening effect from events by box_type."""
    impacts = []

    for box in BOX_TYPES:
        df_box = df[df["box_type"] == box].copy()

        for col, label in [
            ("is_marketing_week", "marketing"),
            ("holiday_week", "holiday"),
        ]:
            base = df_box[df_box[col] == 0]["box_orders"]
            event = df_box[df_box[col] == 1]["box_orders"]

            if len(base) >= 5 and len(event) >= 2:
                uplift = (event.mean() - base.mean()) / base.mean()
                impacts.append({"box_type": box, "event_type": label, "uplift": uplift})

    return pd.DataFrame(impacts)


def apply_adjustment_layer(df: pd.DataFrame, uplift_df: pd.DataFrame) -> pd.DataFrame:
    """Apply learned uplift/dampening adjustments to predicted_box_orders."""
    df = df.copy()

    for _, row in uplift_df.iterrows():
        mask = df["box_type"] == row["box_type"]

        if row["event_type"] == "marketing":
            df.loc[mask & (df["is_marketing_week"] == 1), "adjusted_prediction"] = df[
                "predicted_box_orders"
            ] * (1 + row["uplift"])

        elif row["event_type"] == "holiday":
            df.loc[mask & (df["holiday_week"] == 1), "adjusted_prediction"] = df[
                "predicted_box_orders"
            ] * (1 + row["uplift"])

    df["adjusted_prediction"] = df["adjusted_prediction"].fillna(
        df["predicted_box_orders"]
    )
    return df


def compute_adjusted_metrics(df: pd.DataFrame, split_label: str) -> pd.DataFrame:
    """Return RMSE/MAE for adjusted vs raw forecasts."""
    return (
        df.groupby("box_type")
        .agg(
            rmse_model=("squared_error", lambda x: np.sqrt(np.mean(x))),
            mae_model=("abs_error", "mean"),
            rmse_baseline=("baseline_squared_error", lambda x: np.sqrt(np.mean(x))),
            mae_baseline=("baseline_abs_error", "mean"),
            rmse_adjusted=("adjusted_squared_error", lambda x: np.sqrt(np.mean(x))),
            mae_adjusted=("adjusted_abs_error", "mean"),
            mean_actual=("box_orders", "mean"),
        )
        .assign(
            mape_model=lambda d: 100 * d["mae_model"] / d["mean_actual"],
            mape_adjusted=lambda d: 100 * d["mae_adjusted"] / d["mean_actual"],
            split=split_label,
        )
        .reset_index()
    )
