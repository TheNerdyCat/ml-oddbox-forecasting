import pandas as pd
from pathlib import Path
import pytest
from oddbox_forecasting.config import TEST_NAME, FORECAST_HORIZON
from oddbox_forecasting.pipeline import load_and_prepare, run_demand_forecast_pipeline
from visuals.plots.evaluation import (
    plot_all_evaluation,
    plot_share_model_feature_importances,
)


def run_tests():
    print("Running tests...")
    exit_code = pytest.main(["tests", "-v"])
    if exit_code != 0:
        raise RuntimeError("Tests failed.")
    print("Tests passed.\n")


def main():
    run_tests()
    # Load and process raw data
    raw_path = Path("data/raw/data.csv")
    forecasts_dir = Path("forecasts")
    forecasts_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare(raw_path)

    # Run forecast pipeline
    raw_metrics, adjusted_metrics, full_predictions, total_feat_df, share_feat_df = (
        run_demand_forecast_pipeline(df)
    )

    # Save results
    raw_metrics.to_csv(forecasts_dir / f"{TEST_NAME}_raw_metrics.csv", index=False)
    adjusted_metrics.to_csv(
        forecasts_dir / f"{TEST_NAME}_adjusted_metrics.csv", index=False
    )
    full_predictions.to_csv(forecasts_dir / f"{TEST_NAME}_predictions.csv", index=False)
    total_feat_df.to_csv(
        forecasts_dir / f"{TEST_NAME}_total_feature_importance.csv", index=False
    )
    share_feat_df.to_csv(
        forecasts_dir / f"{TEST_NAME}_share_feature_importance.csv", index=False
    )

    print("\nForecast pipeline complete.")
    print("Raw metrics:\n", raw_metrics.head())
    print("\nAdjusted metrics:\n", adjusted_metrics.head())

    # Extract holdout weeks
    holdout_weeks = (
        full_predictions["week"]
        .sort_values()
        .drop_duplicates()
        .iloc[-FORECAST_HORIZON:]
    )

    # Create directory for visual outputs
    visual_output = Path("visuals/output")
    visual_output.mkdir(parents=True, exist_ok=True)

    # Generate and save plots
    plot_all_evaluation(full_predictions, holdout_weeks, output_dir=str(visual_output))
    plot_share_model_feature_importances(share_feat_df, save_dir="visuals/output")


if __name__ == "__main__":
    main()
