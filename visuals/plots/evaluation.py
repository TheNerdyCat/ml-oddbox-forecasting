import os
import matplotlib.pyplot as plt
import pandas as pd
from oddbox_forecasting.config import BOX_TYPES, TEST_NAME


def _save_or_show(fig, filename: str | None):
    """Helper to either display or save a figure."""
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_forecast_comparison(
    df: pd.DataFrame, holdout_weeks: pd.Series, title: str, save_path: str | None = None
):
    fig, axes = plt.subplots(4, 2, figsize=(20, 16), sharey=True)
    axes = axes.flatten()

    for i, box in enumerate(BOX_TYPES):
        df_box = df[df["box_type"] == box]
        ax = axes[i]

        ax.plot(df_box["week"], df_box["box_orders"], label="Actual", marker="o")
        ax.plot(
            df_box["week"],
            df_box["predicted_box_orders"],
            label="Forecast",
            linestyle="--",
        )
        ax.plot(
            df_box["week"],
            df_box["adjusted_prediction"],
            label="Adjusted",
            linestyle="-.",
        )
        ax.plot(
            df_box["week"], df_box["rolling_baseline"], label="Baseline", linestyle=":"
        )
        ax.axvline(
            holdout_weeks.min(),
            color="red",
            linestyle="--",
            label="Holdout start" if i == 0 else None,
        )

        ax.set_title(box)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True)

    fig.suptitle(title, fontsize=16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_feature_importance(
    model,
    feature_names: list[str],
    title: str = "Feature Importance",
    save_path: str | None = None,
):
    importances = model.feature_importances_
    sorted_idx = importances.argsort()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(feature_names)), importances[sorted_idx], align="center")
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_box_level_errors(df: pd.DataFrame, save_path: str | None = None):
    grouped = (
        df.groupby(["week", "box_type"])
        .agg(
            error=("abs_error", "mean"),
            adjusted_error=("adjusted_abs_error", "mean"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 6))
    for box in BOX_TYPES:
        df_box = grouped[grouped["box_type"] == box]
        ax.plot(df_box["week"], df_box["error"], label=f"{box} Raw", linestyle="--")
        ax.plot(
            df_box["week"], df_box["adjusted_error"], label=f"{box} Adj", linestyle="-"
        )

    ax.set_title("Absolute Error Over Time by Box Type")
    ax.set_ylabel("Absolute Error")
    ax.set_xlabel("Week")
    ax.legend(bbox_to_anchor=(1.05, 1))
    ax.grid(True)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_share_model_feature_importances(
    share_feat_df: pd.DataFrame, save_dir: str | None = None
):
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=share_feat_df, x="feature", y="importance", hue="box_type", ax=ax)
    ax.set_title("Share Model Feature Importances by Box Type")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(f"{save_dir}/{TEST_NAME}_share_model_feature_importance.png")
        plt.close(fig)
    else:
        plt.show()


def plot_all_evaluation(
    df_preds: pd.DataFrame, holdout_weeks: pd.Series, output_dir: str | None = None
):
    plot_forecast_comparison(
        df_preds,
        holdout_weeks,
        title="Forecasts: Actual vs Baseline vs Forecast vs Adjusted",
        save_path=(
            os.path.join(output_dir, f"{TEST_NAME}_forecast_comparison.png")
            if output_dir
            else None
        ),
    )
    plot_box_level_errors(
        df_preds,
        save_path=(
            os.path.join(output_dir, f"{TEST_NAME}_error_timeseries.png")
            if output_dir
            else None
        ),
    )
