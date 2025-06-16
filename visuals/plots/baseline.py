import matplotlib.pyplot as plt


def plot_baseline_forecasts(results: dict, horizon: int = 4):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
    axes = axes.flatten()

    for idx, (box, res) in enumerate(results.items()):
        ax = axes[idx]
        x = range(1, horizon + 1)

        ax.plot(x, res["actual"], marker="o", label="Actual")
        ax.plot(
            x, res["rolling_forecast"], marker="x", linestyle="--", label="Rolling Avg"
        )
        ax.plot(
            x, res["seasonal_naive"], marker="^", linestyle=":", label="Seasonal Naïve"
        )

        ax.set_title(
            f"{box} | RMSE (Roll): {res['rmse_roll']:.1f}, RMSE (Seasonal): {res['rmse_seasonal']:.1f}"
        )
        ax.set_xlabel("Forecast Week")
        ax.set_ylabel("Orders")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=3, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Baseline Forecasts per Box Type", fontsize=16)
    plt.show()
