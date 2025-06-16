# Box types explicitly listed in the dataset
BOX_TYPES = ["LFV", "LV", "MFV", "MV", "SFV", "SV", "XSFV", "FB"]

# Lags to use for time-based feature engineering
FEATURE_LAGS = [1, 2]

# Week-based cyclic encoding
WEEK_CYCLE_PERIOD = 52

# Date format for parsing (if needed in CLI or string conversion)
DATE_FORMAT = "%Y-%m-%d"

# Rolling window for smoothing and baselines
ROLLING_WINDOW = 3

# Forecast horizon (number of holdout weeks at the end)
FORECAST_HORIZON = 4

# Feature columns used in share models
SHARE_FEATURES = [
    "week_sin",
    "week_cos",
    "is_marketing_week",
    "holiday_week",
    "box_orders_lag_1",
    "box_orders_lag_2",
    "weekly_subscribers_lag_1",
    "weekly_subscribers_lag_2",
]

TEST_NAME = "001"
