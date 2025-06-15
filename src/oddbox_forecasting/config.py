# Box types explicitly listed in the dataset
BOX_TYPES = ["LFV", "LV", "MFV", "MV", "SFV", "SV", "XSFV", "FB"]

# Lags to use for time-based feature engineering
FEATURE_LAGS = [1, 2]

# Week-based cyclic encoding
WEEK_CYCLE_PERIOD = 52

# Date format for parsing (if needed in CLI or string conversion)
DATE_FORMAT = "%Y-%m-%d"

ROLLING_WINDOW = 3
