import pandas as pd
from oddbox_forecasting.utils import (
    impute_missing_fortnightly,
)


def test_impute_missing_fortnightly():
    df = pd.DataFrame(
        {
            "week": ["2024-06-10"] * 2,
            "fortnightly_subscribers": [150.0, None],
        }
    )
    filled = impute_missing_fortnightly(df)
    assert filled["fortnightly_subscribers"].isnull().sum() == 0
