# Oddbox Forecasting Pipeline

## Overview

This project forecasts weekly box demand by box type (`FB`, `LFV`, `LV`, etc.) using a two-stage modeling approach. The aim is to generate accurate, interpretable forecasts that outperform simple baselines, while allowing for dynamic event adjustments (e.g. marketing campaigns or holidays).

## Repository Structure

```bash
edwardsimsps@Edwards-MacBook-Pro ml-oddbox-forecasting % tree -I "__pycache__|*.csv|*.png"
.
├── data
│   ├── processed
│   └── raw
├── forecasts
├── LICENSE
├── notebooks
│   ├── 01_eda.ipynb
│   └── 02_model_dev.ipynb
├── poetry.lock
├── pyproject.toml
├── README.md
├── scripts
│   └── run_pipeline.py
├── src
│   └── oddbox_forecasting
│       ├── __init__.py
│       ├── config.py
│       ├── data_loader.py
│       ├── features.py
│       ├── models.py
│       ├── pipeline.py
│       └── utils.py
├── tests
│   ├── run_tests.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_utils.py
└── visuals
    ├── output
    └── plots
        ├── baseline.py
        └── evaluation.py

13 directories, 20 files
```

## Approach

The forecasting pipeline uses a two-stage modeling strategy designed to separate overall demand dynamics from customer preference dynamics:

### 1. **Total Demand Forecasting (Stage 1)**  
The total number of boxes ordered per week is modelled, regardless of box type. This captures high-level demand signals driven by:
- Subscriber counts (weekly and fortnightly)
- Seasonal trends (via cyclic week features)
- Event flags (marketing weeks, holidays)

A LightGBM regressor is usedfor its strong performance on tabular data, built-in handling of missing values, and interpretability via feature importances.

### 2. **Box Share Modeling (Stage 2)**  
Next, we train separate share models per box type, each predicting the fraction of total demand attributable to that box in a given week. These models take features such as:
- Event flags
- Lags of subscriber and box order metrics
- Rolling stats and volatility
- Box-event interaction terms

Combining both stages, the below is computed:

```bash
predicted_box_orders = predicted_total_orders * predicted_box_share
```

This modular approach allows us to decouple overall demand from product mix and makes the models easier to debug, update, and interpret.

### Adjustment Layer

An adjustment layer is applied post-prediction to account for known external events like holidays or marketing campaigns. The adjustment values are learned from historical data, where uplift or dampening effects per box type is estimates based on differences in demand during vs. outside of these events.

This layer allows us to inject causal priors into the final forecast in a simple and interpretable way.


## Why This Forecasting Method?

I chose this decompositional approach (total demand × share) because:
- It reflects how planning decisions are made: total supply and then allocation.
- It makes the modeling more interpretable and maintainable.
- It avoids overfitting by reducing the dimensionality of each task.
- Each component can be evaluated and improved independently.

## Key Assumptions

- **Stationarity**: Past patterns of uplift/dampening from events (e.g., +10% in marketing weeks) will continue to hold.
- **Demand determinism**: Box preferences (shares) are driven by modeled features and not random noise or external shocks.
- **Weekly granularity is sufficient**: I assume that finer resolution (e.g., daily) isn't necessary for decision-making.
- **Holdout weeks are representative**: Future weeks will look like past test periods in terms of seasonality and event patterns.

## What I'd Do with More Time or Data

- **Confidence Intervals**: Use LightGBM quantile regression or residual-based methods to generate upper/lower bounds around forecasts.
- **Daily-Level Modeling**: Aggregate up to weekly forecasts if finer-grained data becomes available.
- **Customer Segmentation**: Incorporate behavioral or demographic data to model box preference by segment.
- **Feature Enrichment**: Add weather, campaign metadata, or search trends.
- **Modeling Interactions**: Introduce global models that learn interactions across box types or a multitask learning setup.


## Future Event Incorporation

Known events can be integrated directly into the pipeline by:
- **Encoding them as binary flags** (`is_bank_holiday`, `is_promotion`)
- **Quantifying expected uplift** (e.g. from past campaign performance)
- **Hardcoding adjustment values** for non-recurring events (e.g., one-off national events)
- **Simulating scenarios** by toggling event inputs for forward-looking forecasting

We can also maintain a calendar of planned events and automatically incorporate it into the modeling and adjustment layers.

## Measuring Forecast Performance in Production

To ensure reliability in production, I'd:
- Track rolling RMSE, MAE, and MAPE weekly
- Monitor drift in input features and prediction residuals
- Set up alerts for large deviations from historical norms or forecast error thresholds
- Log performance per box type to catch regressions
- Build a dashboard summarizing forecast accuracy vs. actuals

I would also implement backtesting for new models before deployment to ensure robust performance on historical holdouts.


## Modeling Pipeline

### 1. **Data Preparation**
- Raw data is loaded and validated for structure (8 rows per week).
- Missing values (e.g., fortnightly subscribers) are imputed.
- Lag, rolling, cyclic, and event interaction features are engineered.
- Processed data is saved under `data/processed/{TEST_NAME}_processed.csv`.

### 2. **Model Training**
- **Total model** is trained using LightGBM on aggregated weekly totals.
- **Share models** are trained separately per box type using LightGBM, targeting the `box_share`.

### 3. **Prediction**
- Total demand predictions are multiplied by share predictions for each box type.
- A rolling 3-week average baseline is used for comparison.

### 4. **Adjustment Layer**
- Uplift values for `is_marketing_week` and `holiday_week` are learned per box type.
- These are used to adjust the forecasts where relevant.

### 5. **Evaluation**
- RMSE, MAE, and MAPE are calculated for:
  - Raw predictions
  - Rolling baseline
  - Adjusted predictions
- Feature importances are recorded.

## Running the Pipeline

Ensure the `data.csv` has been added to the `data/raw` dir.

Run the full pipeline from the root directory using:

```bash
poetry run python -m scripts.run_pipeline
```

This will:
 - Run tests
 - Process the raw data
 - Train models
 - Generate predictions
 - Save metrics and visualizations


## Output Files
All outputs are saved to either forecasts/ or visuals/output/:
 - `forecasts/`
   - `{TEST_NAME}_processed.csv`: Cleaned and feature-engineered data
   - `{TEST_NAME}_predictions.csv`: Final predictions and errors
   - `{TEST_NAME}_raw_metrics.csv`: Model vs baseline errors (raw)
   - `{TEST_NAME}_adjusted_metrics.csv`: Model vs baseline errors (after adjustment)
   - `{TEST_NAME}_total_feature_importance.csv`: Total model feature importances
   - `{TEST_NAME}_share_feature_importance.csv`: Share model importances per box
 - visuals/output/
   - `forecast_comparison.png`: Actual vs Forecast vs Adjusted
   - `error_timeseries.png`: Absolute errors over time
   - `share_model_feature_importance.png`: Importance bar plot by box type

## Configuration

Configurable parameters are located in oddbox_forecasting/config.py:
 - `FEATURE_LAGS`: Lags for lag features (e.g. [1, 2])
 - `ROLLING_WINDOW`: Rolling window size for baseline and features
 - `SHARE_FEATURES`: Features used in share model
 - `FORECAST_HORIZON`: Number of holdout weeks for testing
 - `TEST_NAME`: Unique name used for output filenames
 - `BOX_TYPES`: List of valid box types

## Next Steps
 - Add confidence intervals to forecasts, such as:
 - Quantile predictions from LightGBM (quantile objective)
 - Bootstrapping or prediction intervals based on residuals
 - Visual shading around forecast lines in plots

## Testing
Run all tests with:

```bash
poetry run pytest
```

Unit tests cover:
 - Feature engineering
 - Model training
 - Adjustment logic
 - Output structure