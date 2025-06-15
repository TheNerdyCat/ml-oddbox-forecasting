# ml-oddbox-forecasting
Modular, production-ready forecasting pipeline to predict weekly Oddbox subscription box demand by type.

```
ml-oddbox-forecasting/
├── README.md
├── pyproject.toml            # Poetry-managed dependencies and config
├── poetry.lock
├── data/
│   └── raw/                  # Original data.csv (immutable)
│   └── processed/            # Feature-engineered or intermediate data
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory Data Analysis
│   └── 02_model_dev.ipynb    # Initial modeling
│   └── 03_forecast.ipynb     # Final forecast generation
├── src/
│   └── oddbox_forecasting/
│       ├── __init__.py
│       ├── config.py         # Constants, box types, feature defs
│       ├── data_loader.py    # Load and preprocess data
│       ├── features.py       # Feature engineering functions
│       ├── models.py         # Forecasting model logic
│       ├── pipeline.py       # Full pipeline runner (train → predict)
│       └── utils.py          # Plotting, logging, helpers
├── scripts/
│   └── run_pipeline.py       # Entry point for running the full pipeline
├── forecasts/
│   └── forecast.csv          # Final output: per box type, 4-week forecast
├── visuals/
│   └── plots/                # Time series and uncertainty plots
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_utils.py
├── .gitignore
└── .env                      # Optional: for local path configs
```
