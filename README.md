# Energy demand forecasting with MRFO feature selection

Day-ahead (48 half-hours ahead) forecasting of Great Britain national electricity demand.
LSTM and XGBoost models, each trained on the full feature set and on a subset chosen by binary
Manta Ray Foraging Optimization (MRFO), are compared against multiple linear regression and a
seasonal-naive baseline, with Wilcoxon and Diebold-Mariano tests. Runs are tracked with MLflow.

## Notebooks

1. `01_eda.ipynb`: data cleaning, seasonality (MSTL, DFT), outlier analysis, ACF/PACF lag selection.
2. `02_feature_engineering.ipynb`: correlation / mutual-information checks and construction of the
   day-ahead feature set.
3. `03_model_comparison.ipynb`: MRFO feature selection, hyperparameter tuning, model training,
   evaluation on the full test set, statistical tests and interpretation (SHAP, feature-group
   ablation, Granger causality).

Reusable code lives in `src/energy_forecast/` (data, features, models, evaluation) and is covered
by `tests/`.

## Reproducing the results

1. Install the environment:
   ```
   uv sync
   ```
2. Download the NESO historic demand data (2009–2024) and save it as
   `data/raw/historic_demand_2009_2024.csv`.
3. Download ERA5 temperatures for the seven cities used for the population-weighted temperature.
   This needs a Copernicus Climate Data Store API key in `~/.cdsapirc`:
   ```
   uv run python -m energy_forecast.data.get_weather_data
   ```
4. Clean the demand data and merge in the temperature series, which writes
   `data/processed/demand_with_temperature.csv`:
   ```
   uv run python -m energy_forecast.data.weather
   ```
5. Run the notebooks in order: `01_eda`, `02_feature_engineering` (writes
   `data/processed/model_ready_features_day_ahead.csv`), then `03_model_comparison`.
   Notebook 3 is the slow one: the LSTM MRFO search and LSTM tuning take hours on a CPU;
   a CUDA GPU is recommended.
6. Run the tests:
   ```
   uv run pytest
   ```

MLflow runs are stored locally in `mlruns/`; browse them with `uv run mlflow ui`.
