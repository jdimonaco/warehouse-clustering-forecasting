
import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid # pyright: ignore[reportMissingModuleSource]
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Dict, Tuple, Any, Optional
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from IPython.display import display
import logging

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)  # keep Prophet/Stan progress logs out of the output
logging.getLogger("prophet").setLevel(logging.WARNING)


"""# Forecasting with Prophet

Why Use:

- Prophet models trend and seasonality with minimal tuning. It handles missing data and changepoints, and can include holiday effects when a calendar is provided.

Benefits:

- Fast fitting; clear decomposition into trend/seasonality/holidays.

Target Results:

- Side-by-side performance vs SARIMAX; clear seasonality/trend insights per cluster.

Business Value:

- Provides an alternative forecast and validation, improving confidence with transparent, interpretable results.

## Step 25:  Prepare Data for Prophet

What It Does:

- The data for each cluster is reshaped into two columns: one for dates (ds) and one for values (y), using monthly data. This format is required by Prophet to run forecasts.


Why It’s Important:
- Prophet requires a specific input format to function correctly. Without this structure, the model cannot interpret the time series.
-  Using a monthly frequency aligns the forecasts with typical business planning cycles, making the outputs more actionable for operational decision-making.
"""

def prepare_prophet_data(ts: pd.Series, cluster_col: str = 'cluster') -> Dict[int, pd.DataFrame]:
    """
    Prepares time series data for Prophet modelling for each cluster.

    Steps
    ------
    - Initialises an empty dictionary to store Prophet-ready DataFrames.
    - Iterates through each unique cluster ID, prints preparation message.
    - Selects the time series, resets index, and renames columns to 'ds' and 'y'.
    - Converts 'ds' to datetime, sets as index, resamples to monthly frequency, and forward-fills missing dates.
    - Resets index again and stores the prepared DataFrame in prophet_data.
    - Displays the head of the DataFrame and returns the dictionary.

    Parameters
    ----------
    ts : pd.Series
        Multi-indexed time series with 'cluster' and datetime index, containing monthly volumes.
    cluster_col : str
        Name of the cluster level in the index (default is 'cluster').

    Returns
    -------
    dict
        Dictionary of Prophet-ready DataFrames per cluster.
    """
    prophet_data = {}

    for c in sorted(ts.index.get_level_values(cluster_col).unique()):
        print(f"Preparing Prophet data for Cluster {c}.")

        series = ts.loc[c]

        df_p = series.reset_index().rename(columns={'Received date': 'ds', 'monthly_volume': 'y'})
        df_p['ds'] = pd.to_datetime(df_p['ds'])
        df_p = df_p.set_index('ds').asfreq('M').ffill().reset_index()

        prophet_data[c] = df_p
        display(df_p.head())

    return prophet_data

"""## Step 26: Train Prophet models

What It Does:

- Trains a Prophet model for each cluster using its training time series. The model learns the historical patterns, including trend and seasonality, to make future predictions.

Why It’s Important:

- Fitting the model to the training data allows it to capture the unique characteristics of each cluster's volume history, which is essential for generating accurate forecasts.
"""

def train_prophet_models(ts_train: Dict[int, pd.Series]) -> Dict[int, Prophet]:
    """
    Trains a Prophet model for each cluster using its training time series.

    Steps
    ------
    - Initialises empty dictionary for trained Prophet models.
    - Iterates through clusters and training series, prints training message.
    - Prepares data using helper function, skips if empty.
    - Fits Prophet model and stores in dictionary.
    - Prints completion message and returns trained models.

    Parameters
    ----------
    ts_train : dict
        Dictionary of training time series per cluster.

    Returns
    -------
    dict
        Dictionary of trained Prophet models per cluster.
    """
    prophet_models = {}

    for c, series in ts_train.items():
        print(f"Training Prophet model for Cluster {c} on training data...")

        df_p_train = series.reset_index().rename(columns={'Received date': 'ds', 'monthly_volume': 'y'})
        df_p_train['ds'] = pd.to_datetime(df_p_train['ds'])

        if df_p_train.empty:
            print(f"Skipping Cluster {c}: Training data is empty.")
            continue

        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        m.fit(df_p_train)
        prophet_models[c] = m
        print(f"→ Done for Cluster {c}")

    return prophet_models

"""##  Step 27: Generate and Visualise Prophet Forecasts

What it does:

- Generates and visualises baseline Prophet forecasts (training, actual test, and forecast).
- Visualises decomposed components (trend, seasonality).

Why It's Important:

- Visually assesses how well the baseline Prophet captures patterns and forecasts align with actual test data before tuning.
- Helps understand main time series drivers per cluster.

Output Analysis:

- Cluster 0: Prophet fits the training data closely, including the December drops, and continues the pattern into 2024 with a slight under-forecast in late summer.

- Cluster 1: Captures the seasonal cycle well; some 2024 peaks and troughs are smaller or larger than forecast.

- Cluster 2: Picks up the recovering trend better than SARIMAX, although it still sits slightly below the 2024 actuals.

Conclusion:

- Prophet's flexible trend handles the change in direction in Cluster 2 better than SARIMAX, while both models capture the regular seasonal patterns in Clusters 0 and 1.
"""

def generate_and_visualise_prophet_forecasts(prophet_models: Dict[int, Prophet],ts_train: Dict[int, pd.Series],
                                             ts_test: Dict[int, pd.Series]) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.Series]]:
    """
    Generates forecasts using trained Prophet models for each cluster and plots results.

    Steps
    ------
    - Initialises empty dictionaries for future dataframes and Prophet forecasts.
    - Iterates through trained Prophet models.
    - Prints forecasting message.
    - Generates forecast.
    - Stores forecast.
    - Plots forecast with actuals.
    - Plots components.
    - Prints separator.
    - Returns dictionaries of future dataframes and Prophet forecasts.

    Parameters
    ----------
    prophet_models : dict
        Dictionary of trained Prophet models per cluster.
    ts_train : dict
        Dictionary of training time series per cluster.
    ts_test : dict
        Dictionary of actual test time series per cluster.

    Returns
    -------
    tuple
        Tuple containing:
        - Dictionary of future dataframes per cluster.
        - Dictionary of forecasted values for the test period per cluster.
    """
    prophet_futures = {}
    prophet_forecasts = {}

    for c, m in prophet_models.items():
        print(f"Forecasting for Cluster {c}...")

        periods = len(ts_test[c])
        future = m.make_future_dataframe(periods=periods, freq='M')
        prophet_futures[c] = future

        # Generate forecast
        fcst = m.predict(future)

        # Save forecast for the test period
        prophet_forecasts[c] = fcst.set_index('ds')['yhat'].loc[ts_test[c].index]

        # Show forecast tail
        print("Forecast tail:")
        display(fcst[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

        # Plot forecast with actuals
        plt.figure(figsize=(14, 6))
        plt.plot(ts_train[c].index, ts_train[c].values, label='Training Data', color='orange')
        plt.plot(ts_test[c].index, ts_test[c].values, label='Actual Test Data', color='blue')
        plt.plot(fcst['ds'], fcst['yhat'], label='Prophet Forecast', color='green', linestyle='--')

        plt.title(f'Baseline Prophet Forecast vs. Actuals for Cluster {c}')
        plt.xlabel('Date')
        plt.ylabel('Monthly Volume')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot components (specific to Prophet)
        fig2 = m.plot_components(fcst)
        fig2.suptitle(f'Components – Cluster {c}', y=1.02)
        plt.tight_layout()
        plt.show()

        print("-" * 50)

    return prophet_futures, prophet_forecasts

"""## Step 28: Evaluate Prophet Forecast Accuracy

What It Does:

- Evaluates the accuracy of the baseline Prophet forecast for each cluster using MAE, RMSE, and MAPE on the test period.

Why It's Important:

- Quantifies how well Prophet predicts unseen data.
- Lower values indicate better accuracy.
- Establishes a baseline before tuning and comparison.

Output Analysis:

- Cluster 0: MAPE 5.0%, slightly behind SARIMAX (4.87%).

- Cluster 1: MAPE 5.0%, slightly behind SARIMAX (4.61%).

- Cluster 2: MAPE 4.5%, well ahead of SARIMAX (8.42%).

Conclusion:

- Untuned, Prophet is close to SARIMAX for the seasonal clusters and much better for the trending one (Cluster 2).
"""

def evaluate_prophet_forecasts(ts_test: Dict[int, pd.Series], prophet_forecasts: Dict[int, pd.Series]) -> Dict[int, Dict[str, float]]:
    """
    Evaluates Prophet forecasts for each cluster using standard error metrics.

    Steps
    ------
    - Initialises an empty dictionary to store evaluation metrics.
    - Iterates through each cluster for which forecasts are available.
    - Retrieves actual and predicted time series for the current cluster.
    - Aligns actual and predicted data.
    - Calculates evaluation metrics (MAE, RMSE, MAPE).
    - Stores the calculated metrics in the results dictionary.
    - Prints evaluation results for each cluster.
    - Returns the dictionary containing the evaluation results per cluster.

    Parameters
    ----------
    ts_test : dict
        Dictionary of actual test time series per cluster.
    prophet_forecasts : dict
        Dictionary of forecasted values per cluster from Prophet.

    Returns
    -------
    dict
        Dictionary containing MAE, RMSE, and MAPE for each cluster.
    """
    prophet_metrics = {}

    for c in sorted(ts_test.keys()):
        if c not in prophet_forecasts:
             continue

        actual = ts_test[c]
        pred = prophet_forecasts[c].loc[actual.index]

        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mape = (np.abs((actual - pred) / actual.replace(0, np.nan))).dropna().mean() * 100

        prophet_metrics[c] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

        print(f"Cluster {c} → Prophet MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%")


    return prophet_metrics

"""## Step 29: Prophet Parameter Tuning

What it does:

- This step tests different model settings to see which ones work best. It checks how flexible the trend should be and how seasonality is handled. The model is trained and tested on different parts of the training data, and the best settings are chosen based on the lowest forecast error (MAPE).

Why it's important:

- Selecting the best-fitting parameters through cross-validation on the training data helps ensure the model captures each cluster’s volume behaviour more accurately on unseen data.
"""

def tune_prophet_parameters(ts_train: Dict[int, pd.Series],param_grid: Dict[str, list],min_months_for_cv: int = 24,n_forecasts: int = 12,
                            period_months: int = 6) -> Dict[int, Optional[Dict[str, Any]]]:
    """
    Tunes Prophet hyperparameters for each cluster using cross-validation and selects the best configuration based on MAPE.

    Steps
    ------
    - Initialise dictionary for best parameters; iterate clusters, print tuning message, prepare Prophet data.
    - Skip if data insufficient; else calculate CV window, print settings, initialise best MAPE and parameters.
    - Iterate parameter combinations: fit models, perform cross-validation, calculate mean MAPE, update best if improved.
    - Store results per cluster, print best MAPE or failure message, and return dictionary of best parameters.

    Parameters
    ----------
    ts_train : dict
        Dictionary of training time series per cluster.
    param_grid : dict
        Dictionary specifying the grid of hyperparameters to test.
    min_months_for_cv : int
        Minimum number of months required for cross-validation (default is 24).
    n_forecasts : int
        Number of periods (months) to forecast (default is 12).
    period_months : int
        Period between cross-validation folds in months (default is 6).

    Returns
    -------
    dict
        Dictionary of best hyperparameters per cluster, or None if tuning was not possible.
    """
    best_prophet_params = {}

    for c, series in ts_train.items():
        print(f"\nTuning Cluster {c}...")

        df_p_train = series.reset_index()
        df_p_train.columns = ['ds', 'y']
        df_p_train['ds'] = pd.to_datetime(df_p_train['ds'])

        if len(df_p_train) < min_months_for_cv:
            print(f"  Skipping: Not enough data ({len(df_p_train)} months). Need at least {min_months_for_cv} months for cross-validation.")
            best_prophet_params[c] = None
            continue

        horizon_days = n_forecasts * 30
        period_days = period_months * 30
        total_train_days = len(df_p_train) * 30
        min_initial_days = min_months_for_cv * 30
        initial_window_days = max(min_initial_days, total_train_days - horizon_days - period_days - period_days)


        if initial_window_days < min_initial_days or initial_window_days + horizon_days + period_days > total_train_days:
            print(f"  Skipping Cluster {c}: Cannot find valid cross-validation window with {horizon_days}-day horizon and {period_days}-day period for {len(df_p_train)} months of data.")
            best_prophet_params[c] = None
            continue


        initial_window = f"{initial_window_days} days"
        period = f"{period_days} days"
        horizon = f"{horizon_days} days"

        print(f"  Using CV settings: initial={initial_window}, period={period}, horizon={horizon}")


        best_mape = float('inf')
        best_params = None

        for params in ParameterGrid(param_grid):
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, **params)

            try:
                model.fit(df_p_train)
                df_cv = cross_validation(model, initial=initial_window, period=period, horizon=horizon, disable_tqdm=True)

                if not df_cv.empty:
                    mape = performance_metrics(df_cv, metrics=['mape'])['mape'].mean()
                    if mape < best_mape:
                        best_mape = mape
                        best_params = params

            except Exception:
                continue

        if best_params:
            best_prophet_params[c] = best_params
            print(f"→ Best MAPE for Cluster {c}: {best_mape:.2%} | Params: {best_params}")
        else:
            print(f"→ No valid model found for Cluster {c}.")

    return best_prophet_params

"""## Step 30: Generate and Visualise Prophet Forecasts (tuned)

What it does:

- Trains Prophet with the tuned parameters on the **training period only** (2020–2023) and forecasts the 2024 test year.
- Visualises forecasts and components against actual test data to assess tuning impact.

Why it's important:

- Supports evaluating how tuning changed forecasts compared to the baseline.
- Fitting on the training period only keeps 2024 unseen, so the evaluation is a genuine test. (An earlier version of this notebook fitted the tuned models on the full 2020–2024 series, including the test year, which made the tuned results look far better than they really were.)

Output Analysis:

- Cluster 0: tuning selects a more flexible trend (changepoint_prior_scale 0.5, multiplicative seasonality), which follows the 2024 level more closely.

- Cluster 1: tuning selects changepoint_prior_scale 0.1 with multiplicative seasonality; the forecast is very similar to the baseline.

- Cluster 2: tuning selects the default settings (0.05, additive), so the tuned forecast is identical to the baseline.

Conclusion:

- Tuning helps where the pattern is regular but the level shifts (Cluster 0), and makes little difference elsewhere.
"""

def generate_tuned_prophet_forecasts(prophet_data: Dict[int, pd.DataFrame],ts_test: Dict[int, pd.Series],best_prophet_params: Dict[int, Optional[Dict[str, Any]]]
) -> Dict[int, pd.Series]:
    """
    Generates and visualises forecasts using tuned Prophet models for each cluster.

    Steps
    ------
    - Initialises an empty dictionary for tuned Prophet forecasts.
    - Iterates through each cluster and its best parameters, prints parameter message.
    - Checks availability of parameters, Prophet data, and test data; skips if missing.
    - Retrieves training data, determines forecast length, and initialises Prophet model with yearly seasonality and best parameters.
    - Fits model, creates future dataframe, generates forecast, and extracts 'yhat' for test period.
    - Stores forecast in prophet_forecasts_tuned, plots forecast and components with titles and labels.
    - Displays plots and returns the dictionary of tuned forecasts.

    Parameters
    ----------
    prophet_data : dict
        Dictionary of Prophet-formatted dataframes per cluster (only rows before the test period are used for fitting).
    ts_test : dict
        Dictionary of actual test time series per cluster.
    best_prophet_params : dict
        Dictionary of best hyperparameters per cluster.

    Returns
    -------
    dict
        Dictionary of forecasted values for the test period per cluster.
    """
    prophet_forecasts_tuned = {}

    for c, params in best_prophet_params.items():
        print(f"\nCluster {c} → Using params: {params}")

        if params is None or c not in prophet_data or c not in ts_test:
            print(f"Skipping Cluster {c}: Missing data or parameters.")
            continue

        # Fit on the training period only (data before the first test month), so the
        # 2024 test year stays unseen and the evaluation is a genuine out-of-sample test.
        test_start = ts_test[c].index.min()
        df_p = prophet_data[c]
        df_p = df_p[df_p['ds'] < test_start]
        periods = len(ts_test[c])

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            **params
        )
        model.fit(df_p)

        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast = model.predict(future)

        prophet_forecasts_tuned[c] = forecast.set_index('ds')['yhat'].loc[ts_test[c].index]

        model.plot(forecast)
        plt.title(f"Tuned Prophet Forecast – Cluster {c}")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.show()

        model.plot_components(forecast)
        plt.suptitle(f"Tuned Components – Cluster {c}", y=1.02)
        plt.tight_layout()
        plt.show()

    return prophet_forecasts_tuned

"""## Step 31: Evaluate the Accuracy of the Tuned Forecasts

What It Does:

- Evaluates the accuracy of the tuned Prophet forecasts using the same metrics as in the previous step, and compares them to the baseline Prophet results.

Why It's Important:

- This comparison helps assess the impact of tuning and identify which version of Prophet performs better for each cluster before benchmarking against SARIMAX.

Output Analysis:

- Cluster 0: MAPE improved from 5.0% (baseline) to 4.05% (tuned), the best result for this cluster.

- Cluster 1: MAPE 5.04%, essentially unchanged from the baseline (4.96%).

- Cluster 2: MAPE 4.53%, identical to the baseline because tuning selected the default settings.

Conclusion:

- Tuning gives a real but modest improvement for Cluster 0 only. The gains are smaller than in the first version of this notebook because the tuned models are now evaluated on genuinely unseen data.
"""

def evaluate_tuned_prophet_forecasts(
    ts_test: Dict[int, pd.Series],
    prophet_forecasts_tuned: Dict[int, pd.Series]
) -> Dict[int, Dict[str, float]]:
    """
    Evaluates the accuracy of tuned Prophet forecasts using standard error metrics.

    Steps
    ------
    - Initialise dictionary for metrics.
    - Iterate clusters with tuned forecasts.
    - Get and align actuals and tuned predictions.
    - Calculate MAE, RMSE, and MAPE.
    - Store and print results per cluster.
    - Return metrics dictionary.

    Parameters
    ----------
    ts_test : dict
        Dictionary of actual test time series per cluster.
    prophet_forecasts_tuned : dict
        Dictionary of tuned forecasted values per cluster.

    Returns
    -------
    dict
        Dictionary containing MAE, RMSE, and MAPE for each cluster.
    """
    prophet_metrics_tuned = {}

    print("\nEvaluating Tuned Prophet Forecasts")

    for c in sorted(ts_test):
        if c not in prophet_forecasts_tuned:
            continue

        actual = ts_test[c]
        predicted = prophet_forecasts_tuned[c]

        actual, predicted = actual.align(predicted, join='inner')

        if actual.empty:
            continue

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual.replace(0, np.nan))) * 100

        prophet_metrics_tuned[c] = {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

        print(f"Cluster {c}: MAE = {mae:.2f}, RMSE = {rmse:.2f}, MAPE = {mape:.2f}%")

    return prophet_metrics_tuned

"""## Step 32: Compare with SARIMAX metrics

What it does:

- Compares test-set MAPE and MAE for SARIMAX, baseline Prophet and tuned Prophet for each cluster, and identifies the most accurate model.

Why It's Important:

- It helps identify which model performs better for each cluster, supporting data-driven model selection based on accuracy.

Output analysis:

| Cluster | SARIMAX | Prophet (baseline) | Prophet (tuned) | Best |
|---|---|---|---|---|
| 0 – Fast-Moving Chemicals | 4.87% | 5.00% | **4.05%** | Prophet (tuned) |
| 1 – Slow-Moving Heavy Electronics | **4.61%** | 4.96% | 5.04% | SARIMAX |
| 2 – High-Value Pharmaceuticals | 8.42% | **4.53%** | 4.53% | Prophet |

Conclusion:

- No single model wins everywhere. Prophet is the most accurate for two of the three clusters, and is clearly better where the trend changed direction (Cluster 2). SARIMAX is slightly more accurate for the regular seasonal pattern of Cluster 1.
- All models achieve MAPE below 10% on the 2024 test year, comfortably within the 15% target.
"""

def compare_forecast_metrics(
    prophet_metrics: Dict[int, Dict[str, float]],
    prophet_metrics_tuned: Dict[int, Dict[str, float]],
    evaluation_results: Dict[int, Dict[str, float]],
    cluster_name_mapping: Optional[Dict[int, str]] = None
) -> pd.DataFrame:
    """
    Compares test-set MAPE for baseline Prophet, tuned Prophet and SARIMAX, and identifies the most accurate model per cluster.

    Steps
    ------
    - Builds a table of MAPE (and MAE) for each model and cluster.
    - Identifies the model with the lowest MAPE for each cluster.
    - Displays and returns the comparison table.

    Parameters
    ----------
    prophet_metrics : dict
        Baseline Prophet metrics per cluster.
    prophet_metrics_tuned : dict
        Tuned Prophet metrics per cluster.
    evaluation_results : dict
        SARIMAX metrics per cluster.
    cluster_name_mapping : dict, optional
        Descriptive cluster names.

    Returns
    -------
    pd.DataFrame
        Comparison table with the best model per cluster.
    """
    rows = []
    for c in sorted(evaluation_results):
        row = {
            'Cluster': cluster_name_mapping.get(c, c) if cluster_name_mapping else c,
            'MAPE_SARIMAX': evaluation_results[c]['MAPE'],
            'MAPE_Prophet_Baseline': prophet_metrics.get(c, {}).get('MAPE', np.nan),
            'MAPE_Prophet_Tuned': prophet_metrics_tuned.get(c, {}).get('MAPE', np.nan),
            'MAE_SARIMAX': evaluation_results[c]['MAE'],
            'MAE_Prophet_Baseline': prophet_metrics.get(c, {}).get('MAE', np.nan),
            'MAE_Prophet_Tuned': prophet_metrics_tuned.get(c, {}).get('MAE', np.nan),
        }
        mapes = {'SARIMAX': row['MAPE_SARIMAX'], 'Prophet (baseline)': row['MAPE_Prophet_Baseline'], 'Prophet (tuned)': row['MAPE_Prophet_Tuned']}
        row['Best_Model'] = min(mapes, key=lambda k: mapes[k] if pd.notna(mapes[k]) else np.inf)
        rows.append(row)

    comp = pd.DataFrame(rows).set_index('Cluster').round(2)
    print("Test-set comparison (2024), lower is better:")
    display(comp)
    return comp

"""## Step 33: Plot Actual vs Prophet Prediction

What it does:

- Visualises actual vs. tuned Prophet forecasted monthly volumes per cluster.

Why It's Important:

- Helps validate the model's performance by visually comparing forecasts to actual data.

Output Analysis:

- Cluster 0: close alignment through the year, including the December drop.

- Cluster 1: follows the seasonal direction, with gaps at individual peaks and troughs.

- Cluster 2: follows the shape of the actuals but sits slightly below them.

Conclusion:

- Combined with the metrics, this supports a per-cluster model choice: Prophet for Clusters 0 and 2, SARIMAX for Cluster 1.
"""

def plot_actual_vs_tuned_forecasts(
    ts_test: Dict[int, pd.Series],
    prophet_forecasts_tuned: Dict[int, pd.Series]
) -> None:
    """
    Plots actual vs tuned Prophet forecast for each cluster in the test set.

    Steps
    ------
    - Iterate clusters with test data.
    - Check for tuned Prophet forecast.
    - Get actuals and predictions.
    - Plot actual and predicted values.
    - Display the plot.
    - Skip if forecast is missing.

    Parameters
    ----------
    ts_test : dict
        Dictionary of actual test time series per cluster.
    prophet_forecasts_tuned : dict
        Dictionary of tuned Prophet forecasted values per cluster.
    """
    for c in sorted(ts_test.keys()):
        actual = ts_test[c]

        if c in prophet_forecasts_tuned and prophet_forecasts_tuned[c] is not None:
            pred = prophet_forecasts_tuned[c].loc[actual.index]
            plt.figure(figsize=(10,5))
            plt.plot(actual.index, actual, label='Actual', linewidth=2)
            plt.plot(pred.index,     pred,   label='Tuned Prophet', linestyle='--', linewidth=2)
            plt.title(f'Tuned Prophet vs Actual – Cluster {c}')
            plt.xlabel('Date'); plt.ylabel('Monthly Volume')
            plt.legend(); plt.grid(True); plt.tight_layout()
            plt.show()
        else:
            print(f"Skipping Cluster {c} - Tuned Prophet forecast not available.")