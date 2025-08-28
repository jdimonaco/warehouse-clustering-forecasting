
import statsmodels.api as sm
from typing import Tuple
from typing import Dict
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
from itertools import product
from IPython.display import display
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Too few observations to estimate starting parameters")


"""# Forecasting SARIMAX

Why Use:

- SARIMAX is chosen for forecasting monthly inbound volumes because it models trends and seasonal patterns present in the data.
- Uses SARIMAX to forecast future monthly volumes for each cluster.
- Captures trend and seasonality.

Why it's important:

- Supports inventory planning and resource allocation at cluster level.

- Accounts for seasonal variation common in warehouse operations.

## Step 13: Forecasting Data Preparation

What it does:

- Prepares monthly total volume time series for each cluster.

- Groups by cluster, resamples to month-end (ME), and sums the value column into monthly_volume.

Sets Received date as the time index.

Why it's important:

- Structures data for time-series models to capture cluster-specific trends and seasonality.
"""

def prepare_time_series_by_cluster(df: pd.DataFrame, date_col: str = 'Received date', cluster_col: str = 'cluster',
                                   value_col: str = 'avg_volume') -> pd.Series:
    """
    Prepares a time series of monthly volume per cluster.

    Steps
    ------
    - Selects relevant columns and sets date as index.
    - Groups by cluster and resamples to monthly frequency.
    - Calculates the sum of the value column for each month.
    - Returns the multi-indexed time series.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing date, cluster, and volume information.
    date_col : str
        Name of the column containing date values.
    cluster_col : str
        Name of the column containing cluster labels.
    value_col : str
        Name of the column containing volume values.

    Returns
    -------
    pd.Series
        A time series indexed by cluster and month, showing total monthly volume.
    """
    df_ts = df[[date_col, cluster_col, value_col]].copy()
    df_ts = df_ts.set_index(date_col)

    ts = (df_ts.groupby(cluster_col)[value_col]
          .resample('ME')
          .sum()
          .rename('monthly_volume'))

    print("Monthly volume per cluster:")
    display(ts.head())
    print("Index type:", ts.index.dtype)
    print("Sample index values:", ts.index[:5])

    return ts


"""## Step 14: Stationarity Check (ADF)

What it does:

-  Checks whether each cluster’s monthly volume data is stationary — meaning its statistical properties (like mean and variance) remain constant over time.

Why it's important:
- SARIMAX (like ARIMA) works best when the data is stationary. If the data isn’t stable, the model’s forecasts can become unreliable. This test helps make sure the data is ready for accurate forecasting.
- While the ADF test may indicate non-stationarity, SARIMAX models can handle this through differencing (the 'I' component), making them suitable for such data.
"""

def test_stationarity_by_cluster(ts: pd.Series) -> None:
    """
    Performs the Augmented Dickey-Fuller (ADF) test on each cluster's monthly volume series
    to assess stationarity.

    Steps
    ------
    - Identifies the unique cluster IDs from the time series index.
    - Iterates through each cluster ID.
    - Selects the time series data for the current cluster.
    - Performs the Augmented Dickey-Fuller test on the cluster's time series (dropping NaNs if any).
    - Prints the ADF statistic, p-value, and critical values.
    - Prints a conclusion based on the p-value (stationary if p-value <= 0.05).

    Parameters
    ----------
    ts : pd.Series
        A multi-indexed time series with 'cluster' and datetime index, containing monthly volumes.

    Returns
    -------
    None
        Prints the ADF p-value and conclusion for each cluster.
    """
    clusters = sorted(ts.index.get_level_values('cluster').unique())

    for c in clusters:
        series = ts.loc[c]
        pval = adfuller(series.dropna())[1]
        conclusion = 'Stationary (reject H₀)' if pval <= 0.05 else 'Non-Stationary (fail to reject H₀)'
        print(f"Cluster {c} ADF p-value: {pval:.4f} → {conclusion}")

"""## Step 15: Time Series Decomposition

What it does:

- Breaks each cluster’s monthly volume into trend, seasonal, and residual components using an additive model.

Why it’s important:

- Helps visualise long-term patterns and recurring cycles, confirming seasonality and supporting seasonal models like SARIMAX.

Output Analysis:

- Cluster 0: Lower volume, weak trend, noisy residuals. Seasonality is irregular. Still suitable for SARIMAX if differencing stabilises the series.

- luster 1: Clear trend and strong, repeatable seasonality. Well suited to SARIMAX.

- Cluster 2: Visible trend and noticeable seasonality; residuals show some irregularity. Also suitable for SARIMAX.
"""

def decompose_cluster_time_series(ts: pd.Series, cluster_col: str = 'cluster') -> None:
    """
    Decomposes and visualises the seasonal components of a monthly volume time series for each cluster.

    Steps
    ------
    - Identifies unique cluster IDs from the time series index.
    - Iterates through each cluster, selects its time series, and ensures frequency is set.
    - Performs seasonal decomposition using the additive model.
    - Plots and displays decomposition components: observed, trend, seasonal, and residual.


    Parameters
    ----------
    ts : pd.Series
        A multi-indexed time series with 'cluster' and datetime index, containing monthly volumes.
    cluster_col : str, optional
        The name of the cluster level in the index (default is 'cluster').

    Returns
    -------
    None
        Displays decomposition plots for each cluster.
    """
    clusters = sorted(ts.index.get_level_values(cluster_col).unique())

    for c in clusters:
        series = ts.loc[c]
        series.index.freq = series.index.inferred_freq
        decomposition = seasonal_decompose(series, model='additive')
        fig = decomposition.plot()
        fig.suptitle(f"Cluster {c} Decomposition", y=1.02)
        plt.tight_layout()
        plt.show()

"""## Step 16: ACF & PACF

What it does:

- ACF (Autocorrelation Function): Correlation between the current value and past values at different lags.

- PACF (Partial Autocorrelation Function): Direct relationship with a specific lag after removing intermediate effects.

Why it’s important:

- Helps choose SARIMAX structure:

  - AR terms (how many past months to include),

  - Differencing (I terms) to stabilise the series,

  - Seasonal AR/MA terms if seasonal patterns are present.

Output Analysis:

- Lag 1: Strong correlation across all clusters (current volume is strongly related to last month’s).

- Lag 12: Clear seasonal spike in Cluster 1, confirming annual seasonality.

- Clusters 0 and 2: No significant autocorrelation at lag 12; volumes are driven more by recent months than by same month last year.
"""

def plot_acf_pacf_by_cluster(ts: pd.Series, lags: int = 12, cluster_col: str = 'cluster') -> None:
    """
    Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
    for each cluster's monthly volume time series.

    Steps
    ------
    - Identifies the unique cluster IDs from the time series index.
    - Iterates through each cluster ID.
    - Selects the time series data for the current cluster.
    - Generates and displays the ACF plot for the current cluster's series up to the specified lags.
    - Generates and displays the PACF plot for the current cluster's series up to the specified lags.

    Parameters
    ----------
    ts : pd.Series
        A multi-indexed time series with 'cluster' and datetime index, containing monthly volumes.
    lags : int
        Number of lags to include in the ACF and PACF plots.
    cluster_col : str
        The name of the cluster level in the index.

    Returns
    -------
    None
        Displays ACF and PACF plots for each cluster.
    """
    clusters = sorted(ts.index.get_level_values(cluster_col).unique())

    for c in clusters:
        series = ts.loc[c]
        plot_acf(series, lags=lags)
        plt.title(f"ACF – Cluster {c}")
        plt.tight_layout()
        plt.show()

        plot_pacf(series, lags=lags)
        plt.title(f"PACF – Cluster {c}")
        plt.tight_layout()
        plt.show()

"""## Step 17: Train/Test Split

What it does:

- Each cluster’s monthly volume data is split into:
- Training data: January 2020 to December 2023
- Testing data: January 2024 to December 2024
- This structure is applied consistently across all clusters.

Why it's important:

- Using a fixed 12-month test period allows to:
  - Compare model performance across clusters
  - Evaluate how well each model handles recent unseen data
"""

def split_cluster_time_series(ts: pd.Series, min_months: int = 24, test_size: int = 12) -> Tuple[Dict[int, pd.Series], Dict[int, pd.Series]]:
    """
    Splits each cluster's monthly time series into training and test sets.

    Steps
    ------
    - Initialises empty dictionaries for training and test series per cluster.
    - Iterates through each unique cluster ID, selects and sorts its time series.
    - Checks if the series meets the minimum required months; if valid, splits into training and test sets.
    - Stores each set in its respective dictionary, prints size and date range.
    - Returns the training and test dictionaries.

    Parameters
    ----------
    ts : pd.Series
        A multi-indexed time series with 'cluster' and datetime index, containing monthly volumes.
    min_months : int
        Minimum number of months required to perform a split.
    test_size : int
        Number of months to allocate to the test set.

    Returns
    -------
    Tuple[Dict[int, pd.Series], Dict[int, pd.Series]]
        Two dictionaries containing training and test series for each cluster.
    """
    ts_cluster = {}
    ts_train = {}
    ts_test = {}

    for c in sorted(ts.index.get_level_values('cluster').unique()):
        series = ts.loc[c].sort_index()
        ts_cluster[c] = series

        if len(series) < min_months:
            print(f" Skipping Cluster {c} – only {len(series)} months of data")
            continue

        ts_train[c] = series.iloc[:-test_size]
        ts_test[c] = series.iloc[-test_size:]

        print(f" Cluster {c}:")
        print(f"   Training set: {len(ts_train[c])} months "
              f"({ts_train[c].index.min().strftime('%Y-%m')} to {ts_train[c].index.max().strftime('%Y-%m')})")
        print(f"   Test set:     {len(ts_test[c])} months "
              f"({ts_test[c].index.min().strftime('%Y-%m')} to {ts_test[c].index.max().strftime('%Y-%m')})")
        print("-" * 40)

    return ts_train, ts_test

"""## Step 18: Find Optimal Parameters

What it does:

- Performs a grid search over different combinations of SARIMAX parameters:

  - Non-seasonal: (p, d, q)

  - Seasonal: (P, D, Q, 12)

- Splits the training data for each cluster into an inner-training set and a validation set.

- Fits models on the inner-training set and selects the combination with the lowest Mean Absolute Error (MAE) on the validation set.

Why it’s important:

- Helps select the best model parameters for each cluster to improve forecast accuracy on future data while avoiding overfitting.
"""

def tune_sarimax_per_cluster(ts_train: Dict[int, pd.Series],seasonal_period: int = 12) -> Dict[int, Dict[str, object]]:

    """
    Performs grid search to find the best SARIMAX hyperparameters for each cluster's time series
    using an internal validation split from the training data.

    For each cluster:
    - Splits the training series into an inner-training segment and a validation segment.
    - Fits SARIMAX models on the inner-training segment across a parameter grid.
    - Evaluates mean absolute error (MAE) on the validation segment.
    - Selects and stores the parameters with the lowest validation MAE.

    Parameters
    ----------
    ts_train : dict
        Dictionary mapping cluster IDs to their training time series
    seasonal_period : int, optional
        Seasonal period for SARIMAX.

    Returns
    -------
    dict
        Mapping of cluster IDs to a dictionary containing:
        - 'order': best (p, d, q)
        - 'seasonal_order': best (P, D, Q, s)
        - 'mae': validation MAE for the selected parameters
    """
    p = d = q = range(0, 2)
    P = D = Q = range(0, 2)
    s = seasonal_period

    best_params: Dict[int, Dict[str, object]] = {}

    for c, series in ts_train.items():
        y = pd.Series(series).dropna()

        val_h = int(max(1, min(s, np.ceil(len(y) * 0.1))))
        if len(y) <= val_h + 3:
            val_h = max(1, min(val_h, len(y) // 3))

        y_train_inner = y.iloc[:-val_h]
        y_val_inner = y.iloc[-val_h:]

        best_mae = float("inf")
        best_order = None
        best_seasonal_order = None

        for order in product(p, d, q):
            for seasonal in product(P, D, Q):
                seasonal_order = (seasonal[0], seasonal[1], seasonal[2], s)
                try:
                    model = SARIMAX(
                        y_train_inner,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    )
                    fit_res = model.fit(disp=False)

                    pred = fit_res.get_forecast(steps=val_h).predicted_mean
                    mae = mean_absolute_error(y_val_inner, pred)

                    if mae < best_mae:
                        best_mae = mae
                        best_order = order
                        best_seasonal_order = seasonal_order

                except Exception:
                    continue

        best_params[c] = {
            'order': best_order,
            'seasonal_order': best_seasonal_order,
            'mae': best_mae if np.isfinite(best_mae) else None
        }

        print(f"Cluster {c} → Best MAE: {best_mae:.4f}")
        print(f"Best order: {best_order}, Seasonal: {best_seasonal_order}")

    return best_params

"""## Step 19: Train SARIMAX Models

What it does:
- Fits a SARIMAX model to each cluster’s training data using the optimal parameters identified during hyperparameter tuning.

Why it's important:
- This step enables the model to learn from historical patterns in the data, preparing it to make accurate forecasts even when the data shows some fluctuations over time.
"""

def retrain_and_forecast_sarimax(ts_train: Dict[int, pd.Series], best_params: Dict[int, Dict[str, object]],
                                 n_steps_forecast: int = 12) -> Tuple[Dict[int, object], Dict[int, pd.Series]]:
    """
    Retrains SARIMAX models using the best parameters for each cluster and forecasts the test set.

    Steps
    ------
    - Initialises empty dictionaries to store final fitted models and forecasts.
    - Iterates through each cluster with tuned parameters, prints training start message.
    - Initialises and fit a SARIMAX model using training data.
    - Initialises and fits SARIMAX model using training data, stores in 'final_models'.
    - Generates forecast for upcoming periods, stores in 'cluster_forecasts'.
    - Prints completion message and returns models and forecasts.


    Parameters
    ----------
    ts_train : dict
        Dictionary of training time series per cluster.
    best_params : dict
        Dictionary containing best SARIMAX parameters and MAE per cluster.
    n_steps_forecast : int
        Number of months to forecast (default is 12).

    Returns
    -------
    Tuple[Dict[int, object], Dict[int, pd.Series]]
        A tuple containing:
        - final_models: Fitted SARIMAX models per cluster.
        - cluster_forecasts: Forecasted values per cluster.
    """
    final_models = {}
    cluster_forecasts = {}

    for c in sorted(best_params.keys()):
        print(f"\nTraining final SARIMAX model for Cluster {c}...")

        model = SARIMAX(
            ts_train[c],
            order=best_params[c]['order'],
            seasonal_order=best_params[c]['seasonal_order'],
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        final_models[c] = model.fit(disp=False)
        forecast = final_models[c].get_forecast(steps=n_steps_forecast).predicted_mean

        cluster_forecasts[c] = forecast

        print(f"Forecast completed for Cluster {c}")

    return final_models, cluster_forecasts

"""## Step 20: Residual Diagnostic

What it does:

- Checks the residuals (errors) from the trained models to see if they are random and normally distributed, using plots and the Ljung–Box statistical test.

- A random pattern means the errors are scattered and unpredictable, which is ideal, as it shows the model has done a good job.

Why it's important:

- Confirms whether the model has captured all meaningful patterns in the data and whether its assumptions are met.

- Random residuals indicate a well-fitting mode

Output Analysis:

- Cluster 0: Ljung–Box p-value (lag=12) = 0.996 → residuals appear random. The model adequately captures the patterns for this cluster.

- Cluster 1: Ljung–Box p-value (lag=12) = 0.642 → residuals appear random. The model fit is appropriate for this cluster.

- Cluster 2: Ljung–Box p-value (lag=12) = 0.729 → residuals appear random. The model provides a good fit for this cluster.
"""

def analyse_residuals(final_models: Dict[int, object]) -> None:
    """
    Performs residual diagnostics for each fitted SARIMAX model per cluster.

    Steps
    ------
    - Iterates through each fitted SARIMAX model in the input dictionary.
    - For each model, checks availability and prints a diagnostics header.
    - Extracts residuals, drops NaNs, and creates a figure with three subplots: residuals over time, ACF, and Q-Q plot.
    - Displays the figure and performs the Ljung-Box test for autocorrelation.
    - Prints p-value and conclusion on residual randomness.
    - Handles errors and prints a separator line.
    - Skips clusters with no available model.


    Parameters
    ----------
    final_models : dict
        Dictionary of fitted SARIMAX models per cluster.

    Returns
    -------
    None
        Displays residual plots and Ljung-Box test results for each cluster.
    """
    for c, fitted_model in final_models.items():
        if fitted_model:
            print(f"\nResidual Diagnostics for Cluster {c}:")
            try:
                resid = fitted_model.resid.dropna()

                fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                # Residuals over time
                axs[0].plot(resid)
                axs[0].set_title(f'Cluster {c} Residuals Over Time')
                axs[0].set_xlabel('Date')
                axs[0].set_ylabel('Residual')

                # ACF of residuals
                plot_acf(resid, ax=axs[1], lags=12)
                axs[1].set_title(f'ACF of Residuals – Cluster {c}')

                # Q-Q plot for normality
                sm.qqplot(resid, line='s', ax=axs[2])
                axs[2].set_title(f'Q-Q Plot – Cluster {c}')

                plt.tight_layout()
                plt.show()

                # Ljung-Box test
                lb_result = acorr_ljungbox(resid, lags=[12], return_df=True)
                lb_p = lb_result['lb_pvalue'].iloc[0]

                print(f"  Ljung-Box p-value (lag=12): {lb_p:.3f}")
                if lb_p > 0.05:
                    print("  Residuals appear random.")
                else:
                    print("  Residuals show autocorrelation – model may need refinement.")

            except Exception as e:
                print(f"  Error with diagnostics for Cluster {c}: {e}")

            print("-" * 40)
        else:
            print(f"Skipping Cluster {c} — no model available.")
            print("-" * 40)

"""## Step 21: Make Predictions on Test Data

What it does:

- Generates 12-month forecasts (January to December 2024) for each cluster using the trained SARIMAX models and their best-fit parameters.

Why it's important:

- Forecasting on this unseen test period is crucial for evaluating the model's real-world performance and ensuring it can accurately predict future inbound volumes.
"""

def generate_cluster_forecasts(final_models: Dict[int, Any], ts_test: Dict[int, pd.Series]) -> Dict[int, pd.Series]:
    """
    Generates forecast predictions for the test period using trained SARIMAX models.

    Steps
    ------
    - Initialises an empty dictionary to store cluster forecasts.
    - Iterates through each cluster in the test data, prints forecast generation message.
    - Checks for a fitted model in final_models; if available, determines forecast steps.
    - Generates forecast using get_forecast, aligns index with actual test data.
    - Stores aligned forecast in cluster_forecasts, prints completion message and date range.
    - Displays first 5 rows of the forecast, handles any exceptions.
    - Returns the dictionary of forecasted values for the test period per cluster.

    Parameters
    ----------
    final_models : dict
        Dictionary of fitted SARIMAX models for each cluster.
    ts_test : dict
        Dictionary of actual test time series per cluster.

    Returns
    -------
    dict
        Dictionary of forecasted values for the test period per cluster.
    """
    cluster_forecasts = {}

    for c in sorted(ts_test.keys()):
        print(f"Generating forecasts for Cluster {c}...")
        try:
            n_steps = len(ts_test[c])
            if c not in final_models or final_models[c] is None:
                print(f" Skipping Cluster {c}: No fitted model available.")
                print("-" * 30)
                continue

            fc = final_models[c].get_forecast(steps=n_steps).predicted_mean

            fc.index = ts_test[c].index
            cluster_forecasts[c] = fc

            start = fc.index.min().strftime('%Y-%m')
            end   = fc.index.max().strftime('%Y-%m')
            print(f"Forecast generated for Cluster {c} from {start} to {end}.")
            print("-" * 30)

            print(f"\nCluster {c} Forecasts (first 5 rows):")
            print(fc.head(), "\n")

        except Exception as e:
            print(f" Forecast failed for Cluster {c}: {e}")
            print("-" * 30)

    return cluster_forecasts

"""## Step 22: Comparing Actual vs Predicted values

What it does:

- Plots training, actual test, and forecast series together.

Why it's important:

- Provides a way to compare the performance of the model across different clusters and identify areas where forecasting accuracy needs improvement.

Output Analysis:

- Cluster 0: The forecast appears as a relatively straight line on the test data, indicating it did not capture the fluctuations or trend in the actual volumes for this period.

- Cluster 1: Forecasts generally follow the main trend and seasonal shape, but there are visible differences from actuals in specific months, indicating some missed short-term variability.

- Cluster 2: Forecasts follow the general trend but show larger gaps; some monthly variation is under/over-estimated.

Conclusion:

- The plots indicate that while SARIMAX captures broader patterns for some clusters, it struggles with the variability and specific trend for Cluster 0 on the test data. Evaluate model performance on test data will validate these visual observations with the quantitative metrics.
"""

def plot_actual_vs_predicted(actuals: Dict[int, pd.Series],predictions: Dict[int, pd.Series]) -> None:
    """
    Plots actual vs predicted monthly volume for each cluster.

    Steps
    ------
    - Iterates through each cluster present in the predictions dictionary.
    - Retrieves the actual and predicted time series for the current cluster.
    - Creates a new figure for plotting.
    - Plots the actual values against their index.
    - Plots the predicted values against their index.
    - Sets the title, labels, and legend for the plot.
    - Displays the plot for the current cluster.

    Parameters
    ----------
    actuals : dict
        Dictionary of actual time series per cluster.
    predictions : dict
        Dictionary of predicted time series per cluster.

    Returns
    -------
    None
        Displays line plots comparing actual and predicted values.
    """

    for c in sorted(predictions.keys()):
        actual = actuals[c]
        predicted = predictions[c]

        plt.figure(figsize=(10, 5))
        plt.plot(actual.index, actual.values, label='Actual', marker='o')
        plt.plot(predicted.index, predicted.values, label='Predicted', marker='x')
        plt.title(f'Cluster {c} – Actual vs Predicted Monthly Volume')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

"""## Step 23: Evaluate Model Performance on Test Data

What it does:

- Calculates MAE, RMSE, and MAPE for each cluster’s forecasts.

Why it's important:

- These numbers show how close the forecasts were to the actual values:
MAE and RMSE measure the size of the errors.
- MAPE shows the error as a percentage — useful for comparing across clusters.

Output Analysis:

- Cluster 0: While the forecast visually appeared as a straight line, missing fluctuations in actual volumes, the MAE was 9.75% of the test mean, indicating moderate accuracy over the test period.

- Cluster 1: The forecast captured the overall trend and seasonal shape well, with some short-term deviations. The MAE was 4.48% of the test mean, reflecting strong performance and a good fit for this cluster's patterns.

- Cluster 2: The forecast followed the general trend but showed noticeable gaps in certain months. The MAE was 4.49% of the test mean, demonstrating reliable forecasting for this cluster.

Conclusion:

- All clusters achieved MAPE below 15%, confirming the project hypothesis. SARIMAX performed reliably across cluster types. Prophet will be evaluated next to identify the optimal forecasting model.
"""

def evaluate_forecast_accuracy(ts_test: Dict[int, pd.Series],cluster_forecasts: Dict[int, pd.Series]) -> Dict[int, Dict[str, float]]:
    """
    Evaluates forecast accuracy for each cluster using MAE, RMSE, and MAPE.

    Steps
    ------
    - Initialises empty dictionary to store evaluation results.
    - Iterates through each cluster for which forecasts are available.
    - Retrieves actual and predicted time series for the current cluster.
    - Aligns actual and predicted data.
    - Calculates evaluation metrics (MAE, RMSE, MAPE, Test Mean, MAE %).
    - Stores the calculated metrics in the results dictionary.
    - Prints evaluation results for each cluster.
    - Returns the dictionary containing the evaluation results per cluster.

    Parameters
    ----------
    ts_test : dict
        Dictionary of actual test time series per cluster.
    cluster_forecasts : dict
        Dictionary of predicted time series per cluster.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics per cluster.
    """
    evaluation_results = {}

    for c in sorted(cluster_forecasts.keys()):
        actual = ts_test[c]
        predicted = cluster_forecasts[c]


        actual_aligned, predicted_aligned = actual.align(predicted, join='inner')

        if actual_aligned.empty:
             evaluation_results[c] = {'MAE': np.nan, 'RMSE': np.nan, 'MAPE': np.nan, 'Test_Mean': np.nan, 'MAE_%': np.nan}
             print(f"\n Cluster {c} Evaluation: No data for evaluation after alignment.")
             continue

        mae = mean_absolute_error(actual_aligned, predicted_aligned)
        rmse = np.sqrt(mean_squared_error(actual_aligned, predicted_aligned))

        mape_series = np.abs((actual_aligned - predicted_aligned) / actual_aligned.replace(0, np.nan))
        mape = mape_series.dropna().mean() * 100 if not mape_series.dropna().empty else np.nan

        test_mean = actual_aligned.mean()
        mae_percent = (mae / test_mean) * 100 if test_mean != 0 else np.nan

        evaluation_results[c] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Test_Mean': test_mean,
            'MAE_%': mae_percent
        }

        print(f"\n Cluster {c} Evaluation:")
        print(f"  MAE:  {evaluation_results[c]['MAE']:.2f}")
        print(f"  RMSE: {evaluation_results[c]['RMSE']:.2f}")
        print(f"  MAPE: {evaluation_results[c]['MAPE']:.2f}%")
        print(f"  MAE as % of Test Mean: {evaluation_results[c]['MAE_%']:.2f}%")

    return evaluation_results

"""## Step 24:  Visualise Forecasts on Test Data

What it does:

- Plots the training, actual test, and forecast series together for each cluster.

Why it’s important:

- Gives a visual check of how well the model follows the real pattern, seasonality, and trend—by seeing how close the forecast line is to the actual line.

Output Analysis:

- Cluster 0: The forecast starts to drift away from the actual data around mid-2024, missing several ups and downs. While earlier results showed moderate accuracy (MAE 9.75%), the recent forecast looks less reliable.

- Cluster 1: The forecast stays closer to the actual values than the other clusters, with smaller differences. This matches the strong MAE (4.48%) and confirms it has the best overall fit.

- Cluster 2: The forecast shows bigger and more frequent gaps from the actual data, especially after mid-2024. This supports the earlier finding that Cluster 2 had the weakest fit, even though its MAE (4.49%) was relatively low.

Conclusion:

- Forecast accuracy varies across clusters. Cluster 1 fits the actual data best. Cluster 0 captures the overall trend but misses some month-to-month changes. Cluster 2 shows the biggest differences from actuals. Further improvements should focus on Cluster 2, then Cluster 0, to boost overall performance.
"""

def visualise_forecasts_on_test_data(ts_train: Dict[int, pd.Series],ts_test: Dict[int, pd.Series], cluster_forecasts: Dict[int, pd.Series]) -> None:
    """
    Visualises training data, actual test data, and SARIMAX forecasts for each cluster.

    Steps
    ------
    - Iterates through each cluster with training data.
    - Checks if test data and forecasts are available.
    - Retrieves train, actual test, and forecast series.
    - Plots data.
    - Skips if data is missing.

    Parameters
    ----------
    ts_train : dict
        Dictionary of training time series per cluster.
    ts_test : dict
        Dictionary of actual test time series per cluster.
    cluster_forecasts : dict
        Dictionary of forecasted time series per cluster.

    Returns
    -------
    None
        Displays line plots for each cluster.
    """
    for c in sorted(ts_train.keys()):
        if c in ts_test and c in cluster_forecasts and cluster_forecasts[c] is not None:
            train_series = ts_train[c]
            test_series = ts_test[c]
            forecast_series = cluster_forecasts[c]

            if not isinstance(forecast_series, pd.Series):
                print(f"Skipping Cluster {c} - forecast data is not in expected Series format.")
                continue

            plt.figure(figsize=(14, 6))

            plt.plot(train_series.index, train_series, label='Training Data', color='orange')

            plt.plot(test_series.index, test_series, label='Actual Test Data', color='blue')

            plt.plot(forecast_series.index, forecast_series, label='Forecast', color='green', linestyle='--')

            plt.title(f'Forecast vs. Actuals for Cluster {c}')
            plt.xlabel('Date')
            plt.ylabel('Monthly Volume')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print(f"Skipping Cluster {c} – missing test data or forecast.")
