
import pandas as pd
from IPython.display import display
from sklearn.cluster import KMeans

from preprocessing import (
    inspect_inventory_data,
    feature_engineering,
    perform_full_eda,
    encode_categorical_features,
    convert_boolean_columns,    
    winsorise_outliers,
)
from sentiment import ( 
    read_feedback_data,
    perform_sentiment_analysis,
    apply_inefficiency_categorisation,
    analyse_sentiment_by_category
)
from clustering import (
    scale_numeric_features,
    find_optimal_k,
    assign_clusters,
    calculate_cluster_means,
    rename_clusters,
    plot_clusters_pca,
    summarise_clusters
)
from sarimax import (
    prepare_time_series_by_cluster,
    test_stationarity_by_cluster,
    decompose_cluster_time_series,
    plot_acf_pacf_by_cluster,
    split_cluster_time_series,
    tune_sarimax_per_cluster,
    retrain_and_forecast_sarimax,
    analyse_residuals,
    plot_actual_vs_predicted,
    generate_cluster_forecasts,
    evaluate_forecast_accuracy,
    visualise_forecasts_on_test_data
)
from forecasting_utils import (
    prepare_prophet_data,
    train_prophet_models,
    generate_and_visualise_prophet_forecasts,
    evaluate_prophet_forecasts,
    tune_prophet_parameters,
    generate_tuned_prophet_forecasts,
    evaluate_tuned_prophet_forecasts,
    plot_actual_vs_tuned_forecasts,
    compare_forecast_metrics
)

def main_pipeline() -> dict:
    """
     Executes sentiment analysis first, then inventory clustering & forecasting.
    """
    # --- Feedback Sentiment Analysis ---

    # Step 0.1: Load feedback
    feedback_df = read_feedback_data('warehouse_feedback.csv')
    display(feedback_df.head(5))

    # Step 0.2: Perform sentiment analysis
    feedback_df = perform_sentiment_analysis(feedback_df)
    display(feedback_df[['product_id','feedback','neg_sentiment','neu_sentiment','pos_sentiment','compound_sentiment']].head(5))

    # Step 0.3: Apply inefficiency categorisation
    feedback_df = apply_inefficiency_categorisation(feedback_df)
    display(feedback_df[['product_id','feedback','inefficiency_category']].head(5))
    display(feedback_df['inefficiency_category'].value_counts().to_frame('count'))


    # Step 0.4: Aggregate sentiment by category
    sentiment_summary = analyse_sentiment_by_category(feedback_df)
    if isinstance(sentiment_summary, pd.Series):
      sentiment_summary = sentiment_summary.rename('compound_sentiment').reset_index()
    display(sentiment_summary)

    # Add the detailed sentiment implications output
    print("\nImplications of Sentiment Scores:")
    for index, row in sentiment_summary.iterrows():
        category = row['inefficiency_category']
        sentiment_score = row['compound_sentiment']
        sentiment_label = "Positive" if sentiment_score > 0.05 else ("Negative" if sentiment_score < -0.05 else "Neutral")
        print(f"- {category}: Average sentiment is {sentiment_label} ({sentiment_score:.4f}). This indicates a generally {sentiment_label.lower()} feeling among employees regarding {category}.")


     # --- Clustering & Forecasting ---

    # Step 1: Load and Inspect Inventory Data
    inventory_file_path = "priced_inventory_data (3).csv"
    df = inspect_inventory_data(inventory_file_path)
    print("\nSummary Statistics:")
    print(df.describe())

    # Step 2: Feature Engineering
    df_fe = feature_engineering(df)
    display(df_fe.head())

    # Step 3: Exploratory Data Analysis
    perform_full_eda(df_fe)

    # Step 4: Encoding Categorical Data
    categorical_features = ['category']
    df_enc = encode_categorical_features(df_fe, categorical_features)

    # Step 5: Convert Boolean Columns
    df_bool = convert_boolean_columns(df_enc)

    # Step 6: Winsorise Outliers
    df_win = winsorise_outliers(df_bool)

    # Step 7: Scale Numeric Features
    X_scaled = scale_numeric_features(df_win)

    # Step 8: Find Optimal k
    optimal_k, _, _ = find_optimal_k(X_scaled)

    # Step 9: Assign Clusters
    kmeans_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df_clustered = assign_clusters(df_win, X_scaled, optimal_k)
    df_clustered['cluster'] = kmeans_model.fit_predict(X_scaled)

    # Step 10: Cluster Profiling & Naming
    cluster_means = calculate_cluster_means(df_clustered)
    cluster_name_mapping = {
        0: "Low-Value Chemicals",
        1: "High-Value Pharmaceuticals",
        2: "Mid-Value Electronics"
    }
    cluster_means_named, used_mapping = rename_clusters(cluster_means, cluster_name_mapping)

    # Step 11: PCA Visualisation
    plot_clusters_pca(X_scaled, df_clustered, kmeans_model, 'cluster')

    # Step 12: Cluster Summaries
    summarise_clusters(df_clustered, used_mapping)

    # Step 13: Prepare Time Series
    ts = prepare_time_series_by_cluster(df_clustered)

    # Step 14: Stationarity Check
    test_stationarity_by_cluster(ts)

    # Step 15: Time Series Decomposition
    decompose_cluster_time_series(ts)

    # Step 16: ACF & PACF
    plot_acf_pacf_by_cluster(ts)

    # Step 17: Train/Test Split
    ts_train, ts_test = split_cluster_time_series(ts)

    # Define the number of forecast periods (months)
    n_forecast_periods = 12

    # Step 18: Tune SARIMAX
    sarimax_best_params = tune_sarimax_per_cluster(ts_train, seasonal_period=12)

    # Step 19: Train SARIMAX Models
    final_sarimax_models, sarimax_forecasts = retrain_and_forecast_sarimax(ts_train, sarimax_best_params, n_steps_forecast=n_forecast_periods)

    # Step 20: Residual Diagnostics
    analyse_residuals(final_sarimax_models)

    # Step 21: Forecast with SARIMAX
    sarimax_forecasts_step21 = generate_cluster_forecasts(final_sarimax_models, ts_test)

    # Step 22: Plot Actual vs Predicted
    plot_actual_vs_predicted(ts_test, sarimax_forecasts_step21)

    # Step 23: Evaluate SARIMAX
    sarimax_evaluation_results = evaluate_forecast_accuracy(ts_test, sarimax_forecasts_step21)
    display(pd.DataFrame(sarimax_evaluation_results).T)

    # Step 24: Visualise SARIMAX Forecasts
    visualise_forecasts_on_test_data(ts_train, ts_test, sarimax_forecasts_step21)

    # Step 25: Prepare Prophet Data
    prophet_data = prepare_prophet_data(ts)

    # Step 26: Train Prophet Models
    prophet_models = train_prophet_models(ts_train)

    # Step 27: Generate and Visualise Prophet Forecasts
    _, prophet_baseline_forecasts = generate_and_visualise_prophet_forecasts(prophet_models, ts_train, ts_test)
    prophet_forecasts = prophet_baseline_forecasts

    # Step 28: Evaluate Prophet Forecasts
    prophet_metrics = evaluate_prophet_forecasts(ts_test, prophet_forecasts)
    display(pd.DataFrame(prophet_metrics).T)

    # Step 29: Tune Prophet Parameters
    prophet_param_grid = {
        'changepoint_prior_scale': [0.05, 0.1, 0.5],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    # Pass n_forecast_periods to tune_prophet_parameters
    tuned_params = tune_prophet_parameters(ts_train, prophet_param_grid, n_forecasts=n_forecast_periods)

    # Step 30: Generate Tuned Prophet Forecasts
    prophet_forecasts_tuned = generate_tuned_prophet_forecasts(prophet_data, ts_test, tuned_params)

    # Step 31: Evaluate Tuned Prophet Forecasts
    prophet_metrics_tuned = evaluate_tuned_prophet_forecasts(ts_test, prophet_forecasts_tuned)
    display(pd.DataFrame(prophet_metrics_tuned).T if prophet_metrics_tuned else "No tuned Prophet metrics available.")

    # Step 32: Plot Actual vs Tuned Prophet Forecasts
    plot_actual_vs_tuned_forecasts(ts_test, prophet_forecasts_tuned)

    # Step 33: Compare Forecast Metrics
    comparison_df = compare_forecast_metrics(prophet_metrics_tuned, sarimax_evaluation_results)

    return {
        "feedback_df": feedback_df,
        "sentiment_summary": sentiment_summary,
        "cluster_means": cluster_means_named,
        "cluster_name_mapping": used_mapping,
        "sarimax_forecasts": sarimax_forecasts_step21,
        "prophet_forecasts": prophet_forecasts,
        "tuned_prophet_forecasts": prophet_forecasts_tuned,
        "sarimax_metrics": sarimax_evaluation_results,
        "prophet_metrics": prophet_metrics,
        "prophet_metrics_tuned": prophet_metrics_tuned,
        "comparison": comparison_df,
    }

# Run the main pipeline
if __name__ == "__main__":
    main_pipeline()