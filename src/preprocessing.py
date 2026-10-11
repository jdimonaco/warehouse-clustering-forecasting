import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple
from IPython.display import display



def inspect_inventory_data(file_path: str) -> pd.DataFrame:
    """
    Loads and inspects the inventory dataset.

    Steps
    ------
    - Loads the CSV file into a DataFrame
    - Prints dataset structure using .info()
    - Prints summary statistics using .describe()

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing inventory data.

    Returns
    -------
    pd.DataFrame
        The loaded DataFrame for further processing.

    Example Usage
    -------------
    df = inspect_inventory_data("data/priced_inventory_data (3).csv")
    """
    df = pd.read_csv(file_path)

    print("Dataset Info:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())

    return df

"""## Step 2: Feature Engineering"""

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering on the inventory dataset.

    Steps
    ------
    - Converts 'Received date' to datetime format
    - Imputes missing values using mean
    - Removes duplicate product_id entries
    - Displays quantiles (25%, 50%, 75%) for numerical columns
    - Creates new features:
        - est_monthly_revenue
        - rev_per_kg
        - storage_efficiency
        - volume_weight_ratio
        - is_premium
        - high_seasonality
        - category_popularity

    Returns
    -------
    pd.DataFrame
        Cleaned and enriched DataFrame with new features.

    Example Usage
    -------------
    df = pd.read_csv("data/priced_inventory_data (3).csv")
    df = feature_engineering(df)
    """

    df['Received date'] = pd.to_datetime(df['Received date'], dayfirst=True, errors='coerce')

    print("\nMissing values before imputation:")
    print(df.isnull().sum())

    for column in df.columns:
        if df[column].isnull().any():
            if df[column].dtype in ['int64', 'float64']:
                df[column] = df[column].fillna(df[column].mean())
            else:
                df[column] = df[column].fillna(df[column].mode()[0])

    print("\nMissing values after imputation:")
    print(df.isnull().sum())

    print("\nNumber of duplicate product_ids before removal:", df.duplicated(subset=['product_id']).sum())
    df.drop_duplicates(subset=['product_id'], keep='first', inplace=True)
    print("Number of duplicate product_ids after removal:", df.duplicated(subset=['product_id']).sum())


    numerical_df = df.select_dtypes(include=np.number)
    print("\nQuantiles (25%, 50%, 75%):")
    print(numerical_df.quantile([0.25, 0.5, 0.75]))

    df['est_monthly_revenue'] = df['avg_volume'] * df['price']
    df['rev_per_kg'] = df['est_monthly_revenue'] / df['weight_kg'].replace(0, np.nan)
    df['storage_efficiency'] = df['avg_volume'] / df['storage_cost_per_unit'].replace(0, np.nan)
    df['volume_weight_ratio'] = df['avg_volume'] / df['weight_kg'].replace(0, np.nan)
    df['is_premium'] = (df['price'] > df['price'].median()).astype(int)
    df['high_seasonality'] = (df['seasonality'] > 1.1).astype(int)
    df['category_popularity'] = df.groupby('category')['avg_volume'].transform('mean')

    return df

"""## Step 3: Exploratory Data Analysis"""

def perform_full_eda(df: pd.DataFrame) -> None:
    """
    Performs full exploratory data analysis on the inventory dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The inventory dataset with engineered features.

    Returns
    -------
    None
        Displays all plots.
    """
    sns.set(style="whitegrid")

    # Scatter plots
    # Estimated Monthly Revenue vs. Average Volume by Category
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x='avg_volume', y='est_monthly_revenue', hue='category', s=100)
    plt.title('Estimated Monthly Revenue vs. Average Volume by Category')
    plt.xlabel('Average Volume')
    plt.ylabel('Estimated Monthly Revenue')
    plt.legend(title='Category')
    plt.grid(True)
    plt.show()

    # Estimated Monthly Revenue vs. Storage Cost per Unit by Category
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x='storage_cost_per_unit', y='est_monthly_revenue', hue='category', s=100)
    plt.title('Estimated Monthly Revenue vs. Storage Cost per Unit')
    plt.xlabel('Storage Cost per Unit')
    plt.ylabel('Estimated Monthly Revenue')
    plt.legend(title='Category')
    plt.grid(True)
    plt.show()

    # Revenue per Kg vs. Storage Efficiency by Category
    plt.figure(figsize=(12, 7))
    plot_data = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['rev_per_kg', 'storage_efficiency'])
    sns.scatterplot(data=plot_data, x='storage_efficiency', y='rev_per_kg', hue='category', s=100)
    plt.title('Revenue per Kg vs. Storage Efficiency by Category')
    plt.xlabel('Storage Efficiency')
    plt.ylabel('Revenue per Kg')
    plt.legend(title='Category')
    plt.grid(True)
    plt.show()

    # Pick Frequency vs. Lead Time Days by Category
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x='lead_time_days', y='pick_frequency', hue='category', s=100)
    plt.title('Pick Frequency vs. Lead Time Days by Category')
    plt.xlabel('Lead Time Days')
    plt.ylabel('Pick Frequency')
    plt.legend(title='Category')
    plt.grid(True)
    plt.show()

    # Pick Frequency vs. Average Volume by Category
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=df, x='avg_volume', y='pick_frequency', hue='category', s=100)
    plt.title('Pick Frequency vs. Average Volume by Category')
    plt.xlabel('Average Volume')
    plt.ylabel('Pick Frequency')
    plt.legend(title='Category')
    plt.grid(True)
    plt.show()

    # Pairplot of Key Features by Category
    pairplot_features = ['est_monthly_revenue', 'avg_volume', 'price', 'weight_kg', 'storage_efficiency', 'category']
    sns.pairplot(df[pairplot_features].dropna(), hue='category', diag_kind='kde')
    plt.suptitle('Relationships of Key Features by Category', y=1.02)
    plt.show()

    # Log-transformed revenue
    df['log_est_monthly_revenue'] = np.log1p(df['est_monthly_revenue'])

    # Distribution of Estimated Monthly Revenue
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['est_monthly_revenue'].dropna(), bins=50, kde=True)
    plt.title("Distribution of Estimated Monthly Revenue")
    plt.xlabel("Estimated Monthly Revenue")
    plt.ylabel("Frequency")

    # Distribution of Log(Estimated Monthly Revenue + 1)
    plt.subplot(1, 2, 2)
    sns.histplot(df['log_est_monthly_revenue'].dropna(), bins=50, kde=True, color='orange')
    plt.title("Distribution of Log(Estimated Monthly Revenue + 1)")
    plt.xlabel("Log(Estimated Monthly Revenue + 1)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Average volume distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df['avg_volume'].dropna(), bins=50, kde=True)
    plt.title("Distribution of Average Volume")
    plt.xlabel("Average Volume")
    plt.ylabel("Frequency")
    plt.show()

    # Box Plot of Average Volume
    plt.figure(figsize=(6, 4))
    sns.boxplot(y=df['avg_volume'].dropna())
    plt.title("Box Plot of Average Volume")
    plt.ylabel("Average Volume")
    plt.show()

    # Distribution of Estimated Monthly Revenue by Category
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='category', y='est_monthly_revenue')
    plt.title("Distribution of Estimated Monthly Revenue by Category")
    plt.xlabel("Category")
    plt.ylabel("Estimated Monthly Revenue")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Distribution of Estimated Monthly Revenue by Premium Status
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='is_premium', y='est_monthly_revenue')
    plt.title("Distribution of Estimated Monthly Revenue by Premium Status")
    plt.xlabel("Is Premium (0: No, 1: Yes)")
    plt.ylabel("Estimated Monthly Revenue")
    plt.xticks(ticks=[0, 1], labels=['Not Premium', 'Premium'])
    plt.show()

    # Distribution of Seasonality by Category
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='category', y='seasonality')
    plt.title("Distribution of Seasonality by Category")
    plt.xlabel("Category")
    plt.ylabel("Seasonality Index")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Correlation heatmap of Key Numerical Features
    plt.figure(figsize=(10, 8))
    corr_features = ['est_monthly_revenue', 'avg_volume', 'price', 'weight_kg', 'storage_cost_per_unit', 'seasonality']
    sns.heatmap(df[corr_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Key Numerical Features including Revenue')
    plt.show()

"""## Step 4: Encoding categorical data"""

def encode_categorical_features(df: pd.DataFrame, categorical_features: list) -> pd.DataFrame:
    """
    Applies one-hot encoding to specified categorical features.

    Steps
    ------
    - Selects the specified categorical features.
    - Applies one-hot encoding using 'pd.get_dummies'.
    - Removes the original categorical columns.
    - Returns the DataFrame with encoded features.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing categorical columns.
    categorical_features : list
        List of column names to encode.

    Returns
    -------
    pd.DataFrame
        DataFrame with one-hot encoded categorical features.
    """
    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    print("DataFrame after One-Hot Encoding:")
    display(df_encoded.head())

    print("\nData Types after Encoding:")
    print(df_encoded.dtypes)

    return df_encoded

"""## Step 5: OneHot Encoding categorical data"""

def convert_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts boolean columns in the DataFrame to integer (0/1) format.

    Steps
    ------
    - Iterates through each column in the DataFrame.
    - Checks if the column's data type is boolean.
    - If boolean, converts the column to integer type (0 or 1).
    - Returns the DataFrame with boolean columns converted to integers.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with potential boolean columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with boolean columns converted to integers.
    """
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    print("DataFrame after encoding:")
    display(df.head())

    print("\nData Types after encoding:")
    print(df.dtypes)

    return df

"""## Step 6: Outlier Winsorisation

Winsorisation limits extreme values in data. It replaces values below a low percentile (5%), and values above a high percentile (95%) with that percentile's value.

This reduces the impact of outliers in clustering, leading to more stable groupings.
"""

def winsorise_outliers(df: pd.DataFrame, lower_quantile: float = 0.05, upper_quantile: float = 0.95) -> pd.DataFrame:
    """
    Caps numeric features at specified lower and upper quantiles to reduce the influence of outliers.

    Steps
    ------
    - Creates a copy of the input DataFrame.
    - Identifies and processes numerical columns.
    - Replaces values below the lower quantile and above the upper quantile with the quantile values.
    - Returns the DataFrame with winsorised numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with numerical features.
    lower_quantile : float
        The lower quantile threshold (default is 0.05).
    upper_quantile : float
        The upper quantile threshold (default is 0.95).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with winsorised numeric features.
    """
    df_win = df.copy()
    num_cols = df_win.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        lower = df_win[col].quantile(lower_quantile)
        upper = df_win[col].quantile(upper_quantile)
        df_win[col] = df_win[col].clip(lower, upper)

    print("DataFrame after Winsorisation:")
    display(df_win.head())

    return df_win
