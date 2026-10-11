
from typing import Tuple
from typing import Dict
from typing import Dict, Optional, Any
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display



def scale_numeric_features(df: pd.DataFrame, exclude: Optional[list] = None) -> np.ndarray:
    """
    Scales numeric features using StandardScaler for clustering and analysis.

    Steps
    ------
    - Identifies numerical columns in the DataFrame.
    - Initialises a StandardScaler.
    - Fits the scaler to the numerical features and transforms them.
    - Returns the scaled feature matrix as a NumPy array.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame with numeric features.
    exclude : list, optional
        Numeric columns to leave out of clustering (e.g. date parts and category flags),
        so that products are grouped on how they behave, not on labels or dates.

    Returns
    -------
    np.ndarray
        Scaled numeric feature matrix.
    """
    exclude = exclude or []
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]
    print("Features used for clustering:", num_cols)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_cols])

    print("First 5 rows of scaled features:")
    display(X_scaled[:5])

    return X_scaled

"""## Step 8: Optimal k Detection

What it does:

Tests different numbers of clusters (*k*) to see which one works best by using:

- Inertia: Measures how tightly grouped the items are in each cluster.
- Silhouette Score: Measures how well each item fits in its cluster compared to others.


Why it's important:
- Helps pick the best number of clusters without making them too many (overfitting) or too few (underfitting).


Output Analysis:

- Inertia drops sharply up to k = 3 and flattens afterwards (the "elbow").
- The silhouette score peaks at k = 3 (about 0.45), so three clusters give the clearest separation.
"""

def find_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 11)) -> Tuple[int, list, list]:
    """
    Determines the optimal number of clusters using the Elbow and Silhouette methods.

    Steps
    ------
    - Initialises empty lists for inertias and silhouette scores.
    - Iterates through the specified range of k values.
    - For each k, fits a KMeans model to the scaled data and predicts cluster labels.
    - Appends the inertia and silhouette score to their respective lists.
    - Plots the inertia and silhouette scores against k.
    - Determines the optimal k as the value with the highest silhouette score.
    - Returns the optimal k, the list of inertias, and the list of silhouette scores.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix for clustering.
    k_range : range, optional
        Range of k values to evaluate.

    Returns
    -------
    int
        The optimal number of clusters based on the highest silhouette score.
    list
        List of inertia values for each k.
    list
        List of silhouette scores for each k.
    """
    inertias = []
    silhouettes = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'o-')
    plt.title("Elbow Method")
    plt.xlabel("k")
    plt.ylabel("Inertia")

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouettes, 'o-r')
    plt.title("Silhouette Score")
    plt.xlabel("k")
    plt.ylabel("Score")

    plt.tight_layout()
    plt.show()

    optimal_k = k_range[np.argmax(silhouettes)]
    print(f"Optimal number of clusters based on silhouette score: {optimal_k}")

    return optimal_k, inertias, silhouettes

"""## Step 9: Clustering

What it does:

- K-Means groups items into clusters based on how similar they are. Each cluster has a centroid (average point) in that group.

- To assign an item to a cluster, the model calculates the Euclidean distance—a straight-line distance—between the item and each centroid. The item is placed in the cluster with the nearest centroid.

Why it's important:

- Groups items with similar characteristics for efficient storage and planning.
"""

def assign_clusters(df: pd.DataFrame, X_scaled: np.ndarray, n_clusters: int) -> pd.DataFrame:
    """
    Assigns KMeans cluster labels to the DataFrame and counts the number of items in each cluster.

    Steps
    ------
    - Creates a copy of the input DataFrame.
    - Initialises and fits a KMeans model to the scaled feature matrix, predicts cluster labels.
    - Adds a 'cluster' column with assigned labels, prints item count per cluster.
    - Returns the DataFrame with cluster assignments.

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame to which cluster labels will be added.
    X_scaled : np.ndarray
        Scaled feature matrix used for clustering.
    n_clusters : int
        The number of clusters to form.

    Returns
    -------
    pd.DataFrame
        The DataFrame with an added 'cluster' column indicating cluster assignments.
    """
    df_clustered = df.copy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clustered['cluster'] = kmeans.fit_predict(X_scaled)

    print("Cluster counts:")
    print(df_clustered['cluster'].value_counts())

    return df_clustered

"""## Step 10: Cluster Profiling & Naming

"""

def calculate_cluster_means(df: pd.DataFrame, cluster_col: str = 'cluster') -> pd.DataFrame:
    """
    Groups the DataFrame by the cluster column and calculates the mean of all numerical features.

    Steps
    ------
    - Groups the input DataFrame by the specified cluster column.
    - Calculates the mean of all numerical features for each cluster.
    - Returns a DataFrame containing the mean values per cluster.

    Parameters
    ----------
    df : pd.DataFrame
        The clustered DataFrame containing a column with cluster labels.
    cluster_col : str
        The name of the column containing cluster labels.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the mean values of numerical features for each cluster.
    """
    cluster_means = df.groupby(cluster_col).mean(numeric_only=True)

    print("Mean values of features by cluster:")
    display(cluster_means)

    return cluster_means

"""Why This Step:

- To give each cluster a clear, descriptive name that reflects its operational characteristics.

Why it's important:

- These intuitive names help stakeholders quickly understand product groupings for better warehouse planning.

How names are assigned:

- K-Means numbers its clusters arbitrarily, so names are assigned from each cluster's own profile (see `name_clusters_from_profile`), never from a fixed list. This keeps names correct if the cluster numbering changes.

Output Analysis:

- Even though product category was **not** used as a clustering input, the three behavioural clusters match the three categories for 99.2% of products. Each category behaves distinctly in this warehouse, which validates category-level storage rules.

- Cluster 0 (Fast-Moving Chemicals): the highest pick frequency (9.0), the lightest items (12 kg), the lowest price (£88) and the shortest lead time (18 days). Ideal for high-access zones near picking and packing.

- Cluster 1 (Slow-Moving Heavy Electronics): the lowest pick frequency (2.7), the heaviest items (84 kg), the longest lead time (28 days), strong seasonality and the lowest storage efficiency. Suited to bulk or less accessible storage with heavy-load racking.

- Cluster 2 (High-Value Pharmaceuticals): the highest price (£234) and the highest volumes (198 units on average), with moderate pick frequency (6.4) and the best storage efficiency. Needs secure, moderately accessible storage with capacity for volume.
"""

def rename_clusters(cluster_means: pd.DataFrame, cluster_name_mapping: dict) -> tuple[pd.DataFrame, dict]:
    """
    Renames the index of the cluster means DataFrame using a mapping of cluster labels to descriptive names.

    Steps
    ------
    - Checks that every cluster label has a name in the mapping and raises an error if not.
    - Renames the index of the cluster means DataFrame using the mapping.
    - Returns the DataFrame with the renamed index and the mapping used.

    Parameters
    ----------
    cluster_means : pd.DataFrame
        DataFrame with cluster labels as index and mean values of features.
    cluster_name_mapping : dict
        Dictionary mapping cluster labels (int) to descriptive names (str).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The renamed DataFrame and the mapping used.
    """
    missing = set(cluster_means.index) - set(cluster_name_mapping.keys())
    if missing:
        raise ValueError(f"Cluster name mapping missing entries for: {missing}")

    renamed = cluster_means.rename(index=cluster_name_mapping)
    print("Cluster means with descriptive names:")
    display(renamed)

    return renamed, cluster_name_mapping


def name_clusters_from_profile(df_clustered: pd.DataFrame, categories: pd.Series, cluster_col: str = 'cluster') -> dict:
    """
    Names each cluster from its own profile, so names always match the data (cluster numbers from K-Means are arbitrary).

    Steps
    ------
    - Cross-tabulates clusters against product category to show how closely the behavioural clusters match categories.
    - Identifies the dominant category in each cluster.
    - Assigns a descriptive name based on that category's operational profile.

    Parameters
    ----------
    df_clustered : pd.DataFrame
        DataFrame with cluster labels.
    categories : pd.Series
        Product category for each row (same index as df_clustered).
    cluster_col : str, optional
        Name of the cluster label column.

    Returns
    -------
    dict
        Mapping of cluster label to descriptive name.
    """
    profile_names = {
        'Chemicals': 'Fast-Moving Chemicals',
        'Electronics': 'Slow-Moving Heavy Electronics',
        'Pharmaceuticals': 'High-Value Pharmaceuticals',
    }
    crosstab = pd.crosstab(df_clustered[cluster_col], categories.loc[df_clustered.index])
    print("Clusters vs product category (category was NOT used for clustering):")
    display(crosstab)
    match_rate = crosstab.max(axis=1).sum() / crosstab.values.sum()
    print(f"Share of products whose cluster matches their category's dominant cluster: {match_rate:.1%}")

    dominant = crosstab.idxmax(axis=1)
    return {c: profile_names.get(cat, f'Cluster {c}') for c, cat in dominant.items()}

"""## Block 11: PCA Visualisation of Clusters

What it does:

- Uses Principal Component Analysis (PCA) to reduce the scaled feature space to two dimensions.
- Plots each item in this 2D space, colored by cluster, with centroids marked.

Centroids Calculation:
- K-Means centroids are the average position (mean vector) of all items in each cluster.
- These centroids are transformed into the PCA space and shown as 'X' markers on the plot.
- This visual shows how clusters are separated and where their centers lie in the reduced space.
"""

def plot_clusters_pca(X_scaled: np.ndarray, df: pd.DataFrame, kmeans: KMeans, cluster_col: str = 'cluster') -> pd.DataFrame:
    """
    Projects high-dimensional data into 2D PCA space and visualises KMeans clusters and centroids.

    Steps
    ------
    - Initialises a PCA model to reduce data to 2 components, fits and transforms the scaled feature matrix.
    - Creates a DataFrame with PCA components and cluster labels.
    - Define cluster colour for consistency with EDA visualisations.
    - Plots data points in 2D PCA space, coloured by cluster, and adds cluster centroids.
    - Displays the plot and returns the DataFrame with PCA components and labels.

    Parameters
    ----------
    X_scaled : np.ndarray
        Scaled feature matrix used for PCA projection.
    df : pd.DataFrame
        DataFrame containing cluster labels.
    kmeans : KMeans
        Trained KMeans model with cluster centres.
    cluster_col : str, optional
        Name of the column in df containing cluster labels (default is 'cluster').

    Returns
    -------
    pd.DataFrame
        DataFrame containing PCA components and cluster labels.
    """
    pca = PCA(n_components=2)
    proj = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(proj, columns=['PC1', 'PC2'], index=df.index)
    pca_df[cluster_col] = df[cluster_col]

    color_map = {
        0: sns.color_palette()[0],
        1: sns.color_palette()[2],
        2: sns.color_palette()[1]
    }

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue=cluster_col, palette=color_map, s=60)

    centroids_2d = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker='X', s=200, c='black', label='Centroids')

    plt.title('K-Means Clusters (PCA)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return pca_df

"""## Step 12: Cluster-Based Recommendations"""

def summarise_clusters(
    df: pd.DataFrame,
    cluster_name_mapping: Dict[int, str],
    cluster_col: str = 'cluster'
) -> None:
    """
    Prints a summary of each cluster with descriptive statistics and storage recommendations.

    Steps
    ------
    - Iterates through each unique cluster ID in the DataFrame.
    - Skips cluster -1 (typically for noise or unassigned points).
    - Filters the DataFrame to get data for the current cluster.
    - Retrieves the descriptive name for the cluster using the mapping.
    - Prints summary statistics for the cluster, including product count, average volume, price, etc.
    - Prints a storage recommendation based on the cluster's characteristics, using the cluster name mapping.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing cluster labels and relevant numeric features.
    cluster_name_mapping : dict
        Dictionary mapping cluster numbers to descriptive names.
    cluster_col : str, optional
        Name of the column containing cluster labels (default is 'cluster').
    """
    for c in sorted(df[cluster_col].unique()):
        if c == -1:
            print("\nSkipping cluster -1 (items with missing data).")
            continue

        data_c = df[df[cluster_col] == c].copy()
        cluster_display_name = cluster_name_mapping.get(c, f'Cluster {c} (Unknown Name)')

        print(f"\nCluster {c} ({cluster_display_name}):")
        print(f"  Products: {len(data_c)}")
        print(f"  Average Volume: {data_c['avg_volume'].mean():.1f}")
        print(f"  Average Price: £{data_c['price'].mean():.2f}")
        print(f"  Volume CV: {data_c['volume_cv'].mean():.2f}")

        storage_efficiency_mean = data_c['storage_efficiency'].replace([np.inf, -np.inf], np.nan).mean()
        print(f"  Storage Efficiency: {storage_efficiency_mean:.2f}")
        print(f"  Average Weight: {data_c['weight_kg'].mean():.2f} kg")
        print(f"  Average Pick Frequency: {data_c['pick_frequency'].mean():.1f}")
        print(f"  Average Lead Time: {data_c['lead_time_days'].mean():.1f} days")

        if c in cluster_name_mapping:
            if cluster_name_mapping[c] == "Fast-Moving Chemicals":
                print("  Recommendation: Store in high-access zones near picking/packing. These light items have the highest pick frequency and shortest lead times, so reducing travel time matters most.")
            elif cluster_name_mapping[c] == "Slow-Moving Heavy Electronics":
                print("  Recommendation: Store in bulk or less accessible areas with heavy-load racking. These are the heaviest items with the lowest pick frequency and longest lead times.")
            elif cluster_name_mapping[c] == "High-Value Pharmaceuticals":
                print("  Recommendation: Store in secure, moderately accessible zones with capacity for high volumes. These are the most valuable items, with the highest volumes and moderate pick frequency.")
            else:
                print("  Recommendation: No specific recommendation available for this cluster.")

