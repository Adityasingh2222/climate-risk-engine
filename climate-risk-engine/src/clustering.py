"""
clustering.py
-------------
Unsupervised learning to group portfolio holdings into "climate risk
archetypes" using: sector carbon intensity, empirical carbon beta,
EBITDA margin, and valuation multiple.

REVISED METHODOLOGY (v2 — see REFERENCES.md changelog):
Two issues flagged in review, now addressed:

1. Z-scoring bug: the original version standardized (z-scored) the risk
   ranking across just the k cluster CENTROIDS (e.g. only 3 points for
   k=3) — with that few points, the standard deviation used for
   z-scoring is itself noisy and can distort the ranking. This version
   computes the risk score for every COMPANY using z-scores against the
   full population (already computed by StandardScaler across all
   holdings), then averages within each cluster to rank clusters. This
   is the statistically correct order of operations: standardize against
   the full sample first, aggregate second.

2. Small-sample instability: K-means on a small portfolio (well under
   ~30 holdings) can produce unstable, outlier-driven clusters. This
   version now (a) warns explicitly when the holdings-to-clusters ratio
   is low, and (b) offers Agglomerative (hierarchical) clustering as an
   alternative method, which tends to be more stable on small samples
   and doesn't assume round/equal-sized clusters the way k-means does.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering

FEATURE_COLUMNS = ["carbon_intensity", "beta_carbon", "ebitda_margin", "pe_ratio"]


def build_feature_matrix(fundamentals_df: pd.DataFrame, betas_df: pd.DataFrame,
                          intensity_map: dict) -> pd.DataFrame:
    """
    Merge fundamentals + carbon betas + sector carbon intensity into a
    single feature matrix, one row per ticker.

    intensity_map: dict of sector -> tCO2e/$M (e.g. climate_data.SECTOR_CARBON_INTENSITY)
    """
    df = fundamentals_df.set_index("ticker").copy()
    df["carbon_intensity"] = df["sector"].map(
        lambda s: intensity_map.get(s, intensity_map.get("Unknown", 150.0))
    )
    df = df.join(betas_df[["beta_carbon", "r_squared"]], how="left")
    df["beta_carbon"] = df["beta_carbon"].fillna(0.0)
    return df


def cluster_risk_archetypes(feature_df: pd.DataFrame, n_clusters: int = 3,
                             method: str = "kmeans", seed: int = 42):
    """
    Groups holdings into risk archetypes.

    method: "kmeans" (default) or "agglomerative" (hierarchical -
        generally more stable for small portfolios).

    Returns (feature_df_with_labels, centroids_df, warning_message_or_None).
    """
    X = feature_df[FEATURE_COLUMNS].copy()
    X = X.fillna(X.mean())

    n_samples = len(X)
    n_clusters = max(1, min(n_clusters, n_samples))

    warning = None
    if n_clusters < 2 or n_samples < 3:
        feature_df = feature_df.copy()
        feature_df["cluster"] = 0
        feature_df["archetype"] = "N/A (too few holdings to cluster)"
        return feature_df, pd.DataFrame(), "Portfolio too small to cluster meaningfully (need at least 3 holdings)."

    if n_samples < 4 * n_clusters:
        warning = (
            f"Only {n_samples} holdings for {n_clusters} clusters — cluster "
            f"assignments are illustrative and may be unstable. Consider fewer "
            f"clusters or a larger portfolio for a more reliable grouping."
        )

    # Standardize against the FULL population — this is what fixes the
    # earlier z-scoring bug (was done across only the centroids instead).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)

    labels = model.fit_predict(X_scaled)

    feature_df = feature_df.copy()
    feature_df["cluster"] = labels

    # Per-company risk score using population-level z-scores (statistically
    # sound), THEN averaged within each cluster to rank clusters.
    scaled_df = pd.DataFrame(X_scaled, columns=FEATURE_COLUMNS, index=feature_df.index)
    feature_df["risk_score"] = (
        scaled_df["carbon_intensity"]
        - scaled_df["beta_carbon"]        # more negative beta -> higher risk
        - scaled_df["ebitda_margin"]       # lower margin -> higher risk
    )

    cluster_risk = feature_df.groupby("cluster")["risk_score"].mean().sort_values(ascending=False)
    ranked_cluster_ids = cluster_risk.index.tolist()  # highest risk first

    archetype_names = _name_archetypes(n_clusters)
    label_map = {cid: archetype_names[i] for i, cid in enumerate(ranked_cluster_ids)}
    feature_df["archetype"] = feature_df["cluster"].map(label_map)

    # Centroids (cluster means in original units) for inspection/display.
    if method == "agglomerative":
        centroids = feature_df.groupby("cluster")[FEATURE_COLUMNS].mean().reset_index()
    else:
        centroids = pd.DataFrame(
            scaler.inverse_transform(model.cluster_centers_), columns=FEATURE_COLUMNS
        )
        centroids["cluster"] = range(len(centroids))

    centroids["archetype"] = centroids["cluster"].map(label_map)
    centroids["mean_risk_score"] = centroids["cluster"].map(cluster_risk)

    return feature_df, centroids, warning


def _name_archetypes(n):
    ordered = [
        "High Transition Risk",
        "Moderate Risk / Adapting",
        "Resilient / Low Exposure",
        "Watch List",
        "Niche Cluster",
    ]
    while len(ordered) < n:
        ordered.append(f"Cluster {len(ordered) + 1}")
    return ordered
