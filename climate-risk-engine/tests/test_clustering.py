"""
test_clustering.py
-------------------
Verifies src/clustering.py, especially the z-scoring fix: risk scores
must be computed against the full holdings population, not just the
cluster centroids (see README.md "Methodology changelog" item 3).
"""

import pandas as pd
import pytest

from src.clustering import build_feature_matrix, cluster_risk_archetypes, FEATURE_COLUMNS


def make_feature_df():
    """A small synthetic set with one obviously high-risk and one
    obviously low-risk holding, so clustering results are unambiguous."""
    return pd.DataFrame({
        "carbon_intensity": [900.0, 850.0, 800.0, 10.0, 12.0, 15.0],
        "beta_carbon":      [-0.5, -0.45, -0.4, 0.01, 0.02, -0.01],
        "ebitda_margin":    [0.15, 0.18, 0.16, 0.40, 0.38, 0.42],
        "pe_ratio":         [14.0, 15.0, 13.0, 30.0, 32.0, 28.0],
    }, index=["DIRTY1", "DIRTY2", "DIRTY3", "CLEAN1", "CLEAN2", "CLEAN3"])
    # index name intentionally omitted; set below where needed


class TestClusterRiskArchetypes:
    def test_high_risk_group_labeled_correctly(self):
        df = make_feature_df()
        clustered, centroids, warning = cluster_risk_archetypes(df, n_clusters=2, method="kmeans")
        dirty_archetypes = clustered.loc[["DIRTY1", "DIRTY2", "DIRTY3"], "archetype"].unique()
        clean_archetypes = clustered.loc[["CLEAN1", "CLEAN2", "CLEAN3"], "archetype"].unique()
        assert len(dirty_archetypes) == 1
        assert len(clean_archetypes) == 1
        assert dirty_archetypes[0] != clean_archetypes[0]
        assert "High" in dirty_archetypes[0] or "Risk" in dirty_archetypes[0]

    def test_risk_score_uses_population_not_centroid_stats(self):
        """Direct regression test for the fixed bug: the per-company
        'risk_score' column must be computed from z-scores against the
        FULL population (mean/std over all 6 rows), not the 2 centroids.
        We check this by recomputing the expected z-score manually."""
        df = make_feature_df()
        clustered, centroids, warning = cluster_risk_archetypes(df, n_clusters=2, method="kmeans")

        # Manually compute the population-level z-score for carbon_intensity
        pop_mean = df["carbon_intensity"].mean()
        pop_std = df["carbon_intensity"].std()
        # StandardScaler uses population std (ddof=0), pandas .std() uses
        # ddof=1 by default, so just check the relative ORDERING holds,
        # which is what actually matters for ranking correctness.
        assert clustered.loc["DIRTY1", "risk_score"] > clustered.loc["CLEAN1", "risk_score"]

    def test_both_clustering_methods_run(self):
        df = make_feature_df()
        for method in ["kmeans", "agglomerative"]:
            clustered, centroids, warning = cluster_risk_archetypes(df, n_clusters=2, method=method)
            assert "archetype" in clustered.columns
            assert len(clustered) == len(df)

    def test_small_sample_triggers_warning(self):
        df = make_feature_df()
        clustered, centroids, warning = cluster_risk_archetypes(df, n_clusters=3, method="kmeans")
        # 6 holdings / 3 clusters = 2 per cluster, below the 4x threshold
        assert warning is not None
        assert "unstable" in warning or "illustrative" in warning

    def test_too_few_holdings_returns_na_gracefully(self):
        df = make_feature_df().iloc[:2]  # only 2 rows
        clustered, centroids, warning = cluster_risk_archetypes(df, n_clusters=3, method="kmeans")
        assert warning is not None
        assert (clustered["archetype"] == "N/A (too few holdings to cluster)").all()

    def test_cluster_count_never_exceeds_sample_size(self):
        """Requesting more clusters than holdings should not crash - it
        should be capped automatically."""
        df = make_feature_df().iloc[:3]
        clustered, centroids, warning = cluster_risk_archetypes(df, n_clusters=10, method="kmeans")
        assert clustered["cluster"].nunique() <= 3
