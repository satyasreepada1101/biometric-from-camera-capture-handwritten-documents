"""
Clustering_v2_agglomerative_clustering.py

V2 adaptation of the project's V1 Agglomerative Clustering experiment.
Uses ONLY Clustering_v2/features page-level arrays and saves outputs under
Clustering_v2/clustering_results.

Runs K=40 Agglomerative Clustering with Ward, average, and complete linkage
on two train-fitted feature representations:
  1. Raw AlexNet page features -> PCA retaining 95% variance
  2. StandardScaler -> PCA retaining 95% variance

Writer labels are used only to report clustering metrics; they do not influence
feature preprocessing or cluster formation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
FEATURE_DIR = BASE_DIR / "features"
RESULT_DIR = BASE_DIR / "clustering_results"

N_CLUSTERS = 40
LINKAGES = ["ward", "average", "complete"]
RANDOM_STATE = 42


def safe_silhouette(X, labels):
    occupied = len(np.unique(labels))
    if occupied < 2 or occupied >= len(labels):
        return np.nan
    return float(silhouette_score(X, labels))


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    x_path = FEATURE_DIR / "X_train_page.npy"
    y_path = FEATURE_DIR / "y_train_page.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(
            "Missing X_train_page.npy or y_train_page.npy in Clustering_v2/features. "
            "Run the page aggregation script first."
        )

    X = np.load(x_path).astype(np.float64)
    y = np.load(y_path).ravel()
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("Invalid feature matrix or label dimensions.")
    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinite values.")
    if N_CLUSTERS >= len(X):
        raise ValueError("Number of clusters must be less than number of samples.")

    print("\n========== CLUSTERING_V2 DATASET INFO ==========")
    print(f"Training pages : {len(X)}")
    print(f"Feature count  : {X.shape[1]}")
    print(f"Writers        : {len(np.unique(y))}")
    print(f"Target clusters: {N_CLUSTERS}")
    print("===============================================")

    # Variant A: preserve the original embedding scale before PCA.
    pca_raw = PCA(n_components=0.95, svd_solver="full", random_state=RANDOM_STATE)
    X_raw_pca = pca_raw.fit_transform(X)

    # Variant B: reproduce the V1-style standardized preprocessing, but use
    # a train-only 95%-variance PCA so it remains comparable in dimensionality logic.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca_scaled = PCA(n_components=0.95, svd_solver="full", random_state=RANDOM_STATE)
    X_scaled_pca = pca_scaled.fit_transform(X_scaled)

    feature_sets = {
        "raw_features_pca95": X_raw_pca,
        "standard_scaled_pca95": X_scaled_pca,
    }

    print(f"Raw PCA dimensions      : {X_raw_pca.shape[1]}")
    print(f"Raw PCA variance retained: {pca_raw.explained_variance_ratio_.sum():.4f}")
    print(f"Scaled PCA dimensions   : {X_scaled_pca.shape[1]}")
    print(f"Scaled PCA variance kept: {pca_scaled.explained_variance_ratio_.sum():.4f}")

    rows = []
    assignments = pd.DataFrame({
        "sample_index": np.arange(len(y)),
        "true_writer": y,
    })

    for feature_name, X_features in feature_sets.items():
        for linkage in LINKAGES:
            print(f"\nRunning {feature_name}, linkage={linkage}")
            model = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage=linkage)
            cluster_ids = model.fit_predict(X_features)
            occupied = len(np.unique(cluster_ids))
            silhouette = safe_silhouette(X_features, cluster_ids)
            ari = adjusted_rand_score(y, cluster_ids)
            nmi = normalized_mutual_info_score(y, cluster_ids)

            rows.append({
                "feature_method": feature_name,
                "linkage": linkage,
                "n_samples": len(y),
                "n_features": X_features.shape[1],
                "n_clusters_requested": N_CLUSTERS,
                "n_clusters_found": occupied,
                "silhouette_score": silhouette,
                "adjusted_rand_index": ari,
                "normalized_mutual_info": nmi,
            })
            assignments[f"{feature_name}_{linkage}_cluster"] = cluster_ids
            print(f"Clusters found: {occupied}")
            print(f"Silhouette    : {silhouette:.4f}")
            print(f"ARI           : {ari:.4f}")
            print(f"NMI           : {nmi:.4f}")

    results_df = pd.DataFrame(rows)
    results_path = RESULT_DIR / "agglomerative_comparison.csv"
    assignments_path = RESULT_DIR / "agglomerative_assignments.csv"
    results_df.to_csv(results_path, index=False)
    assignments.to_csv(assignments_path, index=False)

    print("\n========== RESULTS RANKED BY ARI ==========")
    print(results_df.sort_values("adjusted_rand_index", ascending=False).to_string(index=False))
    print("\nSaved results:")
    print(results_path)
    print(assignments_path)
    print("\nNote: these are training clustering metrics. Use the separate held-out script for test evaluation.")


if __name__ == "__main__":
    main()
