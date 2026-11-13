"""
Clustering_v2_agglomerative_test_evaluation.py

Held-out evaluation for V2 Agglomerative Clustering.
Fits preprocessing and Agglomerative Clustering on train pages only, then
assigns test pages using:
  1. nearest training sample's learned cluster
  2. nearest centroid of the learned training clusters

The approach is consistent with the supplied V1 evaluation script, but paths
are restricted to Clustering_v2/features and Clustering_v2/clustering_results.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
FEATURE_DIR = BASE_DIR / "features"
RESULT_DIR = BASE_DIR / "clustering_results"
N_CLUSTERS = 40
RANDOM_STATE = 42


def evaluate(method, predictions, X_test_pca, y_test):
    occupied = len(np.unique(predictions))
    silhouette = (
        float(silhouette_score(X_test_pca, predictions))
        if 1 < occupied < len(predictions)
        else np.nan
    )
    return {
        "method": method,
        "test_samples": len(y_test),
        "occupied_test_clusters": occupied,
        "test_silhouette": silhouette,
        "test_ari": adjusted_rand_score(y_test, predictions),
        "test_nmi": normalized_mutual_info_score(y_test, predictions),
    }


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    names = ["X_train_page.npy", "y_train_page.npy", "X_test_page.npy", "y_test_page.npy"]
    missing = [name for name in names if not (FEATURE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing files in Clustering_v2/features: " + ", ".join(missing)
        )

    X_train = np.load(FEATURE_DIR / "X_train_page.npy").astype(np.float64)
    y_train = np.load(FEATURE_DIR / "y_train_page.npy").ravel()
    X_test = np.load(FEATURE_DIR / "X_test_page.npy").astype(np.float64)
    y_test = np.load(FEATURE_DIR / "y_test_page.npy").ravel()

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and label counts do not match.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train/test feature dimensions do not match.")
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Features contain NaN or infinite values.")

    print("\n========== CLUSTERING_V2 DATASET INFO ==========")
    print(f"Training pages: {len(X_train)}")
    print(f"Test pages    : {len(X_test)}")
    print(f"Feature count : {X_train.shape[1]}")
    print("===============================================")

    # The raw-features + PCA route performed best in the initial V2 comparison,
    # so use it as the primary held-out evaluation representation.
    pca = PCA(n_components=0.95, svd_solver="full", random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    print(f"PCA dimensions: {X_train_pca.shape[1]}")
    print(f"PCA variance  : {pca.explained_variance_ratio_.sum():.4f}")

    ward = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward")
    train_clusters = ward.fit_predict(X_train_pca)
    print(f"Training clusters found: {len(np.unique(train_clusters))}")

    nearest = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nearest.fit(X_train_pca)
    distances, nearest_indices = nearest.kneighbors(X_test_pca)
    nearest_sample_predictions = train_clusters[nearest_indices[:, 0]]

    cluster_ids = np.unique(train_clusters)
    centroids = np.vstack([
        X_train_pca[train_clusters == cluster_id].mean(axis=0)
        for cluster_id in cluster_ids
    ])
    centroid_search = NearestNeighbors(n_neighbors=1, metric="euclidean")
    centroid_search.fit(centroids)
    _, centroid_indices = centroid_search.kneighbors(X_test_pca)
    centroid_predictions = cluster_ids[centroid_indices[:, 0]]

    results = [
        evaluate("Nearest_training_sample", nearest_sample_predictions, X_test_pca, y_test),
        evaluate("Nearest_cluster_centroid", centroid_predictions, X_test_pca, y_test),
    ]
    results_df = pd.DataFrame(results)

    assignments = pd.DataFrame({
        "sample_index": np.arange(len(y_test)),
        "true_writer": y_test,
        "nearest_training_sample_cluster": nearest_sample_predictions,
        "nearest_training_sample_distance": distances[:, 0],
        "nearest_centroid_cluster": centroid_predictions,
    })

    metrics_path = RESULT_DIR / "agglomerative_test_metrics.csv"
    assignments_path = RESULT_DIR / "agglomerative_test_assignments.csv"
    bundle_path = RESULT_DIR / "agglomerative_test_bundle.joblib"
    results_df.to_csv(metrics_path, index=False)
    assignments.to_csv(assignments_path, index=False)
    joblib.dump({
        "pca": pca,
        "ward_training_labels": train_clusters,
        "X_train_pca": X_train_pca,
        "cluster_ids": cluster_ids,
        "cluster_centroids": centroids,
        "nearest_training_sample_model": nearest,
        "nearest_centroid_model": centroid_search,
    }, bundle_path)

    print("\n========== HELD-OUT TEST RESULTS ==========")
    for result in results:
        print(f"\nMethod     : {result['method']}")
        print(f"Clusters   : {result['occupied_test_clusters']}")
        print(f"Silhouette : {result['test_silhouette']:.4f}")
        print(f"ARI        : {result['test_ari']:.4f}")
        print(f"NMI        : {result['test_nmi']:.4f}")

    print("\nSaved files:")
    print(metrics_path)
    print(assignments_path)
    print(bundle_path)
    print("\nEvaluation completed.")
    print("Note: cluster IDs are not writer IDs; these metrics evaluate cluster agreement.")


if __name__ == "__main__":
    main()
