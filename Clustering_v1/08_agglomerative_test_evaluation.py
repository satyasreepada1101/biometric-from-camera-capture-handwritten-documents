"""
Held-out evaluation for PCA-100 + Agglomerative Ward.

Assignment strategies:
1. Nearest training sample
2. Nearest cluster centroid

All preprocessing and clustering are fitted on training pages only.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = ROOT / "features"
RESULT_DIR = ROOT / "Clustering" / "clustering_results"

X_TRAIN_PATH = FEATURE_DIR / "X_train_page.npy"
Y_TRAIN_PATH = FEATURE_DIR / "y_train_page.npy"
X_TEST_PATH = FEATURE_DIR / "X_test_page.npy"
Y_TEST_PATH = FEATURE_DIR / "y_test_page.npy"

N_CLUSTERS = 40
PCA_COMPONENTS = 100
RANDOM_STATE = 42


def evaluate(method, predictions, X_test_pca, y_test):
    occupied = len(np.unique(predictions))

    silhouette = (
        silhouette_score(X_test_pca, predictions)
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

    # Load arrays
    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH, allow_pickle=True).ravel()
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH, allow_pickle=True).ravel()

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and label counts do not match.")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train/test feature dimensions do not match.")

    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Features contain NaN or infinite values.")

    print("\n========== DATASET INFO ==========")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples    : {len(X_test)}")
    print(f"Feature count   : {X_train.shape[1]}")
    print("==================================")

    # Fit preprocessing on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(
        n_components=PCA_COMPONENTS,
        random_state=RANDOM_STATE,
    )
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(
        f"\nPCA explained variance: "
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )

    # Fit Ward clustering on training data only
    ward = AgglomerativeClustering(
        n_clusters=N_CLUSTERS,
        linkage="ward",
    )
    train_clusters = ward.fit_predict(X_train_pca)

    print(f"Training clusters found: {len(np.unique(train_clusters))}")

    # --------------------------------------------------
    # STRATEGY 1: NEAREST TRAINING SAMPLE
    # --------------------------------------------------

    nearest = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nearest.fit(X_train_pca)

    distances, nearest_indices = nearest.kneighbors(X_test_pca)

    nearest_sample_predictions = train_clusters[nearest_indices[:, 0]]

    # --------------------------------------------------
    # STRATEGY 2: NEAREST CLUSTER CENTROID
    # --------------------------------------------------

    cluster_ids = np.unique(train_clusters)

    centroids = np.vstack([
        X_train_pca[train_clusters == cluster_id].mean(axis=0)
        for cluster_id in cluster_ids
    ])

    centroid_search = NearestNeighbors(
        n_neighbors=1,
        metric="euclidean",
    )
    centroid_search.fit(centroids)

    _, centroid_indices = centroid_search.kneighbors(X_test_pca)

    centroid_predictions = cluster_ids[centroid_indices[:, 0]]

    # --------------------------------------------------
    # EVALUATION
    # --------------------------------------------------

    results = [
        evaluate(
            "Nearest_training_sample",
            nearest_sample_predictions,
            X_test_pca,
            y_test,
        ),
        evaluate(
            "Nearest_cluster_centroid",
            centroid_predictions,
            X_test_pca,
            y_test,
        ),
    ]

    results_df = pd.DataFrame(results)

    assignments = pd.DataFrame({
        "sample_index": np.arange(len(y_test)),
        "true_writer": y_test,
        "nearest_training_sample_cluster": nearest_sample_predictions,
        "nearest_training_sample_distance": distances[:, 0],
        "nearest_centroid_cluster": centroid_predictions,
    })

    # Save evaluation artifacts
    metrics_path = RESULT_DIR / "08_agglomerative_test_metrics.csv"
    assignments_path = RESULT_DIR / "08_agglomerative_test_assignments.csv"
    model_path = RESULT_DIR / "08_agglomerative_test_bundle.joblib"

    results_df.to_csv(metrics_path, index=False)
    assignments.to_csv(assignments_path, index=False)

    joblib.dump({
        "scaler": scaler,
        "pca": pca,
        "ward_training_labels": train_clusters,
        "X_train_pca": X_train_pca,
        "cluster_ids": cluster_ids,
        "cluster_centroids": centroids,
        "nearest_training_sample_model": nearest,
        "nearest_centroid_model": centroid_search,
    }, model_path)

    # Print results
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
    print(model_path)
    print("\nEvaluation completed.")


if __name__ == "__main__":
    main()