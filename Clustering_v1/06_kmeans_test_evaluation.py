"""
Held-out Test Evaluation: K-Means Clustering
---------------------------------------------
Compares:
1. StandardScaler + K-Means (K=40)
2. StandardScaler + PCA(100) + K-Means (K=40)

Models are fitted only on training pages.
Test pages are assigned to the nearest learned cluster.

Inputs:
    features/X_train_page.npy
    features/y_train_page.npy
    features/X_test_page.npy
    features/y_test_page.npy

Outputs:
    Clustering/clustering_results/06_test_evaluation_metrics.csv
    Clustering/clustering_results/06_test_cluster_assignments.csv
    Clustering/clustering_results/06_kmeans_test_models.joblib
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = PROJECT_ROOT / "features"
RESULT_DIR = PROJECT_ROOT / "Clustering" / "clustering_results"

X_TRAIN_PATH = FEATURE_DIR / "X_train_page.npy"
Y_TRAIN_PATH = FEATURE_DIR / "y_train_page.npy"
X_TEST_PATH = FEATURE_DIR / "X_test_page.npy"
Y_TEST_PATH = FEATURE_DIR / "y_test_page.npy"

N_CLUSTERS = 40
PCA_COMPONENTS = 100
RANDOM_STATE = 42
N_INIT = 20


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():
    X_train = np.load(X_TRAIN_PATH)
    y_train = np.load(Y_TRAIN_PATH, allow_pickle=True).ravel()

    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH, allow_pickle=True).ravel()

    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Training and test features must be 2D matrices.")

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "Training and test feature dimensions do not match."
        )

    if len(X_train) != len(y_train):
        raise ValueError("Training features and labels have different lengths.")

    if len(X_test) != len(y_test):
        raise ValueError("Test features and labels have different lengths.")

    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Features contain NaN or infinite values.")

    print("\n========== DATASET INFO ==========")
    print(f"Training features: {X_train.shape}")
    print(f"Test features    : {X_test.shape}")
    print(f"Training writers : {len(np.unique(y_train))}")
    print(f"Test writers     : {len(np.unique(y_test))}")
    print("==================================\n")

    return X_train, y_train, X_test, y_test


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

def evaluate_test_clusters(
    method_name,
    X_test_transformed,
    y_test,
    predicted_clusters,
):
    occupied = len(np.unique(predicted_clusters))

    # Silhouette requires at least two clusters and fewer clusters
    # than the number of test samples.
    if 1 < occupied < len(predicted_clusters):
        silhouette = silhouette_score(
            X_test_transformed,
            predicted_clusters,
        )
    else:
        silhouette = np.nan

    return {
        "method": method_name,
        "test_samples": len(y_test),
        "n_clusters_predicted": occupied,
        "test_silhouette": silhouette,
        "test_ari": adjusted_rand_score(
            y_test, predicted_clusters
        ),
        "test_nmi": normalized_mutual_info_score(
            y_test, predicted_clusters
        ),
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_test, y_test = load_data()

    if len(X_train) <= N_CLUSTERS:
        raise ValueError(
            "Training sample count must exceed the number of clusters."
        )

    if PCA_COMPONENTS >= min(X_train.shape):
        raise ValueError(
            f"PCA_COMPONENTS={PCA_COMPONENTS} is too large "
            f"for training shape {X_train.shape}."
        )

    # Fit the scaler using training data only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []
    assignments = pd.DataFrame({
        "sample_index": np.arange(len(y_test)),
        "true_writer": y_test,
    })

    model_bundles = {}

    # --------------------------------------------------
    # MODEL 1: ORIGINAL SCALED FEATURES
    # --------------------------------------------------

    print("Training K-Means on original scaled features...")

    original_kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )

    original_kmeans.fit(X_train_scaled)
    original_predictions = original_kmeans.predict(X_test_scaled)

    original_result = evaluate_test_clusters(
        "Original_scaled",
        X_test_scaled,
        y_test,
        original_predictions,
    )

    results.append(original_result)
    assignments["original_scaled_cluster"] = original_predictions

    model_bundles["Original_scaled"] = {
        "scaler": scaler,
        "pca": None,
        "kmeans": original_kmeans,
    }

    # --------------------------------------------------
    # MODEL 2: PCA-100 FEATURES
    # --------------------------------------------------

    print("Training PCA + K-Means on training features...")

    pca = PCA(
        n_components=PCA_COMPONENTS,
        random_state=RANDOM_STATE,
    )

    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(
        f"PCA components: {PCA_COMPONENTS}\n"
        f"Training explained variance: "
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )

    pca_kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )

    pca_kmeans.fit(X_train_pca)
    pca_predictions = pca_kmeans.predict(X_test_pca)

    pca_result = evaluate_test_clusters(
        f"PCA_{PCA_COMPONENTS}",
        X_test_pca,
        y_test,
        pca_predictions,
    )

    results.append(pca_result)
    assignments["pca_cluster"] = pca_predictions

    model_bundles[f"PCA_{PCA_COMPONENTS}"] = {
        "scaler": scaler,
        "pca": pca,
        "kmeans": pca_kmeans,
    }

    # --------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------

    metrics_df = pd.DataFrame(results)

    metrics_path = RESULT_DIR / "06_test_evaluation_metrics.csv"
    assignments_path = RESULT_DIR / "06_test_cluster_assignments.csv"
    models_path = RESULT_DIR / "06_kmeans_test_models.joblib"

    metrics_df.to_csv(metrics_path, index=False)
    assignments.to_csv(assignments_path, index=False)
    joblib.dump(model_bundles, models_path)

    print("\n========== HELD-OUT TEST RESULTS ==========")

    for result in results:
        print(f"\nMethod             : {result['method']}")
        print(f"Test samples       : {result['test_samples']}")
        print(f"Occupied clusters  : {result['n_clusters_predicted']}")
        print(f"Test silhouette    : {result['test_silhouette']:.4f}")
        print(f"Test ARI           : {result['test_ari']:.4f}")
        print(f"Test NMI           : {result['test_nmi']:.4f}")

    print("\n===========================================")
    print(f"Metrics     : {metrics_path}")
    print(f"Assignments : {assignments_path}")
    print(f"Saved models: {models_path}")
    print("\nTest evaluation completed.")


if __name__ == "__main__":
    main()