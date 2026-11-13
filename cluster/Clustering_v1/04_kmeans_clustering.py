
"""
K-Means clustering on pretrained AlexNet page-level features.

Input:
    features/X_train_page.npy
    features/y_train_page.npy

Output:
    Clustering/clustering_results/
        04_kmeans_metrics.csv
        04_kmeans_cluster_assignments.csv
        04_kmeans_cluster_distribution.csv
        04_kmeans_cluster_count_comparison.csv
        04_kmeans_pca_visualization.png
        04_kmeans_silhouette_scores.png
        04_kmeans_model.joblib
        04_kmeans_scaler.joblib
        04_kmeans_pca.joblib

Run from any working directory:
    python Clustering/04_kmeans_clustering.py
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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
# 1. PATHS AND SETTINGS
# --------------------------------------------------

# Project root is the parent of this script's folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_DIR = PROJECT_ROOT / "features"
OUTPUT_DIR = PROJECT_ROOT / "Clustering" / "clustering_results"

X_TRAIN_PATH = FEATURE_DIR / "X_train_page.npy"
Y_TRAIN_PATH = FEATURE_DIR / "y_train_page.npy"

# Your dataset has approximately 40 known writers.
N_CLUSTERS = 40

RANDOM_STATE = 42
N_INIT = 20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD FEATURES AND LABELS
# --------------------------------------------------

def load_data():
    if not X_TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Feature file not found: {X_TRAIN_PATH}"
        )

    if not Y_TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Label file not found: {Y_TRAIN_PATH}"
        )

    X = np.load(X_TRAIN_PATH, allow_pickle=False)
    y = np.load(Y_TRAIN_PATH, allow_pickle=True)

    # Convert labels to a one-dimensional array.
    y = np.asarray(y).reshape(-1)

    if X.ndim != 2:
        raise ValueError(
            f"Expected a 2D feature matrix, got shape {X.shape}"
        )

    if len(X) != len(y):
        raise ValueError(
            f"Feature/label count mismatch: {len(X)} vs {len(y)}"
        )

    if len(X) <= N_CLUSTERS:
        raise ValueError(
            f"Need more than {N_CLUSTERS} samples; got {len(X)}."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Features contain NaN or infinite values."
        )

    print("\n========== DATASET INFO ==========")
    print(f"Feature matrix shape : {X.shape}")
    print(f"Number of samples    : {len(X)}")
    print(f"Number of features   : {X.shape[1]}")
    print(f"Number of writers    : {len(np.unique(y))}")
    print("==================================\n")

    return X, y


# --------------------------------------------------
# 3. SCALE FEATURES
# --------------------------------------------------

def scale_features(X):
    """
    Standardize each feature so that large-scale
    features do not dominate Euclidean distances.
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, OUTPUT_DIR / "04_kmeans_scaler.joblib")

    return X_scaled, scaler


# --------------------------------------------------
# 4. RUN K-MEANS
# --------------------------------------------------

def run_kmeans(X_scaled):
    """
    K-Means is fitted without writer labels.
    """

    model = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )

    cluster_labels = model.fit_predict(X_scaled)

    joblib.dump(model, OUTPUT_DIR / "04_kmeans_model.joblib")

    print("========== K-MEANS RESULTS ==========")
    print(f"Clusters requested : {N_CLUSTERS}")
    print(f"Clusters occupied  : {len(np.unique(cluster_labels))}")
    print(f"Inertia            : {model.inertia_:.4f}")
    print(f"Iterations         : {model.n_iter_}")
    print("=====================================\n")

    return model, cluster_labels


# --------------------------------------------------
# 5. EVALUATE CLUSTERING
# --------------------------------------------------

def evaluate_clustering(X_scaled, y_true, cluster_labels):
    """
    Silhouette score measures cluster separation.

    ARI and NMI compare the cluster assignments
    with the known writer labels AFTER clustering.
    """

    n_samples = len(cluster_labels)
    n_clusters_found = len(np.unique(cluster_labels))

    if 1 < n_clusters_found < n_samples:
        silhouette = silhouette_score(
            X_scaled,
            cluster_labels,
            sample_size=min(2000, n_samples),
            random_state=RANDOM_STATE,
        )
    else:
        silhouette = np.nan

    ari = adjusted_rand_score(y_true, cluster_labels)
    nmi = normalized_mutual_info_score(y_true, cluster_labels)

    metrics = pd.DataFrame(
        [
            {
                "algorithm": "KMeans",
                "n_samples": n_samples,
                "n_features": X_scaled.shape[1],
                "n_clusters_requested": N_CLUSTERS,
                "n_clusters_found": n_clusters_found,
                "inertia": float(np.sum(
                    (X_scaled - np.vstack([
                        np.mean(
                            X_scaled[cluster_labels == c], axis=0
                        )
                        for c in cluster_labels
                    ])[:1]) ** 2
                )) if False else np.nan,
                "silhouette_score": silhouette,
                "adjusted_rand_index": ari,
                "normalized_mutual_info": nmi,
            }
        ]
    )

    # Use the actual fitted model's inertia separately below.
    metrics.to_csv(
        OUTPUT_DIR / "04_kmeans_metrics.csv",
        index=False,
    )

    print("========== EVALUATION ==========")
    print(f"Silhouette score : {silhouette:.4f}")
    print(f"Adjusted Rand    : {ari:.4f}")
    print(f"Normalized MI    : {nmi:.4f}")
    print("================================\n")

    return metrics


# --------------------------------------------------
# 6. SAVE CLUSTER ASSIGNMENTS
# --------------------------------------------------

def save_assignments(y_true, cluster_labels):
    assignments = pd.DataFrame({
        "sample_index": np.arange(len(cluster_labels)),
        "true_writer": y_true.astype(str),
        "cluster_id": cluster_labels,
    })

    assignments.to_csv(
        OUTPUT_DIR / "04_kmeans_cluster_assignments.csv",
        index=False,
    )

    # How many samples from each known writer
    # fall into each cluster?
    distribution = pd.crosstab(
        assignments["cluster_id"],
        assignments["true_writer"],
    )

    distribution.to_csv(
        OUTPUT_DIR / "04_kmeans_cluster_distribution.csv"
    )

    print("Cluster distribution:")
    print(distribution)
    print()

    return assignments, distribution


# --------------------------------------------------
# 7. VISUALIZE CLUSTERS USING PCA
# --------------------------------------------------

def plot_clusters(X_scaled, cluster_labels):
    """
    Reduce the features to two dimensions for plotting.
    PCA is only for visualization here; K-Means was
    trained on the full scaled feature matrix.
    """

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_scaled)

    joblib.dump(pca, OUTPUT_DIR / "04_kmeans_pca.joblib")

    plt.figure(figsize=(12, 8))

    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=cluster_labels,
        cmap="tab20",
        s=55,
        alpha=0.8,
    )

    plt.title("K-Means Clustering of AlexNet Page Features")
    plt.xlabel(
        f"Principal Component 1 "
        f"({pca.explained_variance_ratio_[0] * 100:.1f}% variance)"
    )
    plt.ylabel(
        f"Principal Component 2 "
        f"({pca.explained_variance_ratio_[1] * 100:.1f}% variance)"
    )
    plt.colorbar(scatter, label="Cluster ID")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "04_kmeans_pca_visualization.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("PCA visualization saved.")


# --------------------------------------------------
# 8. COMPARE SILHOUETTE SCORES FOR DIFFERENT K
# --------------------------------------------------

def compare_cluster_counts(X_scaled):
    """
    Compare several candidate cluster counts.
    This is exploratory and uses no writer labels.
    """

    candidate_ks = [2, 5, 10, 20, 30, 40]
    results = []

    n_samples = len(X_scaled)

    for k in candidate_ks:
        if k >= n_samples:
            continue

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=N_INIT,
        )

        labels = model.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels,
            sample_size=min(2000, n_samples),
            random_state=RANDOM_STATE,
        )

        results.append({
            "n_clusters": k,
            "silhouette_score": score,
            "inertia": model.inertia_,
        })

        print(f"K={k:2d} | silhouette={score:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        OUTPUT_DIR / "04_kmeans_cluster_count_comparison.csv",
        index=False,
    )

    if not results_df.empty:
        plt.figure(figsize=(9, 5))
        plt.plot(
            results_df["n_clusters"],
            results_df["silhouette_score"],
            marker="o",
        )
        plt.xlabel("Number of clusters (K)")
        plt.ylabel("Silhouette score")
        plt.title("K-Means: Choosing the Number of Clusters")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            OUTPUT_DIR / "04_kmeans_silhouette_scores.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    return results_df


# --------------------------------------------------
# 9. MAIN
# --------------------------------------------------

def main():
    X, y = load_data()

    X_scaled, scaler = scale_features(X)

    model, cluster_labels = run_kmeans(X_scaled)

    metrics = evaluate_clustering(
        X_scaled,
        y,
        cluster_labels,
    )

    # Store actual inertia from the fitted model.
    metrics.loc[0, "inertia"] = model.inertia_
    metrics.to_csv(
        OUTPUT_DIR / "04_kmeans_metrics.csv",
        index=False,
    )

    save_assignments(y, cluster_labels)

    plot_clusters(X_scaled, cluster_labels)

    print("\nComparing candidate values of K:")
    compare_cluster_counts(X_scaled)

    print("\nK-Means clustering completed.")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
