"""
PCA + K-Means Clustering Comparison
-----------------------------------
Compares original AlexNet features with PCA-reduced features.

Input:
    features/X_train_page.npy
    features/y_train_page.npy

Output:
    Clustering/clustering_results/05_pca_kmeans_comparison.csv
    Clustering/clustering_results/05_pca_kmeans_best_model.joblib
    Clustering/clustering_results/05_pca_kmeans_comparison.png
    Clustering/clustering_results/05_pca_<components>_transformer.joblib
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
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = PROJECT_ROOT / "features"
RESULT_DIR = PROJECT_ROOT / "Clustering" / "clustering_results"

X_PATH = FEATURE_DIR / "X_train_page.npy"
Y_PATH = FEATURE_DIR / "y_train_page.npy"

K_VALUES = [2, 5, 10, 20, 30, 40]
PCA_COMPONENTS = [50, 100, 200]

RANDOM_STATE = 42
N_INIT = 20


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_data():
    X = np.load(X_PATH)
    y = np.load(Y_PATH, allow_pickle=True).ravel()

    if X.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got {X.shape}")

    if len(X) != len(y):
        raise ValueError(
            f"Feature/label count mismatch: {len(X)} features, {len(y)} labels"
        )

    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")

    print("\n========== DATASET INFO ==========")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of samples   : {len(X)}")
    print(f"Number of features  : {X.shape[1]}")
    print(f"Number of writers   : {len(np.unique(y))}")
    print("==================================\n")

    return X, y


# --------------------------------------------------
# EVALUATE ONE CONFIGURATION
# --------------------------------------------------

def evaluate_kmeans(X, y, feature_method, k):
    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )

    cluster_ids = model.fit_predict(X)
    occupied = len(np.unique(cluster_ids))

    silhouette = (
        silhouette_score(
            X,
            cluster_ids,
            sample_size=min(2000, len(X)),
            random_state=RANDOM_STATE,
        )
        if 1 < occupied < len(X)
        else np.nan
    )

    return {
        "feature_method": feature_method,
        "n_features": X.shape[1],
        "n_clusters_requested": k,
        "n_clusters_found": occupied,
        "inertia": float(model.inertia_),
        "silhouette_score": float(silhouette),
        "adjusted_rand_index": float(adjusted_rand_score(y, cluster_ids)),
        "normalized_mutual_info": float(
            normalized_mutual_info_score(y, cluster_ids)
        ),
        "model": model,
        "cluster_ids": cluster_ids,
    }


# --------------------------------------------------
# MAIN EXPERIMENT
# --------------------------------------------------

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()

    # Scale once, fitting only on the training set.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    feature_sets = {
        "Original_scaled": X_scaled,
    }

    # PCA is fitted on training data only.
    for n_components in PCA_COMPONENTS:
        if n_components >= min(X_scaled.shape):
            print(
                f"Skipping PCA-{n_components}: "
                "component count is too large for this dataset."
            )
            continue

        pca = PCA(
            n_components=n_components,
            random_state=RANDOM_STATE,
        )

        X_reduced = pca.fit_transform(X_scaled)
        method_name = f"PCA_{n_components}"

        feature_sets[method_name] = X_reduced

        print(
            f"{method_name}: "
            f"{pca.n_components_} components, "
            f"{pca.explained_variance_ratio_.sum():.4f} "
            "total explained variance"
        )

        joblib.dump(
            pca,
            RESULT_DIR / f"05_{method_name.lower()}_transformer.joblib",
        )

    # Run all K values for each feature representation.
    all_results = []
    best_result = None

    for method_name, X_features in feature_sets.items():
        print(f"\n========== {method_name} ==========")

        for k in K_VALUES:
            if k >= len(X_features):
                continue

            result = evaluate_kmeans(X_features, y, method_name, k)

            print(
                f"K={k:2d} | "
                f"Silhouette={result['silhouette_score']:.4f} | "
                f"ARI={result['adjusted_rand_index']:.4f} | "
                f"NMI={result['normalized_mutual_info']:.4f}"
            )

            all_results.append({
                key: value
                for key, value in result.items()
                if key not in ("model", "cluster_ids")
            })

            # Select the best configuration using silhouette score.
            if np.isfinite(result["silhouette_score"]):
                if (
                    best_result is None
                    or result["silhouette_score"]
                    > best_result["silhouette_score"]
                ):
                    best_result = {
                        **result,
                        "scaler": scaler,
                        "pca": (
                            None
                            if method_name == "Original_scaled"
                            else joblib.load(
                                RESULT_DIR
                                / f"05_{method_name.lower()}_transformer.joblib"
                            )
                        ),
                    }

    # Save the comparison table.
    results_df = pd.DataFrame(all_results)
    csv_path = RESULT_DIR / "05_pca_kmeans_comparison.csv"
    results_df.to_csv(csv_path, index=False)

    # Plot silhouette score against K for each feature representation.
    plt.figure(figsize=(10, 6))

    for method_name in feature_sets:
        subset = results_df[
            results_df["feature_method"] == method_name
        ].sort_values("n_clusters_requested")

        plt.plot(
            subset["n_clusters_requested"],
            subset["silhouette_score"],
            marker="o",
            label=method_name,
        )

    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Silhouette score")
    plt.title("K-Means: Original Features vs PCA Features")
    plt.xticks(K_VALUES)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plot_path = RESULT_DIR / "05_pca_kmeans_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    if best_result is not None:
        # Save the best model and all preprocessing needed to transform data.
        best_bundle = {
            "feature_method": best_result["feature_method"],
            "n_clusters": best_result["n_clusters_requested"],
            "scaler": best_result["scaler"],
            "pca": best_result["pca"],
            "kmeans": best_result["model"],
            "silhouette_score": best_result["silhouette_score"],
            "adjusted_rand_index": best_result["adjusted_rand_index"],
            "normalized_mutual_info": best_result["normalized_mutual_info"],
        }

        joblib.dump(
            best_bundle,
            RESULT_DIR / "05_pca_kmeans_best_model.joblib",
        )

        print("\n========== BEST BY SILHOUETTE ==========")
        print(f"Feature method: {best_result['feature_method']}")
        print(f"K             : {best_result['n_clusters_requested']}")
        print(f"Silhouette    : {best_result['silhouette_score']:.4f}")
        print(f"ARI           : {best_result['adjusted_rand_index']:.4f}")
        print(f"NMI           : {best_result['normalized_mutual_info']:.4f}")
        print("========================================")

    print("\nExperiment completed.")
    print(f"CSV results : {csv_path}")
    print(f"Plot        : {plot_path}")
    print(f"Output dir  : {RESULT_DIR}")


if __name__ == "__main__":
    main()