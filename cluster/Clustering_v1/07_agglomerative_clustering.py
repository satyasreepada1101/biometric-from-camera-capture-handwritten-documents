from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
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

ROOT = Path(__file__).resolve().parent.parent
FEATURE_DIR = ROOT / "features"
RESULT_DIR = ROOT / "Clustering" / "clustering_results"

X_PATH = FEATURE_DIR / "X_train_page.npy"
Y_PATH = FEATURE_DIR / "y_train_page.npy"

N_CLUSTERS = 40
PCA_COMPONENTS = 100
LINKAGES = ["ward", "average", "complete"]


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    X = np.load(X_PATH)
    y = np.load(Y_PATH, allow_pickle=True).ravel()

    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("Invalid feature matrix or label dimensions.")

    if not np.isfinite(X).all():
        raise ValueError("Features contain NaN or infinite values.")

    if N_CLUSTERS >= len(X):
        raise ValueError("Number of clusters must be less than samples.")

    print("\n========== DATASET INFO ==========")
    print(f"Samples       : {len(X)}")
    print(f"Features      : {X.shape[1]}")
    print(f"Writers       : {len(np.unique(y))}")
    print(f"Target clusters: {N_CLUSTERS}")
    print("==================================")

    # Fit preprocessing on training pages.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    feature_sets = {
        "Original_scaled": X_scaled,
        f"PCA_{PCA_COMPONENTS}": X_pca,
    }

    print(
        f"\nPCA explained variance: "
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )

    results = []
    assignment_data = pd.DataFrame({
        "sample_index": np.arange(len(y)),
        "true_writer": y,
    })

    for feature_name, X_features in feature_sets.items():
        for linkage in LINKAGES:
            print(f"\nRunning {feature_name}, linkage={linkage}")

            model = AgglomerativeClustering(
                n_clusters=N_CLUSTERS,
                linkage=linkage,
            )

            cluster_ids = model.fit_predict(X_features)
            occupied = len(np.unique(cluster_ids))

            silhouette = silhouette_score(
                X_features,
                cluster_ids,
            )

            ari = adjusted_rand_score(y, cluster_ids)
            nmi = normalized_mutual_info_score(y, cluster_ids)

            results.append({
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

            column_name = f"{feature_name}_{linkage}_cluster"
            assignment_data[column_name] = cluster_ids

            print(f"Clusters found: {occupied}")
            print(f"Silhouette    : {silhouette:.4f}")
            print(f"ARI           : {ari:.4f}")
            print(f"NMI           : {nmi:.4f}")

    results_df = pd.DataFrame(results)

    results_path = RESULT_DIR / "07_agglomerative_comparison.csv"
    assignments_path = RESULT_DIR / "07_agglomerative_assignments.csv"

    results_df.to_csv(results_path, index=False)
    assignment_data.to_csv(assignments_path, index=False)

    print("\n========== RESULTS RANKED BY ARI ==========")
    print(
        results_df.sort_values(
            "adjusted_rand_index",
            ascending=False,
        ).to_string(index=False)
    )

    print("\nSaved results:")
    print(results_path)
    print(assignments_path)


if __name__ == "__main__":
    main()