"""
Clustering_v2_kmeans.py

Runs K-Means on page-level AlexNet features created by
Clustering_v2_aggregate_page_features.py.

Important:
- K-Means is fit using TRAIN page features only.
- Test pages are transformed using train-fitted preprocessing and assigned
  to the trained clusters.
- Writer labels are used ONLY to evaluate the clusters, never to fit K-Means
  or choose the recommended K.
- The recommended K is selected by the training silhouette score.
"""

from pathlib import Path
import warnings

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
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore", category=RuntimeWarning)

BASE_DIR = Path(__file__).resolve().parent
FEATURE_DIR = BASE_DIR / "features"
RESULT_DIR = BASE_DIR / "clustering_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
# Includes 40 (one expected cluster per writer) but also tests alternatives.
K_VALUES = [2, 5, 10, 20, 30, 40, 50]


def clustering_accuracy(y_true, cluster_ids):
    """Best one-to-one mapping between arbitrary cluster IDs and writer IDs."""
    y_true = np.asarray(y_true)
    cluster_ids = np.asarray(cluster_ids)

    true_values, true_inverse = np.unique(y_true, return_inverse=True)
    cluster_values, cluster_inverse = np.unique(cluster_ids, return_inverse=True)

    contingency = np.zeros(
        (len(true_values), len(cluster_values)), dtype=np.int64
    )
    np.add.at(contingency, (true_inverse, cluster_inverse), 1)

    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    matched = contingency[row_ind, col_ind].sum()
    return float(matched / len(y_true))


def safe_silhouette(X, cluster_ids):
    """Silhouette is undefined if there is only one cluster or one per sample."""
    labels = np.asarray(cluster_ids)
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        return np.nan
    # Exact silhouette is manageable for this project's page counts.
    return float(silhouette_score(X, labels, metric="euclidean"))


def evaluate(y_true, cluster_ids):
    return {
        "ARI": adjusted_rand_score(y_true, cluster_ids),
        "NMI": normalized_mutual_info_score(y_true, cluster_ids),
        "Clustering_Accuracy": clustering_accuracy(y_true, cluster_ids),
    }


def main():
    required_files = [
        "X_train_page.npy",
        "y_train_page.npy",
        "X_test_page.npy",
        "y_test_page.npy",
        "train_page_metadata.csv",
        "test_page_metadata.csv",
    ]
    missing = [name for name in required_files if not (FEATURE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required page-level files in Clustering_v2/features:\n- "
            + "\n- ".join(missing)
            + "\nRun Clustering_v2_aggregate_page_features.py first."
        )

    X_train = np.load(FEATURE_DIR / "X_train_page.npy")
    y_train = np.load(FEATURE_DIR / "y_train_page.npy").ravel()
    X_test = np.load(FEATURE_DIR / "X_test_page.npy")
    y_test = np.load(FEATURE_DIR / "y_test_page.npy").ravel()
    train_meta = pd.read_csv(FEATURE_DIR / "train_page_metadata.csv")
    test_meta = pd.read_csv(FEATURE_DIR / "test_page_metadata.csv")

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and label counts do not match.")
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("Expected 2D page feature arrays.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train and test feature dimensions do not match.")
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training data must contain at least two writers.")

    print("=" * 64)
    print("CLUSTERING_V2 — K-MEANS ON PAGE-LEVEL ALEXNET FEATURES")
    print("=" * 64)
    print(f"Train pages/features : {X_train.shape}")
    print(f"Test pages/features  : {X_test.shape}")
    print(f"Unique train writers : {len(np.unique(y_train))}")
    print(f"Unique test writers  : {len(np.unique(y_test))}")
    print("\nPreprocessing: StandardScaler + PCA (95% variance), fitted on train only.")

    # Fit preprocessing ONLY on training data to prevent test-data leakage.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=0.95, svd_solver="full")
    X_train_ready = pca.fit_transform(X_train_scaled)
    X_test_ready = pca.transform(X_test_scaled)

    print(f"PCA dimensions       : {X_train_ready.shape[1]}")
    print(f"Variance retained    : {pca.explained_variance_ratio_.sum():.4f}")

    valid_k = [k for k in K_VALUES if 2 <= k < len(X_train)]
    rows = []
    fitted_models = {}

    print("\nTesting K values:", valid_k)
    print("Silhouette selects the recommended K; labels are evaluation-only.\n")

    for k in valid_k:
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=20,
            max_iter=500,
        )
        train_clusters = model.fit_predict(X_train_ready)
        test_clusters = model.predict(X_test_ready)
        fitted_models[k] = model

        row = {
            "K": k,
            "Train_Silhouette": safe_silhouette(X_train_ready, train_clusters),
            "Train_ARI": adjusted_rand_score(y_train, train_clusters),
            "Train_NMI": normalized_mutual_info_score(y_train, train_clusters),
            "Train_Clustering_Accuracy": clustering_accuracy(y_train, train_clusters),
            "Test_ARI": adjusted_rand_score(y_test, test_clusters),
            "Test_NMI": normalized_mutual_info_score(y_test, test_clusters),
            "Test_Clustering_Accuracy": clustering_accuracy(y_test, test_clusters),
            "Train_Clusters_Actually_Used": len(np.unique(train_clusters)),
            "Test_Clusters_Used": len(np.unique(test_clusters)),
        }
        rows.append(row)
        print(
            f"K={k:>2} | train silhouette={row['Train_Silhouette']:.4f} "
            f"| test ARI={row['Test_ARI']:.4f} "
            f"| test NMI={row['Test_NMI']:.4f} "
            f"| test clustering accuracy={row['Test_Clustering_Accuracy']:.4f}"
        )

    results = pd.DataFrame(rows)
    results_path = RESULT_DIR / "kmeans_comparison.csv"
    results.to_csv(results_path, index=False)

    # Choose K using only the training silhouette score, not writer labels.
    usable = results.dropna(subset=["Train_Silhouette"])
    if usable.empty:
        raise RuntimeError("Could not calculate a valid training silhouette score.")
    best_row = usable.loc[usable["Train_Silhouette"].idxmax()]
    best_k = int(best_row["K"])
    best_model = fitted_models[best_k]

    # Save the selected model's cluster assignments. These are cluster IDs,
    # not writer IDs.
    train_clusters = best_model.predict(X_train_ready)
    test_clusters = best_model.predict(X_test_ready)

    train_out = train_meta.copy()
    test_out = test_meta.copy()
    train_out["cluster_id"] = train_clusters
    test_out["cluster_id"] = test_clusters
    train_out.to_csv(RESULT_DIR / f"train_clusters_k{best_k}.csv", index=False)
    test_out.to_csv(RESULT_DIR / f"test_clusters_k{best_k}.csv", index=False)

    # Save preprocessing and model so the experiment is reproducible.
    import joblib
    joblib.dump(
        {"scaler": scaler, "pca": pca, "kmeans": best_model, "selected_k": best_k},
        RESULT_DIR / "best_kmeans_pipeline.joblib",
    )

    print("\n" + "=" * 64)
    print(f"RECOMMENDED K (training silhouette only): {best_k}")
    print(f"Training silhouette                 : {best_row['Train_Silhouette']:.4f}")
    print(f"Comparison table saved              : {results_path}")
    print(f"Train cluster assignments saved     : {RESULT_DIR / f'train_clusters_k{best_k}.csv'}")
    print(f"Test cluster assignments saved      : {RESULT_DIR / f'test_clusters_k{best_k}.csv'}")
    print(f"Model pipeline saved                : {RESULT_DIR / 'best_kmeans_pipeline.joblib'}")
    print("\nReminder: cluster IDs are arbitrary and do not directly equal writer IDs.")
    print("Use test metrics as the final held-out evaluation; do not tune K on them.")


if __name__ == "__main__":
    main()
