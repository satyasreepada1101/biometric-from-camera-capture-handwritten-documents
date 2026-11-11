"""
Clustering_v2_k40_diagnostics.py

A fixed K=40 baseline for the 40-writer task.
Uses train page features to fit StandardScaler, PCA, and K-Means.
Test writer labels are used only for evaluation, not fitting.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment

BASE_DIR = Path(__file__).resolve().parent
FEATURE_DIR = BASE_DIR / "features"
RESULT_DIR = BASE_DIR / "clustering_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
K = 40


def one_to_one_accuracy(y_true, clusters):
    """Hungarian one-to-one cluster/label match; interpret cautiously for K != #classes."""
    y_true = np.asarray(y_true)
    clusters = np.asarray(clusters)
    true_values, true_inverse = np.unique(y_true, return_inverse=True)
    cluster_values, cluster_inverse = np.unique(clusters, return_inverse=True)
    table = np.zeros((len(true_values), len(cluster_values)), dtype=np.int64)
    np.add.at(table, (true_inverse, cluster_inverse), 1)
    r, c = linear_sum_assignment(table.max() - table)
    return float(table[r, c].sum() / len(y_true))


def purity(y_true, clusters):
    """Fraction of samples that belong to the majority writer in their cluster."""
    frame = pd.DataFrame({"writer": y_true, "cluster": clusters})
    majority_counts = frame.groupby("cluster")["writer"].value_counts()
    return float(majority_counts.groupby(level=0).max().sum() / len(frame))


def main():
    required = [
        "X_train_page.npy", "y_train_page.npy",
        "X_test_page.npy", "y_test_page.npy",
        "train_page_metadata.csv", "test_page_metadata.csv",
    ]
    missing = [f for f in required if not (FEATURE_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing files in Clustering_v2/features: " + ", ".join(missing)
        )

    X_train = np.load(FEATURE_DIR / "X_train_page.npy")
    y_train = np.load(FEATURE_DIR / "y_train_page.npy").ravel()
    X_test = np.load(FEATURE_DIR / "X_test_page.npy")
    y_test = np.load(FEATURE_DIR / "y_test_page.npy").ravel()
    train_meta = pd.read_csv(FEATURE_DIR / "train_page_metadata.csv")
    test_meta = pd.read_csv(FEATURE_DIR / "test_page_metadata.csv")

    print("Fixed K-Means baseline: K=40 (because the dataset has 40 writers)")
    print("Writer labels are not used to fit K-Means.")
    print(f"Train shape: {X_train.shape}; test shape: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    pca = PCA(n_components=0.95, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    model = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=20, max_iter=500)
    train_clusters = model.fit_predict(X_train_pca)
    test_clusters = model.predict(X_test_pca)

    metrics = {
        "K": K,
        "PCA_dimensions": X_train_pca.shape[1],
        "PCA_variance_retained": float(pca.explained_variance_ratio_.sum()),
        "Train_ARI": adjusted_rand_score(y_train, train_clusters),
        "Train_NMI": normalized_mutual_info_score(y_train, train_clusters),
        "Train_Purity": purity(y_train, train_clusters),
        "Train_Homogeneity": homogeneity_score(y_train, train_clusters),
        "Train_Completeness": completeness_score(y_train, train_clusters),
        "Train_V_measure": v_measure_score(y_train, train_clusters),
        "Test_ARI": adjusted_rand_score(y_test, test_clusters),
        "Test_NMI": normalized_mutual_info_score(y_test, test_clusters),
        "Test_Purity": purity(y_test, test_clusters),
        "Test_Homogeneity": homogeneity_score(y_test, test_clusters),
        "Test_Completeness": completeness_score(y_test, test_clusters),
        "Test_V_measure": v_measure_score(y_test, test_clusters),
        "Test_OneToOne_Accuracy": one_to_one_accuracy(y_test, test_clusters),
        "Train_clusters_used": len(np.unique(train_clusters)),
        "Test_clusters_used": len(np.unique(test_clusters)),
    }
    pd.DataFrame([metrics]).to_csv(RESULT_DIR / "k40_diagnostics_metrics.csv", index=False)

    train_out = train_meta.copy()
    train_out["cluster_id"] = train_clusters
    test_out = test_meta.copy()
    test_out["cluster_id"] = test_clusters
    train_out.to_csv(RESULT_DIR / "train_clusters_k40.csv", index=False)
    test_out.to_csv(RESULT_DIR / "test_clusters_k40.csv", index=False)

    # For each test cluster, report its size and writer composition.
    composition = (
        pd.DataFrame({"cluster_id": test_clusters, "writer_id": y_test})
        .groupby(["cluster_id", "writer_id"])
        .size()
        .rename("page_count")
        .reset_index()
        .sort_values(["cluster_id", "page_count"], ascending=[True, False])
    )
    composition.to_csv(RESULT_DIR / "test_k40_cluster_writer_composition.csv", index=False)

    print("\n--- K=40 metrics ---")
    for name, value in metrics.items():
        print(f"{name}: {value}")
    print("\nSaved:")
    print(RESULT_DIR / "k40_diagnostics_metrics.csv")
    print(RESULT_DIR / "train_clusters_k40.csv")
    print(RESULT_DIR / "test_clusters_k40.csv")
    print(RESULT_DIR / "test_k40_cluster_writer_composition.csv")
    print("\nImportant: these are exploratory clustering results, not proof of writer identification.")


if __name__ == "__main__":
    main()
