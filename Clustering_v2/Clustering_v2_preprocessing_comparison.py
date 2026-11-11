"""
Clustering_v2_preprocessing_comparison.py

Diagnostic experiment comparing preprocessing choices for K-Means at K=40.
All transformations are fitted on training features only. Writer labels are
used only to calculate evaluation metrics, never to fit transformations or K-Means.
No option is automatically declared the winner based on test labels.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler, normalize
from scipy.optimize import linear_sum_assignment

BASE_DIR = Path(__file__).resolve().parent
FEATURE_DIR = BASE_DIR / "features"
RESULT_DIR = BASE_DIR / "clustering_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
K = 40
RANDOM_STATE = 42


def clustering_accuracy(y_true, cluster_ids):
    y_true = np.asarray(y_true)
    cluster_ids = np.asarray(cluster_ids)
    true_values, true_inverse = np.unique(y_true, return_inverse=True)
    cluster_values, cluster_inverse = np.unique(cluster_ids, return_inverse=True)
    table = np.zeros((len(true_values), len(cluster_values)), dtype=np.int64)
    np.add.at(table, (true_inverse, cluster_inverse), 1)
    r, c = linear_sum_assignment(table.max() - table)
    return float(table[r, c].sum() / len(y_true))


def purity(y_true, cluster_ids):
    frame = pd.DataFrame({"writer": y_true, "cluster": cluster_ids})
    counts = frame.groupby("cluster")["writer"].value_counts()
    return float(counts.groupby(level=0).max().sum() / len(frame))


def silhouette_or_nan(X, labels):
    if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(labels):
        return np.nan
    return float(silhouette_score(X, labels))


def main():
    required = [
        "X_train_page.npy", "y_train_page.npy",
        "X_test_page.npy", "y_test_page.npy",
        "train_page_metadata.csv", "test_page_metadata.csv",
    ]
    missing = [f for f in required if not (FEATURE_DIR / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing page-level files in Clustering_v2/features: "
            + ", ".join(missing)
        )

    X_train = np.load(FEATURE_DIR / "X_train_page.npy").astype(np.float64)
    y_train = np.load(FEATURE_DIR / "y_train_page.npy").ravel()
    X_test = np.load(FEATURE_DIR / "X_test_page.npy").astype(np.float64)
    y_test = np.load(FEATURE_DIR / "y_test_page.npy").ravel()
    train_meta = pd.read_csv(FEATURE_DIR / "train_page_metadata.csv")
    test_meta = pd.read_csv(FEATURE_DIR / "test_page_metadata.csv")

    print("=" * 68)
    print("PREPROCESSING COMPARISON — FIXED K=40")
    print("=" * 68)
    print("Writer labels are used only for evaluation.")
    print("Each transform is fitted on train pages only.")
    print(f"Train: {X_train.shape}; test: {X_test.shape}")

    variants = {}

    # Variant 1: reproduce the earlier pipeline.
    scaler1 = StandardScaler()
    tr1 = scaler1.fit_transform(X_train)
    te1 = scaler1.transform(X_test)
    pca1 = PCA(n_components=0.95, svd_solver="full")
    variants["standard_scaler_then_pca95"] = (
        pca1.fit_transform(tr1), pca1.transform(te1),
        f"PCA dims={pca1.n_components_}, variance={pca1.explained_variance_ratio_.sum():.4f}"
    )

    # Variant 2: PCA on original feature scale (PCA still centers the data).
    pca2 = PCA(n_components=0.95, svd_solver="full")
    variants["raw_features_then_pca95"] = (
        pca2.fit_transform(X_train), pca2.transform(X_test),
        f"PCA dims={pca2.n_components_}, variance={pca2.explained_variance_ratio_.sum():.4f}"
    )

    # Variant 3: L2-normalize each page vector, then fit PCA on train only.
    tr3 = normalize(X_train, norm="l2")
    te3 = normalize(X_test, norm="l2")
    pca3 = PCA(n_components=0.95, svd_solver="full")
    variants["l2_normalize_then_pca95"] = (
        pca3.fit_transform(tr3), pca3.transform(te3),
        f"PCA dims={pca3.n_components_}, variance={pca3.explained_variance_ratio_.sum():.4f}"
    )

    # Variant 4: StandardScaler + PCA, then L2-normalize the reduced vectors.
    tr4 = normalize(variants["standard_scaler_then_pca95"][0], norm="l2")
    te4 = normalize(variants["standard_scaler_then_pca95"][1], norm="l2")
    variants["standard_scaler_pca95_then_l2"] = (
        tr4, te4, "same PCA dimensions as standard_scaler_then_pca95; reduced vectors L2-normalized"
    )

    rows = []
    for name, (Xt, Xv, details) in variants.items():
        model = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=20, max_iter=500)
        train_clusters = model.fit_predict(Xt)
        test_clusters = model.predict(Xv)

        row = {
            "preprocessing": name,
            "details": details,
            "K": K,
            "train_silhouette": silhouette_or_nan(Xt, train_clusters),
            "train_ARI": adjusted_rand_score(y_train, train_clusters),
            "train_NMI": normalized_mutual_info_score(y_train, train_clusters),
            "train_purity": purity(y_train, train_clusters),
            "test_ARI": adjusted_rand_score(y_test, test_clusters),
            "test_NMI": normalized_mutual_info_score(y_test, test_clusters),
            "test_purity": purity(y_test, test_clusters),
            "test_one_to_one_accuracy": clustering_accuracy(y_test, test_clusters),
            "train_clusters_used": len(np.unique(train_clusters)),
            "test_clusters_used": len(np.unique(test_clusters)),
        }
        rows.append(row)
        print(f"\n{name}")
        print(details)
        print(f"Train silhouette: {row['train_silhouette']:.4f}")
        print(f"Test ARI: {row['test_ARI']:.4f}")
        print(f"Test NMI: {row['test_NMI']:.4f}")
        print(f"Test purity: {row['test_purity']:.4f}")
        print(f"Test one-to-one accuracy: {row['test_one_to_one_accuracy']:.4f}")
        print(f"Clusters used: train={row['train_clusters_used']}, test={row['test_clusters_used']}")

        # Store assignments for inspection; not a claim of a selected winner.
        train_out = train_meta.copy()
        train_out["cluster_id"] = train_clusters
        test_out = test_meta.copy()
        test_out["cluster_id"] = test_clusters
        train_out.to_csv(RESULT_DIR / f"train_{name}_k40.csv", index=False)
        test_out.to_csv(RESULT_DIR / f"test_{name}_k40.csv", index=False)

    output = RESULT_DIR / "preprocessing_comparison_k40.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print("\n" + "=" * 68)
    print("Comparison table saved to:")
    print(output)
    print("\nDo not choose a pipeline solely because its test score is highest.")
    print("Treat this as a diagnostic comparison and discuss the held-out results transparently.")


if __name__ == "__main__":
    main()
