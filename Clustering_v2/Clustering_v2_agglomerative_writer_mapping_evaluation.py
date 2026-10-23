"""
Clustering_v2_agglomerative_writer_mapping_evaluation.py

Evaluates writer prediction from V2 Ward Agglomerative clusters using a
cluster-to-writer mapping learned from TRAIN labels only.

Uses raw page-level AlexNet features -> PCA retaining 95% variance.
Fits PCA and Agglomerative clustering on training pages only.
Assigns test pages by nearest training sample or nearest cluster centroid.
The test labels are used only for final scoring.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import NearestNeighbors

BASE_DIR = Path(__file__).resolve().parent
FEATURE_DIR = BASE_DIR / "features"
RESULT_DIR = BASE_DIR / "clustering_results"
N_CLUSTERS = 40
RANDOM_STATE = 42


def find_writer_column(df):
    for candidate in ("writer_id", "writer", "true_writer", "label", "y"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not find writer label column. Available columns: "
        + ", ".join(map(str, df.columns))
    )


def main():
    paths = {
        "X_train": FEATURE_DIR / "X_train_page.npy",
        "y_train": FEATURE_DIR / "y_train_page.npy",
        "X_test": FEATURE_DIR / "X_test_page.npy",
        "y_test": FEATURE_DIR / "y_test_page.npy",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required page feature files:\n" + "\n".join(missing))

    X_train = np.load(paths["X_train"]).astype(np.float64)
    y_train = np.load(paths["y_train"]).ravel()
    X_test = np.load(paths["X_test"]).astype(np.float64)
    y_test = np.load(paths["y_test"]).ravel()

    train_meta_path = FEATURE_DIR / "train_page_metadata.csv"
    test_meta_path = FEATURE_DIR / "test_page_metadata.csv"
    train_meta = pd.read_csv(train_meta_path) if train_meta_path.exists() else pd.DataFrame()
    test_meta = pd.read_csv(test_meta_path) if test_meta_path.exists() else pd.DataFrame()

    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Feature and label counts do not match.")

    print("=" * 68)
    print("AGGLOMERATIVE — TRAIN-DERIVED WRITER MAPPING EVALUATION")
    print("=" * 68)
    print("PCA and Ward clustering are fitted on training pages only.")
    print("Writer labels are used to map training clusters and score test predictions.")
    print(f"Train pages: {len(X_train)}; test pages: {len(X_test)}")

    pca = PCA(n_components=0.95, svd_solver="full", random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    ward = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward")
    train_clusters = ward.fit_predict(X_train_pca)

    # Map each training cluster to its most common training writer.
    mapping_frame = pd.DataFrame({"cluster_id": train_clusters, "writer": y_train})
    cluster_to_writer = (
        mapping_frame.groupby("cluster_id")["writer"]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )

    # Strategy A: cluster of the nearest training page.
    nearest = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(X_train_pca)
    distances, indices = nearest.kneighbors(X_test_pca)
    nearest_sample_clusters = train_clusters[indices[:, 0]]

    # Strategy B: cluster of the nearest training cluster centroid.
    cluster_ids = np.unique(train_clusters)
    centroids = np.vstack([
        X_train_pca[train_clusters == cid].mean(axis=0) for cid in cluster_ids
    ])
    centroid_search = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(centroids)
    _, centroid_indices = centroid_search.kneighbors(X_test_pca)
    centroid_clusters = cluster_ids[centroid_indices[:, 0]]

    pred_a = pd.Series(nearest_sample_clusters).map(cluster_to_writer).to_numpy()
    pred_b = pd.Series(centroid_clusters).map(cluster_to_writer).to_numpy()

    print(f"PCA dimensions: {X_train_pca.shape[1]}")
    print(f"PCA variance retained: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Training clusters: {len(np.unique(train_clusters))}")
    print(f"\nNearest training sample writer accuracy: {accuracy_score(y_test, pred_a):.4f}")
    print(f"Nearest cluster centroid writer accuracy: {accuracy_score(y_test, pred_b):.4f}")

    print("\n--- Nearest training sample classification report ---")
    print(classification_report(y_test, pred_a, zero_division=0))
    print("\n--- Nearest centroid classification report ---")
    print(classification_report(y_test, pred_b, zero_division=0))

    assignments = pd.DataFrame({
        "sample_index": np.arange(len(y_test)),
        "true_writer": y_test,
        "nearest_training_sample_cluster": nearest_sample_clusters,
        "nearest_training_sample_predicted_writer": pred_a,
        "nearest_training_sample_distance": distances[:, 0],
        "nearest_centroid_cluster": centroid_clusters,
        "nearest_centroid_predicted_writer": pred_b,
    })
    # Include useful metadata if available.
    if not test_meta.empty:
        for col in test_meta.columns:
            if col not in assignments.columns and len(test_meta) == len(assignments):
                assignments[col] = test_meta[col].values

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = RESULT_DIR / "agglomerative_train_mapped_writer_predictions.csv"
    map_path = RESULT_DIR / "agglomerative_train_cluster_writer_mapping.csv"
    assignments.to_csv(pred_path, index=False)

    map_rows = []
    for cid, group in mapping_frame.groupby("cluster_id"):
        counts = group["writer"].value_counts()
        map_rows.append({
            "cluster_id": cid,
            "mapped_writer_from_train": cluster_to_writer[cid],
            "train_pages_in_cluster": len(group),
            "distinct_train_writers_in_cluster": group["writer"].nunique(),
            "majority_writer_train_pages": int(counts.iloc[0]),
            "majority_share_in_cluster": float(counts.iloc[0] / len(group)),
        })
    pd.DataFrame(map_rows).to_csv(map_path, index=False)

    print("\nSaved predictions:", pred_path)
    print("Saved train-derived mapping:", map_path)
    print("\nInterpretation: this is a simple cluster-to-writer mapping baseline,")
    print("not a fully supervised writer classifier.")


if __name__ == "__main__":
    main()
