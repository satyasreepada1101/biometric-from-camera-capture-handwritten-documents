"""
Clustering_v2_evaluate_train_mapped_clusters.py

Evaluates whether K=40 cluster IDs learned from training pages can be mapped
to writer identities using TRAIN labels only, then applied to test pages.

Uses the saved raw_features_then_pca95 K=40 assignments from the preprocessing
comparison. It does not fit a model or tune anything on test labels.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "clustering_results"

TRAIN_FILE = RESULT_DIR / "train_raw_features_then_pca95_k40.csv"
TEST_FILE = RESULT_DIR / "test_raw_features_then_pca95_k40.csv"


def find_writer_column(df):
    for candidate in ("writer_id", "writer", "label", "y"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not find writer label column. Available columns: "
        + ", ".join(map(str, df.columns))
    )


def main():
    for path in (TRAIN_FILE, TEST_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {path}\n"
                "First run Clustering_v2_preprocessing_comparison.py."
            )

    train = pd.read_csv(TRAIN_FILE)
    test = pd.read_csv(TEST_FILE)
    if "cluster_id" not in train.columns or "cluster_id" not in test.columns:
        raise ValueError("Expected a cluster_id column in both assignment files.")

    train_writer_col = find_writer_column(train)
    test_writer_col = find_writer_column(test)

    # Learn each cluster's writer identity using training labels only.
    # A cluster with tied writers is mapped to the most frequent writer;
    # pandas' deterministic ordering resolves ties.
    cluster_to_writer = (
        train.groupby("cluster_id")[train_writer_col]
        .agg(lambda s: s.value_counts().index[0])
        .to_dict()
    )

    test["predicted_writer_from_train_mapping"] = (
        test["cluster_id"].map(cluster_to_writer)
    )
    # If a cluster somehow has no training mapping, mark it as unknown.
    y_true = test[test_writer_col].astype(str)
    y_pred = test["predicted_writer_from_train_mapping"].astype("string").fillna("UNKNOWN").astype(str)

    accuracy = accuracy_score(y_true, y_pred)
    mapped_coverage = float(test["predicted_writer_from_train_mapping"].notna().mean())
    ari = adjusted_rand_score(test[test_writer_col], test["cluster_id"])
    nmi = normalized_mutual_info_score(test[test_writer_col], test["cluster_id"])

    print("=" * 68)
    print("TRAIN-MAPPED WRITER EVALUATION — RAW FEATURES + PCA, K=40")
    print("=" * 68)
    print("Cluster-to-writer mapping learned from TRAIN labels only.")
    print("Test labels are used only to score predictions.")
    print(f"Test pages: {len(test)}")
    print(f"Mapped test-page coverage: {mapped_coverage:.1%}")
    print(f"Test writer prediction accuracy (train-derived mapping): {accuracy:.4f}")
    print(f"Test ARI (cluster agreement metric): {ari:.4f}")
    print(f"Test NMI (cluster agreement metric): {nmi:.4f}")

    print("\nClassification report (unmapped/unknown pages count as incorrect):")
    print(classification_report(y_true, y_pred, zero_division=0))

    output_predictions = RESULT_DIR / "test_train_mapped_writer_predictions_k40_raw_pca.csv"
    test.to_csv(output_predictions, index=False)

    # Save mapping and training cluster composition for transparency.
    mapping_rows = []
    for cluster_id, group in train.groupby("cluster_id"):
        counts = group[train_writer_col].value_counts()
        mapping_rows.append({
            "cluster_id": cluster_id,
            "mapped_writer_from_train": cluster_to_writer[cluster_id],
            "train_pages_in_cluster": int(len(group)),
            "distinct_train_writers_in_cluster": int(group[train_writer_col].nunique()),
            "majority_writer_train_pages": int(counts.iloc[0]),
            "majority_share_in_cluster": float(counts.iloc[0] / len(group)),
        })
    mapping_path = RESULT_DIR / "train_derived_cluster_to_writer_mapping_k40_raw_pca.csv"
    pd.DataFrame(mapping_rows).to_csv(mapping_path, index=False)

    print("\nSaved test predictions:")
    print(output_predictions)
    print("Saved train-derived cluster-to-writer mapping:")
    print(mapping_path)
    print("\nInterpretation: this is a more realistic writer prediction check than")
    print("matching test cluster IDs to test labels after seeing the test labels.")


if __name__ == "__main__":
    main()
