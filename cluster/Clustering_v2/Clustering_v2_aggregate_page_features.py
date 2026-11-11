"""
Aggregate AlexNet patch features into one feature vector per original page.

Expected files in Clustering_v2/features:
  X_train_patches.npy
  y_train_patches.npy
  X_test_patches.npy
  y_test_patches.npy
  train_patch_metadata.csv
  test_patch_metadata.csv

Outputs are saved in the same features folder:
  X_train_page.npy, y_train_page.npy
  X_test_page.npy, y_test_page.npy
  train_page_metadata.csv, test_page_metadata.csv
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
FEATURE_DIR = SCRIPT_DIR / "features"


def original_page_id(patch_path):
    """
    Example:
      patch_dataset/train/A/page_2_patch_01.jpg -> page_2

    The writer is stored in a separate metadata column, so the final
    page identity is the combination of writer + page_id.
    """
    filename = str(patch_path).replace("\\", "/").split("/")[-1]
    stem = Path(filename).stem
    match = re.match(r"^(.*)_patch_\d+$", stem, flags=re.IGNORECASE)

    if not match:
        raise ValueError(
            f"Cannot infer original page name from patch path: {patch_path}\n"
            "Expected a filename like page_2_patch_01.jpg"
        )

    return match.group(1)


def aggregate_split(split):
    X_path = FEATURE_DIR / f"X_{split}_patches.npy"
    y_path = FEATURE_DIR / f"y_{split}_patches.npy"
    metadata_path = FEATURE_DIR / f"{split}_patch_metadata.csv"

    for path in (X_path, y_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    X = np.load(X_path)
    y = np.load(y_path)
    metadata = pd.read_csv(metadata_path)

    if len(X) != len(y) or len(X) != len(metadata):
        raise ValueError(
            f"{split}: feature, label, and metadata row counts differ: "
            f"{len(X)}, {len(y)}, {len(metadata)}"
        )

    required_columns = {"writer_id", "writer", "patch_path"}
    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(f"{metadata_path} is missing columns: {sorted(missing)}")

    metadata = metadata.copy()
    metadata["page_id"] = metadata["patch_path"].map(original_page_id)

    # Ensure each patch's label agrees with its metadata writer_id.
    metadata_writer_ids = metadata["writer_id"].to_numpy()
    if not np.array_equal(y, metadata_writer_ids):
        raise ValueError(
            f"{split}: y_{split}_patches.npy does not match writer_id "
            f"in {split}_patch_metadata.csv. Check that both came from "
            "the same feature-extraction run."
        )

    # Group by writer AND original page, then mean-pool patch vectors.
    page_features = []
    page_labels = []
    page_rows = []

    grouped = metadata.groupby(["writer_id", "writer", "page_id"], sort=True)

    for (writer_id, writer, page_id), row_indices in grouped.indices.items():
        indices = np.asarray(row_indices, dtype=int)
        page_features.append(X[indices].mean(axis=0))
        page_labels.append(int(writer_id))
        page_rows.append({
            "writer_id": int(writer_id),
            "writer": str(writer),
            "page_id": str(page_id),
            "number_of_patches": int(len(indices)),
        })

    X_page = np.vstack(page_features).astype(np.float32)
    y_page = np.asarray(page_labels, dtype=np.int64)
    page_metadata = pd.DataFrame(page_rows)

    np.save(FEATURE_DIR / f"X_{split}_page.npy", X_page)
    np.save(FEATURE_DIR / f"y_{split}_page.npy", y_page)
    page_metadata.to_csv(FEATURE_DIR / f"{split}_page_metadata.csv", index=False)

    pages_per_writer = page_metadata.groupby("writer")["page_id"].nunique()

    print(f"\n========== {split.upper()} PAGE AGGREGATION ==========")
    print(f"Patch feature shape : {X.shape}")
    print(f"Page feature shape  : {X_page.shape}")
    print(f"Unique writers      : {page_metadata['writer'].nunique()}")
    print(f"Pages per writer    : {pages_per_writer.value_counts().sort_index().to_dict()}")
    print(f"Patches per page    : {page_metadata['number_of_patches'].describe().round(2).to_dict()}")
    print(f"Saved               : {FEATURE_DIR / f'X_{split}_page.npy'}")
    print(f"Saved               : {FEATURE_DIR / f'y_{split}_page.npy'}")
    print(f"Saved               : {FEATURE_DIR / f'{split}_page_metadata.csv'}")

    return page_metadata


def main():
    if not FEATURE_DIR.exists():
        raise FileNotFoundError(f"Feature folder not found: {FEATURE_DIR}")

    train_meta = aggregate_split("train")
    test_meta = aggregate_split("test")

    # Expected for this 3:2 experiment: 3 train pages and 2 test pages
    # for each of the same 40 writers.
    train_counts = train_meta.groupby("writer")["page_id"].nunique()
    test_counts = test_meta.groupby("writer")["page_id"].nunique()

    if set(train_counts.index) != set(test_counts.index):
        raise ValueError("Train and test writer sets differ.")

    if not (train_counts == 3).all():
        raise ValueError(
            "Expected exactly 3 training pages per writer, got: "
            f"{train_counts.value_counts().to_dict()}"
        )

    if not (test_counts == 2).all():
        raise ValueError(
            "Expected exactly 2 test pages per writer, got: "
            f"{test_counts.value_counts().to_dict()}"
        )

    print("\n3:2 split verification passed: 3 train pages and 2 test pages per writer.")
    print("Next step: run K-Means using X_train_page.npy and y_train_page.npy.")


if __name__ == "__main__":
    main()
