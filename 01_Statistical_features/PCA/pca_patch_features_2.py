import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "patch_features"
TARGETS = (0.90, 0.95, 0.98, 0.99)
PAGE_COLUMNS = ["person_id", "image_name"]
FEATURE_COLUMNS = ["R_mean", "R_var", "G_mean", "G_var", "B_mean", "B_var"]

sys.path.insert(0, str(BASE_DIR))
from run_statistical_classifiers import build_models


def main():
    source = BASE_DIR / "features_2.csv"
    patches = pd.read_csv(source)
    required = PAGE_COLUMNS + ["patch_number"] + FEATURE_COLUMNS
    missing = [column for column in required if column not in patches.columns]
    if missing:
        raise ValueError(f"features_2.csv is missing columns: {missing}")
    if patches[PAGE_COLUMNS + ["patch_number"]].isna().any().any():
        raise ValueError("features_2.csv contains missing page or patch identifiers.")
    if patches.duplicated(PAGE_COLUMNS + ["patch_number"]).any():
        raise ValueError("features_2.csv contains duplicate page/patch identifiers.")
    if not np.isfinite(patches[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("features_2.csv contains missing or non-finite RGB features.")

    grouped = patches.groupby(PAGE_COLUMNS, sort=False)
    features = grouped[FEATURE_COLUMNS].mean().reset_index()
    metadata = features[PAGE_COLUMNS].copy()
    metadata["patch_count"] = grouped.size().to_numpy()

    image_pages = pd.read_csv(BASE_DIR / "features.csv")[PAGE_COLUMNS]
    if list(map(tuple, metadata[PAGE_COLUMNS].to_numpy())) != list(
        map(tuple, image_pages.to_numpy())
    ):
        raise ValueError(
            "Page identifiers/order in features_2.csv do not match features.csv."
        )

    labels = LabelEncoder()
    y = labels.fit_transform(metadata["person_id"].astype(str))
    train_idx, test_idx = train_test_split(
        np.arange(len(metadata)), test_size=0.2, random_state=42, stratify=y
    )
    x = features[FEATURE_COLUMNS].to_numpy(dtype=float)
    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "X_train_original.npy", x_train)
    np.save(OUTPUT_DIR / "X_test_original.npy", x_test)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)
    pd.DataFrame({
        "class_index": np.arange(len(labels.classes_)),
        "person_id": labels.classes_,
    }).to_csv(OUTPUT_DIR / "label_classes.csv", index=False)
    metadata.iloc[train_idx].to_csv(OUTPUT_DIR / "metadata_train.csv", index=False)
    metadata.iloc[test_idx].to_csv(OUTPUT_DIR / "metadata_test.csv", index=False)

    start = time.perf_counter()
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    scaler_time = time.perf_counter() - start
    full_pca = PCA(svd_solver="full").fit(x_train_scaled)
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    component_counts = {
        target: int(np.searchsorted(cumulative, target) + 1)
        for target in TARGETS
    }

    models = build_models()
    results = []

    def evaluate(
        configuration, train_x, test_x, components, variance, preprocessing_time
    ):
        for name, model in models.items():
            start = time.perf_counter()
            model.fit(train_x, y_train)
            train_time = time.perf_counter() - start
            start = time.perf_counter()
            predictions = model.predict(test_x)
            test_time = time.perf_counter() - start
            results.append({
                "dataset": "patch_features",
                "configuration": configuration,
                "variance_target": (
                    float(configuration[4:]) if configuration.startswith("PCA ") else np.nan
                ),
                "original_features": len(FEATURE_COLUMNS),
                "reduced_features": components,
                "explained_variance": variance,
                "preprocessing_fit_transform_seconds": preprocessing_time,
                "classifier": name,
                "train_time_seconds": train_time,
                "test_time_seconds": test_time,
                "accuracy": accuracy_score(y_test, predictions),
                "precision_macro": precision_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "recall_macro": recall_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "f1_macro": f1_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
            })

    evaluate("Original", x_train, x_test, len(FEATURE_COLUMNS), 1.0, 0.0)
    for target, components in component_counts.items():
        start = time.perf_counter()
        pca = PCA(n_components=components, svd_solver="full").fit(x_train_scaled)
        train_pca = pca.transform(x_train_scaled)
        test_pca = pca.transform(x_test_scaled)
        preprocessing_time = scaler_time + time.perf_counter() - start
        percent = int(target * 100)
        variance = float(pca.explained_variance_ratio_.sum())
        np.save(OUTPUT_DIR / f"X_train_pca_{percent}.npy", train_pca)
        np.save(OUTPUT_DIR / f"X_test_pca_{percent}.npy", test_pca)
        joblib.dump({
            "scaler": scaler,
            "pca": pca,
            "feature_columns": FEATURE_COLUMNS,
            "variance_target": target,
            "explained_variance": variance,
            "class_labels": labels.classes_,
        }, OUTPUT_DIR / f"pca_{percent}.joblib")
        evaluate(
            f"PCA {target:.2f}", train_pca, test_pca, components, variance,
            preprocessing_time,
        )

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "pca_comparison.csv", index=False)
    pd.DataFrame({
        "variance_target": list(component_counts),
        "components": list(component_counts.values()),
        "explained_variance": [
            cumulative[count - 1] for count in component_counts.values()
        ],
    }).to_csv(OUTPUT_DIR / "pca_dimensions.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", markersize=3)
    for target, count in component_counts.items():
        ax.axhline(target, linestyle="--", linewidth=0.8, alpha=0.65)
        ax.axvline(count, linestyle=":", linewidth=0.8, alpha=0.65)
        ax.annotate(f"{target:.0%}: {count}", (count, target), fontsize=8)
    ax.set(
        xlabel="Number of principal components",
        ylabel="Cumulative explained variance",
        title="Training-set PCA explained variance: patch features",
        ylim=(0, 1.02),
    )
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "explained_variance.png", dpi=180)
    plt.close(fig)

    patch_distribution = metadata["patch_count"].value_counts().sort_index()
    patch_summary = ", ".join(
        f"{int(count)} patches: {int(pages)} pages"
        for count, pages in patch_distribution.items()
    )
    (OUTPUT_DIR / "experiment_notes.txt").write_text(
        "PCA and StandardScaler are fitted on training pages only.\n"
        "The same stratified 80/20 page split (random_state=42) is used for "
        "original and PCA features.\n"
        "Only the six RGB mean/variance features are used; patch dimensions "
        "are excluded.\n"
        f"All pages are retained and mean-pooled over available patches "
        f"({patch_summary}).\n"
        "Mean pooling discards patch positions. Different patch counts may "
        "also cover different image areas.\n"
        "metadata_train.csv and metadata_test.csv preserve each page's patch_count.\n"
        "The held-out set contains one page per writer in this dataset; "
        "classification metrics should be interpreted cautiously.\n",
        encoding="utf-8",
    )
    print(f"Completed patch-level PCA. Results: {OUTPUT_DIR}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
