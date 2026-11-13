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
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "statistical_features"
TARGETS = (0.90, 0.95, 0.98, 0.99)

sys.path.insert(0, str(BASE_DIR))
from run_statistical_classifiers import build_models


def main():
    df = pd.read_csv(BASE_DIR / "features.csv")
    if not {"person_id", "image_name"}.issubset(df.columns):
        raise ValueError("features.csv must contain person_id and image_name.")
    if df[["person_id", "image_name"]].isna().any().any():
        raise ValueError("features.csv contains missing page identifiers.")
    if df.duplicated(["person_id", "image_name"]).any():
        raise ValueError("features.csv contains duplicate page identifiers.")

    feature_columns = [
        column for column in df.select_dtypes(include=[np.number]).columns
    ]
    if len(feature_columns) != 24:
        raise ValueError(f"Expected 24 numeric features, found {len(feature_columns)}.")
    features = df[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(features).all():
        raise ValueError("features.csv contains missing or non-finite features.")

    labels = LabelEncoder()
    y = labels.fit_transform(df["person_id"].astype(str))
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=42, stratify=y
    )
    x_train, x_test = features[train_idx], features[test_idx]
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
    df.iloc[train_idx][["person_id", "image_name"]].to_csv(
        OUTPUT_DIR / "metadata_train.csv", index=False
    )
    df.iloc[test_idx][["person_id", "image_name"]].to_csv(
        OUTPUT_DIR / "metadata_test.csv", index=False
    )

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
                "dataset": "statistical_features",
                "configuration": configuration,
                "variance_target": (
                    float(configuration[4:]) if configuration.startswith("PCA ") else np.nan
                ),
                "original_features": len(feature_columns),
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

    evaluate("Original", x_train, x_test, len(feature_columns), 1.0, 0.0)
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
            "feature_columns": feature_columns,
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
        title="Training-set PCA explained variance: statistical features",
        ylim=(0, 1.02),
    )
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "explained_variance.png", dpi=180)
    plt.close(fig)

    (OUTPUT_DIR / "experiment_notes.txt").write_text(
        "PCA and StandardScaler are fitted on training pages only.\n"
        "The same stratified 80/20 page split (random_state=42) is used for "
        "original and PCA features.\n"
        "The held-out set contains one page per writer in this dataset; "
        "classification metrics should be interpreted cautiously.\n",
        encoding="utf-8",
    )
    print(f"Completed image-level PCA. Results: {OUTPUT_DIR}")
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
