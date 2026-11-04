import time
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
FEATURE_DIR = BASE_DIR / "mobilenetv2" / "features"
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "mobilenetv2"
VARIANCE_TARGETS = (0.90, 0.95, 0.98, 0.99)


def build_classifiers():
    return {
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=10, gamma="scale")),
        ]),
        "SVM_LINEAR": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="linear", C=1)),
        ]),
        "KNN_3": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=3)),
        ]),
        "KNN_5": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "RANDOM_FOREST": RandomForestClassifier(
            n_estimators=200, random_state=42, n_jobs=-1
        ),
        "LOGISTIC_REGRESSION": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=42)),
        ]),
        "NAIVE_BAYES": GaussianNB(),
        "DECISION_TREE": DecisionTreeClassifier(max_depth=20, random_state=42),
        "XGBOOST": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric="mlogloss",
        ),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(128,), max_iter=1000, random_state=42
            )),
        ]),
        "ADABOOST": AdaBoostClassifier(
            n_estimators=200, learning_rate=0.5, random_state=42
        ),
    }


def load_page_metadata(split):
    metadata = pd.read_csv(FEATURE_DIR / f"{split}_patch_metadata.csv")
    columns = ["writer_id", "writer", "patch_path"]
    if not set(columns).issubset(metadata.columns):
        raise ValueError(f"{split} patch metadata is missing required columns.")
    if metadata[columns].isna().any().any():
        raise ValueError(f"{split} patch metadata contains missing values.")

    metadata = metadata.copy()
    metadata["page_name"] = metadata["patch_path"].map(
        lambda path: Path(path).name.rsplit("_patch_", 1)[0]
    )
    pages = []
    for (writer_id, writer, page_name), group in metadata.groupby(
        ["writer_id", "writer", "page_name"], sort=False
    ):
        pages.append({
            "writer_id": int(writer_id),
            "writer": writer,
            "page_name": page_name,
            "patch_count": len(group),
        })
    return pd.DataFrame(pages)


def evaluate_variant(
    configuration,
    x_train,
    x_test,
    y_train,
    y_test,
    class_ids,
    class_names,
    preprocessing_seconds,
    feature_count,
    explained_variance,
    results,
):
    for name, classifier in build_classifiers().items():
        start = time.perf_counter()
        classifier.fit(x_train, y_train)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predictions = classifier.predict(x_test)
        test_seconds = time.perf_counter() - start

        results.append({
            "configuration": configuration,
            "classifier": name,
            "original_features": 1280,
            "reduced_features": feature_count,
            "explained_variance": explained_variance,
            "preprocessing_fit_transform_seconds": preprocessing_seconds,
            "train_time_seconds": train_seconds,
            "test_time_seconds": test_seconds,
            "accuracy": accuracy_score(y_test, predictions),
            "precision_macro": precision_score(
                y_test, predictions, labels=class_ids,
                average="macro", zero_division=0,
            ),
            "recall_macro": recall_score(
                y_test, predictions, labels=class_ids,
                average="macro", zero_division=0,
            ),
            "f1_macro": f1_score(
                y_test, predictions, labels=class_ids,
                average="macro", zero_division=0,
            ),
        })

        matrix = confusion_matrix(y_test, predictions, labels=class_ids)
        fig, ax = plt.subplots(figsize=(12, 10))
        image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        ax.set_title(f"MobileNetV2 {configuration} - {name}")
        ax.set_xlabel("Predicted writer")
        ax.set_ylabel("Actual writer")
        ax.set_xticks(class_ids)
        ax.set_yticks(class_ids)
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        safe_configuration = configuration.lower().replace(" ", "_")
        fig.savefig(
            OUTPUT_DIR
            / f"{safe_configuration}_{name.lower()}_confusion_matrix.png",
            dpi=160,
        )
        plt.close(fig)


def main():
    x_train = np.load(FEATURE_DIR / "X_train_page.npy")
    x_test = np.load(FEATURE_DIR / "X_test_page.npy")
    y_train_raw = np.load(FEATURE_DIR / "y_train_page.npy")
    y_test_raw = np.load(FEATURE_DIR / "y_test_page.npy")
    train_metadata = load_page_metadata("train")
    test_metadata = load_page_metadata("test")

    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("Page-level MobileNetV2 features must be 2D.")
    if x_train.shape[1] != 1280 or x_test.shape[1] != 1280:
        raise ValueError("Expected 1280-dimensional MobileNetV2 features.")
    if len(x_train) != len(y_train_raw) or len(x_test) != len(y_test_raw):
        raise ValueError("Page feature and label counts do not match.")
    if len(train_metadata) != len(x_train) or len(test_metadata) != len(x_test):
        raise ValueError("Page metadata and feature counts do not match.")
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Page features contain non-finite values.")

    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train_raw, y_test_raw]))
    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    class_ids = np.arange(len(label_encoder.classes_))
    writer_names = {
        **dict(zip(train_metadata["writer_id"], train_metadata["writer"])),
        **dict(zip(test_metadata["writer_id"], test_metadata["writer"])),
    }
    class_names = [writer_names[writer_id] for writer_id in label_encoder.classes_]
    if not np.array_equal(train_metadata["writer_id"], y_train_raw):
        raise ValueError("Training page metadata does not align with page labels.")
    if not np.array_equal(test_metadata["writer_id"], y_test_raw):
        raise ValueError("Testing page metadata does not align with page labels.")

    train_pages = set(zip(train_metadata["writer"], train_metadata["page_name"]))
    test_pages = set(zip(test_metadata["writer"], test_metadata["page_name"]))
    if train_pages & test_pages:
        raise ValueError("A page occurs in both training and testing data.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_DIR / "X_train_original.npy", x_train)
    np.save(OUTPUT_DIR / "X_test_original.npy", x_test)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)
    pd.DataFrame({
        "class_index": class_ids,
        "writer_id": label_encoder.classes_,
    }).to_csv(OUTPUT_DIR / "label_classes.csv", index=False)
    train_metadata.to_csv(OUTPUT_DIR / "metadata_train.csv", index=False)
    test_metadata.to_csv(OUTPUT_DIR / "metadata_test.csv", index=False)

    maximum_components = min(len(x_train) - 1, x_train.shape[1])
    start = time.perf_counter()
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    scaler_seconds = time.perf_counter() - start

    start = time.perf_counter()
    full_pca = PCA(n_components=maximum_components, svd_solver="full")
    full_pca.fit(x_train_scaled)
    full_pca_seconds = time.perf_counter() - start
    cumulative_variance = np.cumsum(full_pca.explained_variance_ratio_)
    component_counts = {
        target: min(
            int(np.searchsorted(cumulative_variance, target) + 1),
            maximum_components,
        )
        for target in VARIANCE_TARGETS
    }

    results = []
    evaluate_variant(
        "Original", x_train, x_test, y_train, y_test,
        class_ids, class_names, 0.0, x_train.shape[1], 1.0, results,
    )

    dimensions = []
    for target, component_count in component_counts.items():
        start = time.perf_counter()
        pca = PCA(n_components=component_count, svd_solver="full")
        pca.fit(x_train_scaled)
        x_train_pca = pca.transform(x_train_scaled)
        x_test_pca = pca.transform(x_test_scaled)
        preprocessing_seconds = (
            scaler_seconds + full_pca_seconds + time.perf_counter() - start
        )
        variance = float(pca.explained_variance_ratio_.sum())
        percentage = int(target * 100)

        np.save(OUTPUT_DIR / f"X_train_pca_{percentage}.npy", x_train_pca)
        np.save(OUTPUT_DIR / f"X_test_pca_{percentage}.npy", x_test_pca)
        joblib.dump({
            "scaler": scaler,
            "pca": pca,
            "variance_target": target,
            "explained_variance": variance,
        }, OUTPUT_DIR / f"pca_{percentage}.joblib")
        dimensions.append({
            "variance_target": target,
            "original_features": x_train.shape[1],
            "components": component_count,
            "explained_variance": variance,
            "maximum_possible_components": maximum_components,
        })

        evaluate_variant(
            f"PCA {target:.2f}", x_train_pca, x_test_pca,
            y_train, y_test, class_ids, class_names,
            preprocessing_seconds, component_count, variance, results,
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / "pca_comparison.csv", index=False)
    pd.DataFrame(dimensions).to_csv(
        OUTPUT_DIR / "pca_dimensions.csv", index=False
    )

    baseline_path = (
        BASE_DIR / "mobilenetv2" / "classifier_results"
        / "classifier_comparison.csv"
    )
    baseline = pd.read_csv(baseline_path)
    baseline_columns = {
        "Classifier", "Accuracy", "Precision", "Recall", "F1",
    }
    if not baseline_columns.issubset(baseline.columns):
        raise ValueError(f"Unexpected baseline columns in {baseline_path}.")
    current_original = results_df[results_df["configuration"] == "Original"]
    comparison = baseline.merge(
        current_original,
        left_on="Classifier",
        right_on="classifier",
        validate="one_to_one",
    )
    metric_pairs = {
        "Accuracy": "accuracy",
        "Precision": "precision_macro",
        "Recall": "recall_macro",
        "F1": "f1_macro",
    }
    for baseline_metric, current_metric in metric_pairs.items():
        comparison[f"{baseline_metric}_difference"] = (
            comparison[current_metric] - comparison[baseline_metric]
        )
        comparison[f"{baseline_metric}_matches"] = np.isclose(
            comparison[current_metric], comparison[baseline_metric]
        )
    comparison.to_csv(
        OUTPUT_DIR / "baseline_comparison.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    components = np.arange(1, len(cumulative_variance) + 1)
    ax.plot(components, cumulative_variance, marker="o", markersize=3)
    for target, count in component_counts.items():
        ax.axhline(target, linestyle="--", linewidth=0.8, alpha=0.65)
        ax.axvline(count, linestyle=":", linewidth=0.8, alpha=0.65)
        ax.annotate(f"{target:.0%}: {count}", (count, target), fontsize=8)
    ax.set(
        xlabel="Number of principal components",
        ylabel="Cumulative explained variance",
        title="MobileNetV2 training-page PCA explained variance",
        ylim=(0, 1.02),
    )
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "explained_variance.png", dpi=180)
    plt.close(fig)

    mismatches = []
    for metric in metric_pairs:
        count = int((~comparison[f"{metric}_matches"]).sum())
        if count:
            classifiers = comparison.loc[
                ~comparison[f"{metric}_matches"], "Classifier"
            ].tolist()
            mismatches.append(f"{metric}: {', '.join(classifiers)}")
    baseline_note = (
        "The original-feature baseline matches the saved baseline CSV."
        if not mismatches else
        "Saved baseline CSV differs from this rerun for " + "; ".join(mismatches) + "."
    )
    (OUTPUT_DIR / "experiment_notes.txt").write_text(
        "PCA and StandardScaler are fitted on training pages only.\n"
        "Existing page-level mean-pooled features and train/test split are reused.\n"
        f"Training pages: {len(x_train)}; test pages: {len(x_test)}; "
        f"maximum PCA components: {maximum_components}.\n"
        f"{baseline_note}\n"
        "The held-out set contains one page per writer; classification "
        "metrics should be interpreted cautiously.\n",
        encoding="utf-8",
    )
    print(f"MobileNetV2 PCA complete. Results saved to: {OUTPUT_DIR}")
    print(pd.DataFrame(dimensions).to_string(index=False))
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50% of the number of samples",
            category=UserWarning,
        )
        main()
