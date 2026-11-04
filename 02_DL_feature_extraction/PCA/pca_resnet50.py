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
FEATURE_DIR = BASE_DIR / "resnet50" / "features"
BASELINE_DIR = BASE_DIR / "resnet50" / "classifier_results"
OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "resnet50"
VARIANCE_TARGETS = (0.90, 0.95, 0.98, 0.99)
EXPECTED_FEATURES = 2048


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


def load_page_data(split):
    features = np.load(FEATURE_DIR / f"X_{split}_page.npy")
    labels = np.load(FEATURE_DIR / f"y_{split}_page.npy")
    patches = np.load(FEATURE_DIR / f"X_{split}_patches.npy")
    metadata = pd.read_csv(FEATURE_DIR / f"{split}_patch_metadata.csv")

    required = ["writer_id", "writer", "patch_path"]
    if not set(required).issubset(metadata.columns):
        raise ValueError(f"{split} patch metadata is missing required columns.")
    if metadata[required].isna().any().any():
        raise ValueError(f"{split} patch metadata contains missing values.")
    if len(patches) != len(metadata):
        raise ValueError(f"{split} patch features and metadata counts differ.")
    if features.ndim != 2 or features.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"{split} page features must have {EXPECTED_FEATURES} dimensions."
        )
    if patches.ndim != 2 or patches.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"{split} patch features must have {EXPECTED_FEATURES} dimensions."
        )
    if not np.isfinite(features).all() or not np.isfinite(patches).all():
        raise ValueError(f"{split} feature arrays contain non-finite values.")

    metadata = metadata.copy()
    metadata["page_name"] = metadata["patch_path"].map(
        lambda path: Path(path).name.rsplit("_patch_", 1)[0]
    )
    groups = metadata.groupby(
        ["writer_id", "writer", "page_name"], sort=False
    )

    page_rows = []
    pooled_features = []
    for (writer_id, writer, page_name), group in groups:
        indices = group.index.to_numpy()
        pooled_features.append(patches[indices].mean(axis=0))
        page_rows.append({
            "writer_id": int(writer_id),
            "writer": writer,
            "page_name": page_name,
            "patch_count": len(group),
        })

    page_metadata = pd.DataFrame(page_rows)
    pooled_features = np.asarray(pooled_features)
    if len(features) != len(page_metadata) or len(labels) != len(page_metadata):
        raise ValueError(f"{split} page features, labels, and metadata do not align.")
    if not np.array_equal(page_metadata["writer_id"].to_numpy(), labels):
        raise ValueError(f"{split} page metadata does not align with page labels.")
    if not np.allclose(pooled_features, features, rtol=1e-5, atol=1e-6):
        max_diff = float(np.max(np.abs(pooled_features - features)))
        raise ValueError(
            f"{split} page features differ from patch mean pooling "
            f"(max absolute difference {max_diff})."
        )

    return features, labels, page_metadata, len(groups), float(
        np.max(np.abs(pooled_features - features))
    )


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
    model_dir = OUTPUT_DIR / "classifier_models" / (
        configuration.lower().replace(" ", "_")
    )
    model_dir.mkdir(parents=True, exist_ok=True)

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
            "original_features": EXPECTED_FEATURES,
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
        joblib.dump(classifier, model_dir / f"{name.lower()}.joblib")

        matrix = confusion_matrix(y_test, predictions, labels=class_ids)
        fig, ax = plt.subplots(figsize=(12, 10))
        image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        ax.set_title(f"ResNet50 {configuration} - {name}")
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
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing ResNet50 PCA results: {OUTPUT_DIR}"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train_raw, train_metadata, train_page_count, train_pool_diff = (
        load_page_data("train")
    )
    x_test, y_test_raw, test_metadata, test_page_count, test_pool_diff = (
        load_page_data("test")
    )
    if train_page_count != 160 or test_page_count != 40:
        raise ValueError(
            f"Expected 160 train and 40 test pages; found "
            f"{train_page_count} and {test_page_count}."
        )

    train_pages = set(zip(train_metadata["writer"], train_metadata["page_name"]))
    test_pages = set(zip(test_metadata["writer"], test_metadata["page_name"]))
    overlap = train_pages & test_pages
    if overlap:
        raise ValueError(f"Pages occur in both splits: {sorted(overlap)}")

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

    np.save(OUTPUT_DIR / "X_train_original.npy", x_train)
    np.save(OUTPUT_DIR / "X_test_original.npy", x_test)
    np.save(OUTPUT_DIR / "y_train.npy", y_train)
    np.save(OUTPUT_DIR / "y_test.npy", y_test)
    pd.DataFrame({
        "class_index": class_ids,
        "writer_id": label_encoder.classes_,
        "writer": class_names,
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
            "training_page_count": len(x_train),
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
    pd.DataFrame({
        "component": np.arange(1, len(cumulative_variance) + 1),
        "explained_variance_ratio": full_pca.explained_variance_ratio_,
        "cumulative_explained_variance": cumulative_variance,
    }).to_csv(OUTPUT_DIR / "explained_variance.csv", index=False)

    baseline_path = BASELINE_DIR / "classifier_comparison.csv"
    baseline = pd.read_csv(baseline_path)
    baseline_columns = {"Classifier", "Accuracy", "Precision", "Recall", "F1"}
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
        title="ResNet50 training-page PCA explained variance",
        ylim=(0, 1.02),
    )
    ax.grid(linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "explained_variance.png", dpi=180)
    plt.close(fig)

    mismatches = []
    for metric in metric_pairs:
        differing = comparison.loc[
            ~comparison[f"{metric}_matches"], "Classifier"
        ].tolist()
        if differing:
            mismatches.append(f"{metric}: {', '.join(differing)}")
    baseline_note = (
        "All original-feature metrics match the saved baseline."
        if not mismatches else
        "Saved baseline CSV differs from this rerun for "
        + "; ".join(mismatches) + "."
    )
    (OUTPUT_DIR / "experiment_notes.txt").write_text(
        "Scaler and PCA are fitted on training pages only.\n"
        "Existing page-level mean-pooled features and train/test split are reused.\n"
        f"Training pages: {len(x_train)}; test pages: {len(x_test)}; "
        f"maximum PCA components: {maximum_components}.\n"
        f"{baseline_note}\n"
        "The held-out set contains one page per writer; classification "
        "metrics should be interpreted cautiously.\n",
        encoding="utf-8",
    )

    validation = [
        ("feature_dimension", x_train.shape[1] == EXPECTED_FEATURES,
         f"train={x_train.shape[1]}, test={x_test.shape[1]}, expected={EXPECTED_FEATURES}"),
        ("training_page_count", train_page_count == 160, str(train_page_count)),
        ("testing_page_count", test_page_count == 40, str(test_page_count)),
        ("train_patch_mean_pooling", train_pool_diff <= 1e-6, str(train_pool_diff)),
        ("test_patch_mean_pooling", test_pool_diff <= 1e-6, str(test_pool_diff)),
        ("no_train_test_page_overlap", len(overlap) == 0, str(len(overlap))),
        ("scaler_fit_on_training_pages",
         int(scaler.n_samples_seen_) == len(x_train), str(scaler.n_samples_seen_)),
        ("pca_fit_on_training_pages",
         all(int(joblib.load(OUTPUT_DIR / f"pca_{int(t * 100)}.joblib")["pca"].n_samples_) == len(x_train)
             for t in VARIANCE_TARGETS),
         str(len(x_train))),
        ("classifier_configurations", len(results_df) == 55, str(len(results_df))),
        ("confusion_matrices",
         len(list(OUTPUT_DIR.glob("*_confusion_matrix.png"))) == 55,
         str(len(list(OUTPUT_DIR.glob("*_confusion_matrix.png"))))),
        ("baseline_comparison_rows", len(comparison) == 11, str(len(comparison))),
    ]
    validation_df = pd.DataFrame(
        validation, columns=["check", "passed", "details"]
    )
    validation_df.to_csv(OUTPUT_DIR / "validation_results.csv", index=False)
    if not validation_df["passed"].all():
        raise RuntimeError(
            "One or more result validation checks failed; see validation_results.csv."
        )

    print(f"ResNet50 PCA complete. Results saved to: {OUTPUT_DIR}")
    print(pd.DataFrame(dimensions).to_string(index=False))
    for configuration, group in results_df.groupby("configuration", sort=False):
        best_accuracy = group.loc[group["accuracy"].idxmax()]
        best_f1 = group.loc[group["f1_macro"].idxmax()]
        print(
            f"{configuration}: best accuracy {best_accuracy['classifier']} "
            f"{best_accuracy['accuracy']:.3f}; best macro-F1 "
            f"{best_f1['classifier']} {best_f1['f1_macro']:.3f}"
        )
    print(validation_df.to_string(index=False))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The number of unique classes is greater than 50% of the number of samples",
            category=UserWarning,
        )
        main()
