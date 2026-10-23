# ============================================================
# 03_train_classifiers.py
#
# Purpose:
#   1. Load AlexNet features
#   2. Aggregate patches into page-level features
#   3. Train classical ML classifiers
#   4. Compare their performance
#
# AlexNet is NOT used here.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_DIR = Path("features")

CLASSIFIER_RESULTS_DIR = Path(
    "classifier_results"
)

CLASSIFIER_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD PATCH FEATURES
# ============================================================

print("=" * 60)
print("CLASSICAL MACHINE LEARNING")
print("=" * 60)

print("\nLoading AlexNet features...")


X_train_patches = np.load(
    FEATURE_DIR /
    "X_train_patches.npy"
)

y_train_patches = np.load(
    FEATURE_DIR /
    "y_train_patches.npy"
)


X_test_patches = np.load(
    FEATURE_DIR /
    "X_test_patches.npy"
)

y_test_patches = np.load(
    FEATURE_DIR /
    "y_test_patches.npy"
)


# ============================================================
# LOAD METADATA
# ============================================================

train_metadata = pd.read_csv(
    FEATURE_DIR /
    "train_patch_metadata.csv"
)

test_metadata = pd.read_csv(
    FEATURE_DIR /
    "test_patch_metadata.csv"
)


# ============================================================
# PAGE FEATURE AGGREGATION
# ============================================================

def create_page_features(
    X_patches,
    metadata
):
    """
    Combine all patch features belonging to
    the same original page.

    Mean pooling is used:

        page_feature =
            mean(all patch features)
    """

    # --------------------------------------------------------
    # Extract original page from patch filename.
    #
    # Example:
    #
    # page_1_patch_01.jpg
    #
    # becomes:
    #
    # page_1
    # --------------------------------------------------------

    metadata = metadata.copy()

    metadata["page_key"] = (
        metadata["patch_path"]
        .apply(
            lambda x:
            Path(x).name.rsplit(
                "_patch_",
                1
            )[0]
        )
    )

    page_features = []

    page_labels = []

    page_names = []

    # --------------------------------------------------------
    # Group patches belonging to each page.
    # --------------------------------------------------------

    grouped = metadata.groupby(
        ["writer_id", "writer", "page_key"],
        sort=False
    )

    # --------------------------------------------------------
    # Important:
    #
    # X_patches and metadata have the same order because
    # they were generated together in script 02.
    #
    # Therefore we can use the DataFrame row positions.
    # --------------------------------------------------------

    metadata = metadata.reset_index(
        drop=True
    )

    for (
        writer_id,
        writer,
        page_key
    ), group in grouped:

        indices = group.index.to_numpy()

        # --------------------------------------------
        # Get features belonging to this page
        # --------------------------------------------

        page_patch_features = (
            X_patches[indices]
        )

        # --------------------------------------------
        # Mean pooling
        # --------------------------------------------

        page_feature = (
            page_patch_features.mean(
                axis=0
            )
        )

        page_features.append(
            page_feature
        )

        page_labels.append(
            writer_id
        )

        page_names.append(
            f"{writer}/{page_key}"
        )

    page_features = np.array(
        page_features
    )

    page_labels = np.array(
        page_labels,
        dtype=np.int64
    )

    return (
        page_features,
        page_labels,
        page_names
    )


# ============================================================
# CREATE PAGE-LEVEL FEATURES
# ============================================================

print("\nAggregating training patches...")

X_train, y_train, train_pages = (
    create_page_features(
        X_train_patches,
        train_metadata
    )
)

print(
    "Training page feature shape:",
    X_train.shape
)

print(
    "Training labels shape:",
    y_train.shape
)


print("\nAggregating testing patches...")

X_test, y_test, test_pages = (
    create_page_features(
        X_test_patches,
        test_metadata
    )
)

print(
    "Testing page feature shape:",
    X_test.shape
)

print(
    "Testing labels shape:",
    y_test.shape
)


# ============================================================
# EXPECTED SHAPES
# ============================================================

print("\nExpected:")
print(
    "Training: approximately 160 pages × 4096"
)

print(
    "Testing: approximately 40 pages × 4096"
)


# ============================================================
# CLASSIFIERS
# ============================================================

classifiers = {

    "SVM_RBF": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale"
            )
        )
    ]),


    "SVM_LINEAR": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            SVC(
                kernel="linear",
                C=1
            )
        )
    ]),


    "KNN_3": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=3
            )
        )
    ]),


    "KNN_5": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=5
            )
        )
    ]),


    "RANDOM_FOREST": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),


    "LOGISTIC_REGRESSION": Pipeline([
        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=3000,
                random_state=42
            )
        )
    ]),
            "NAIVE_BAYES": GaussianNB(),

"DECISION_TREE": DecisionTreeClassifier(
    max_depth=20,
    random_state=42
),

"XGBOOST": XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="mlogloss"
),

"MLP": Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=1000,
            random_state=42
        )
    )
]),

"ADABOOST": AdaBoostClassifier(
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)
}


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

results = []


for name, classifier in classifiers.items():

    print("\n")
    print("=" * 60)
    print(
        f"TRAINING: {name}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    classifier.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = classifier.predict(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"\nAccuracy : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Precision: "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Recall   : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"F1-score : "
        f"{f1 * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append(
        {
            "Classifier": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(
        figsize=(14, 12)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        f"Confusion Matrix - {name}"
    )

    plt.xlabel(
        "Predicted Writer"
    )

    plt.ylabel(
        "Actual Writer"
    )

    plt.colorbar()

    plt.tight_layout()

    confusion_path = (
        CLASSIFIER_RESULTS_DIR /
        f"{name}_confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=200
    )

    plt.close()

    print(
        f"Confusion matrix saved to:"
    )

    print(
        confusion_path
    )


# ============================================================
# RESULTS TABLE
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "Accuracy",
    ascending=False
)


print("\n")
print("=" * 60)
print("FINAL CLASSIFIER COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    CLASSIFIER_RESULTS_DIR /
    "classifier_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print(
    f"\nResults saved to:"
)

print(
    results_path
)


# ============================================================
# SAVE PAGE FEATURES
# ============================================================

np.save(
    FEATURE_DIR /
    "X_train_page.npy",
    X_train
)

np.save(
    FEATURE_DIR /
    "y_train_page.npy",
    y_train
)

np.save(
    FEATURE_DIR /
    "X_test_page.npy",
    X_test
)

np.save(
    FEATURE_DIR /
    "y_test_page.npy",
    y_test
)


print(
    "\nPage-level feature files saved."
)


# ============================================================
# END
# ============================================================

print("\n")
print("=" * 60)
print("DONE")
print("=" * 60)