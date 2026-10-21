
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def build_models():
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "MLP Classifier": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=1000,
                early_stopping=True,
                random_state=42
            ))
        ]),
        "Naive Bayes": GaussianNB(),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000, random_state=42
            ))
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="linear", random_state=42))
        ]),
        "RBF SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", random_state=42))
        ]),
        "KNN (K=3)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=3))
        ]),
        "KNN (K=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5))
        ]),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        )
    except ImportError:
        print("NOTE: xgboost is not installed; XGBoost will be skipped.")
        print("Install it with: pip install xgboost")

    return models


def main():
    # Get the folder where this Python file is saved
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Input and output paths
    features_path = os.path.join(script_dir, "features.csv")
    output_dir = os.path.join(script_dir, "output")

    # Create output folder if it does not already exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    df = pd.read_csv(features_path)

    print("Features loaded successfully.")
    print(f"Number of rows: {len(df)}")
    print(f"Output folder: {output_dir}")
    if "person_id" not in df.columns:
        raise ValueError("CSV must contain a 'person_id' target column.")

    # Exclude identifiers and keep only numeric feature columns.
    y_raw = df["person_id"].astype(str)
    excluded = {"person_id", "image_name"}
    feature_df = df.drop(columns=[c for c in excluded if c in df.columns])
    X = feature_df.select_dtypes(include=[np.number]).copy()

    if X.empty:
        raise ValueError("No numeric feature columns were found.")
    if X.isna().any().any():
        # Median imputation is fit on training data only via the pipeline below.
        from sklearn.impute import SimpleImputer
        X_values = X
    else:
        X_values = X

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_

    # Stratification keeps writer proportions similar in train and test.
    X_train, X_test, y_train, y_test = train_test_split(
        X_values, y, test_size=0.2, random_state=42, stratify=y
    )

    # Impute missing values using training-set medians only, avoiding test leakage.
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    models = build_models()
    results = []
    prediction_store = {}

    for name, model in models.items():
        print(f"Training: {name}")
        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            prediction_store[name] = predictions

            results.append({
                "Classifier": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision_macro": precision_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "Recall_macro": recall_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "F1_macro": f1_score(
                    y_test, predictions, average="macro", zero_division=0
                ),
                "Precision_weighted": precision_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "Recall_weighted": recall_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
                "F1_weighted": f1_score(
                    y_test, predictions, average="weighted", zero_division=0
                ),
            })

            # Save one confusion-matrix PNG per classifier.
            cm = confusion_matrix(
                y_test, predictions, labels=np.arange(len(class_names))
            )
            fig_size = max(10, min(22, len(class_names) * 0.45))
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
            ax.set_title(f"Confusion Matrix - {name}")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            ticks = np.arange(len(class_names))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels(class_names, rotation=90, fontsize=7)
            ax.set_yticklabels(class_names, fontsize=7)
            ax.set_xlabel("Predicted writer")
            ax.set_ylabel("True writer")
            # Avoid cluttering the plot with numbers if there are many classes.
            if len(class_names) <= 20:
                threshold = cm.max() / 2 if cm.size else 0
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(
                            j, i, str(cm[i, j]),
                            ha="center", va="center",
                            color="white" if cm[i, j] > threshold else "black",
                            fontsize=7
                        )
            fig.tight_layout()
            safe_name = "".join(
                ch.lower() if ch.isalnum() else "_" for ch in name
            ).strip("_")
            fig.savefig(os.path.join(output_dir, f"confusion_matrix_{safe_name}.png"),
                        dpi=200, bbox_inches="tight")
            plt.close(fig)

            # Save per-class precision/recall/F1 for each classifier.
            report = classification_report(
                y_test, predictions, labels=np.arange(len(class_names)),
                target_names=class_names, output_dict=True, zero_division=0
            )
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(os.path.join(output_dir, f"classification_report_{safe_name}.csv"))

        except Exception as exc:
            print(f"  FAILED for {name}: {exc}")

    if not results:
        raise RuntimeError("All classifiers failed; check the errors printed above.")

    results_df = pd.DataFrame(results).sort_values(
        by="F1_macro", ascending=False
    )
    results_df.to_csv(os.path.join(output_dir, "classifier_metrics.csv"), index=False)

    # Combined metric comparison charts.
    for metric in ["Accuracy", "Precision_macro", "Recall_macro", "F1_macro"]:
        plot_df = results_df.sort_values(metric, ascending=True)
        fig_height = max(6, len(plot_df) * 0.45)
        fig, ax = plt.subplots(figsize=(11, fig_height))
        ax.barh(plot_df["Classifier"], plot_df[metric])
        ax.set_xlabel(metric.replace("_", " "))
        ax.set_title(f"Classifier Comparison: {metric.replace('_', ' ')}")
        ax.set_xlim(0, 1)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"comparison_{metric.lower()}.png"),
                    dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("\nFinished.")
    print(f"Metrics CSV: {os.path.join(output_dir, 'classifier_metrics.csv')}")
    print(f"Graphs and per-class reports: {output_dir}")
    print("\nResults (sorted by macro F1):")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
