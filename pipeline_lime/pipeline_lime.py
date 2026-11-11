from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from lime.lime_tabular import LimeTabularExplainer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_ROOT = PROJECT_ROOT / "02_DL_feature_extraction"
RESULT_ROOT = Path(__file__).resolve().parent / "results"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODELS = {
    "alexnet": {
        "feature_dir": FEATURE_ROOT / "alexnet" / "features",
        "feature_dimension": 4096,
    },
    "resnet50": {
        "feature_dir": FEATURE_ROOT / "resnet50" / "features",
        "feature_dimension": 2048,
    },
    "mobilenetv2": {
        "feature_dir": FEATURE_ROOT / "mobilenetv2" / "features",
        "feature_dimension": 1280,
    },
    "efficientnet_b0": {
        "feature_dir": FEATURE_ROOT / "efficientnet_b0" / "features",
        "feature_dimension": 1280,
    },
}


# ============================================================
# LOAD PAGE-LEVEL FEATURES
# ============================================================

def load_features(feature_dir):
    """
    Load page-level CNN features and labels.
    """

    X_train = np.load(feature_dir / "X_train_page.npy")
    y_train = np.load(feature_dir / "y_train_page.npy")
    X_test = np.load(feature_dir / "X_test_page.npy")
    y_test = np.load(feature_dir / "y_test_page.npy")

    return X_train, y_train, X_test, y_test


# ============================================================
# BUILD PIPELINE
# ============================================================

def build_pipeline():
    """
    Create the preprocessing + calibrated SVM-RBF pipeline.
    """

    svm_classifier = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        random_state=42
    )

    calibrated_svm = CalibratedClassifierCV(
        estimator=svm_classifier,
        method="sigmoid",
        cv=4,
        ensemble=False
    )

    pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                calibrated_svm
            ),
        ]
    )

    return pipeline


# ============================================================
# TRAIN PIPELINE
# ============================================================

def train_pipeline(pipeline, X_train, y_train):
    """
    Train the complete pipeline.
    """

    pipeline.fit(X_train, y_train)

    return pipeline


# ============================================================
# EVALUATE PIPELINE
# ============================================================

def evaluate_pipeline(pipeline, X_test, y_test):
    """
    Evaluate the trained pipeline on unseen test pages.
    """

    predictions = pipeline.predict(X_test)

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

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "classification_report": report,
        "confusion_matrix": matrix,
        "predictions": predictions,
    }


# ============================================================
# CREATE FEATURE NAMES
# ============================================================

def create_feature_names(feature_dimension):
    """
    Create readable names for CNN latent feature dimensions.

    Example:
        CNN_Feature_0001
        CNN_Feature_0002
        ...
    """

    feature_names = [
        f"CNN_Feature_{i + 1:04d}"
        for i in range(feature_dimension)
    ]

    return feature_names


# ============================================================
# CREATE LIME EXPLAINER
# ============================================================

def create_lime_explainer(
    X_train,
    y_train,
    feature_names
):
    """
    Create a LIME tabular explainer.
    """

    class_names = [
        str(label)
        for label in np.unique(y_train)
    ]

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42
    )

    return explainer


# ============================================================
# SELECT SAMPLE FOR LIME
# ============================================================

def select_lime_sample(
    y_test,
    predictions
):
    """
    Select a correctly classified test sample when possible.
    """

    correct_indices = np.where(
        predictions == y_test
    )[0]

    if len(correct_indices) > 0:
        return int(correct_indices[0])

    return 0


# ============================================================
# CREATE AND SAVE LIME EXPLANATION
# ============================================================

def create_lime_explanation(
    explainer,
    pipeline,
    X_test,
    y_test,
    predictions,
    output_dir,
    sample_index=None
):
    """
    Generate a LIME explanation and save it as both
    HTML and PNG.
    """

    if sample_index is None:
        sample_index = select_lime_sample(
            y_test,
            predictions
        )

    sample = X_test[sample_index]

    # --------------------------------------------------------
    # Generate LIME explanation
    # --------------------------------------------------------

    explanation = explainer.explain_instance(
        data_row=sample,
        predict_fn=pipeline.predict_proba,
        num_features=10,
        top_labels=1
    )

    # --------------------------------------------------------
    # Get predicted writer
    # --------------------------------------------------------

    predicted_label = predictions[sample_index]

    # --------------------------------------------------------
    # LIME stores the explanation using its own class index.
    # Use the label actually returned by LIME.
    # --------------------------------------------------------

    lime_label = explanation.top_labels[0]

    # --------------------------------------------------------
    # Save HTML explanation
    # --------------------------------------------------------

    html_path = (
        output_dir /
        "lime_explanation.html"
    )

    explanation.save_to_file(
        str(html_path)
    )

    # --------------------------------------------------------
    # Create Matplotlib LIME figure
    # --------------------------------------------------------

    figure = explanation.as_pyplot_figure(
        label=lime_label
    )

    # --------------------------------------------------------
    # Add project title
    # --------------------------------------------------------

    figure.suptitle(
        f"LIME Explanation - SVM-RBF\n"
        f"Predicted Writer: {predicted_label}",
        fontsize=14,
        y=1.02
    )

    figure.tight_layout()

    # --------------------------------------------------------
    # Save PNG
    # --------------------------------------------------------

    png_path = (
        output_dir /
        "lime_explanation.png"
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    return (
        sample_index,
        html_path,
        png_path
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    model_name,
    evaluation_results,
    y_test,
    sample_index,
    output_dir
):
    """
    Save numerical evaluation results.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    predictions = evaluation_results[
        "predictions"
    ]

    metrics_df = pd.DataFrame(
        [
            {
                "CNN_Model": model_name,
                "Classifier": "SVM_RBF",
                "Accuracy":
                    evaluation_results["accuracy"],
                "Precision_Macro":
                    evaluation_results["precision_macro"],
                "Recall_Macro":
                    evaluation_results["recall_macro"],
                "F1_Macro":
                    evaluation_results["f1_macro"],
                "LIME_Sample_Index":
                    sample_index,
                "LIME_True_Label":
                    y_test[sample_index],
                "LIME_Predicted_Label":
                    predictions[sample_index],
            }
        ]
    )

    metrics_df.to_csv(
        output_dir / "pipeline_results.csv",
        index=False
    )

    prediction_df = pd.DataFrame(
        {
            "True_Label": y_test,
            "Predicted_Label": predictions,
            "Correct": y_test == predictions
        }
    )

    prediction_df.to_csv(
        output_dir / "test_predictions.csv",
        index=False
    )

    np.savetxt(
        output_dir / "confusion_matrix.csv",
        evaluation_results["confusion_matrix"],
        delimiter=",",
        fmt="%d"
    )

    with open(
        output_dir / "classification_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            evaluation_results[
                "classification_report"
            ]
        )


# ============================================================
# RUN ONE MODEL
# ============================================================

def run_model(
    model_name,
    model_config
):
    """
    Run the complete Pipeline + LIME experiment
    for one CNN feature extractor.
    """

    feature_dir = model_config["feature_dir"]
    expected_dimension = model_config[
        "feature_dimension"
    ]

    output_dir = (
        RESULT_ROOT /
        model_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\n" + "=" * 70)
    print(
        f"Pipeline + LIME: {model_name}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_train, y_train, X_test, y_test = (
        load_features(feature_dir)
    )

    print(
        f"Training shape: {X_train.shape}"
    )

    print(
        f"Test shape:     {X_test.shape}"
    )

    # --------------------------------------------------------
    # Verify dimensions
    # --------------------------------------------------------

    if X_train.shape[1] != expected_dimension:

        raise ValueError(
            f"{model_name}: expected "
            f"{expected_dimension} features, "
            f"found {X_train.shape[1]}"
        )

    if X_test.shape[1] != expected_dimension:

        raise ValueError(
            f"{model_name}: expected "
            f"{expected_dimension} test features, "
            f"found {X_test.shape[1]}"
        )

    # --------------------------------------------------------
    # Create feature names
    # --------------------------------------------------------

    feature_names = create_feature_names(
        expected_dimension
    )

    # --------------------------------------------------------
    # Build pipeline
    # --------------------------------------------------------

    pipeline = build_pipeline()

    print("\nTraining pipeline...")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    pipeline = train_pipeline(
        pipeline,
        X_train,
        y_train
    )

    print(
        "Pipeline training completed."
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluation_results = evaluate_pipeline(
        pipeline,
        X_test,
        y_test
    )

    print(
        f"\nAccuracy : "
        f"{evaluation_results['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{evaluation_results['precision_macro']:.4f}"
    )

    print(
        f"Recall   : "
        f"{evaluation_results['recall_macro']:.4f}"
    )

    print(
        f"F1-score : "
        f"{evaluation_results['f1_macro']:.4f}"
    )

    # --------------------------------------------------------
    # Create LIME explainer
    # --------------------------------------------------------

    print(
        "\nCreating LIME explainer..."
    )

    explainer = create_lime_explainer(
        X_train,
        y_train,
        feature_names
    )

    # --------------------------------------------------------
    # Create LIME explanation
    # --------------------------------------------------------

    print(
        "Generating LIME explanation..."
    )

    (
        sample_index,
        html_path,
        png_path
    ) = create_lime_explanation(
        explainer,
        pipeline,
        X_test,
        y_test,
        evaluation_results["predictions"],
        output_dir
    )

    print(
        f"\nLIME sample index: "
        f"{sample_index}"
    )

    print(
        f"True writer: "
        f"{y_test[sample_index]}"
    )

    print(
        f"Predicted writer: "
        f"{evaluation_results['predictions'][sample_index]}"
    )

    print(
        f"\nHTML explanation:"
        f"\n{html_path}"
    )

    print(
        f"\nPNG explanation:"
        f"\n{png_path}"
    )

    # --------------------------------------------------------
    # Save numerical results
    # --------------------------------------------------------

    save_results(
        model_name,
        evaluation_results,
        y_test,
        sample_index,
        output_dir
    )

    print(
        f"\nAll results saved in:"
        f"\n{output_dir}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "PIPELINE + LIME EXPLAINABILITY EXPERIMENT"
    )
    print("=" * 70)

    RESULT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    for model_name, model_config in MODELS.items():

        run_model(
            model_name,
            model_config
        )

    print("\n" + "=" * 70)
    print(
        "ALL EXPERIMENTS COMPLETED"
    )
    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()