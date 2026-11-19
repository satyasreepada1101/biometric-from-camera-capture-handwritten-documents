"""
Create baseline-vs-augmentation comparison tables and plots.

1. Extract outputs.zip.
2. Keep this script beside the extracted "outputs" folder, OR edit OUTPUTS_DIR.
3. Run: python compare_augmentations.py

The script uses the exact folder names and CSV naming patterns found in outputs.zip.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# This script can sit inside "03_Transfer_learning/augmentation models".
# It searches recursively for each output-* folder, including folders nested
# inside the individual model/augmentation folders and the separate baseline folder.
SCRIPT_DIR = Path(__file__).resolve().parent
SEARCH_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "augmentation_comparison_results"

# Exact directory names from the uploaded ZIP.
EXPERIMENTS = {
    "AlexNet": {
        "Baseline": "output-baseline-alexnet",
        "Rotation": "output-alexnet-rotation",
        "Translation": "output-alexnet-translation",
        "Scaling": "output-alexnet-scaling",
        "Brightness/Contrast": "output-alexnet-brightness",
    },
    "EfficientNet-B0": {
        "Baseline": "output-efficientnet-baseline",
        "Rotation": "output-efficientnet-rotation",
        "Translation": "output-efficentnet-translation",
        "Scaling": "output-efficientnet-scaling",
        "Brightness/Contrast": "output-efficientnet-brigtness",
    },
    "MobileNetV2": {
        "Baseline": "output-mobilenet-baseline",
        "Rotation": "output-mobilenet-rotation",
        "Translation": "output-mobilenet-translation",
        "Scaling": "output-mobilenet-scaling",
        "Brightness/Contrast": "output-mobilevnet-brigthness",
    },
    "ResNet50": {
        "Baseline": "output-resnet50-baseline",
        "Rotation": "output-resnet50-rotation",
        "Translation": "output-resnet-translation",
        "Scaling": "output-resnet50-scalin",
        "Brightness/Contrast": "output-resnet50-brigthness",
    },
}

METRIC_COLUMNS = [
    "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted",
]

def find_experiment_folder(folder_name: str):
    matches = [p for p in SEARCH_ROOT.rglob(folder_name) if p.is_dir()]
    if not matches:
        raise FileNotFoundError(
            f"Could not find experiment folder '{folder_name}' under: {SEARCH_ROOT}"
        )
    if len(matches) > 1:
        # Prefer the first exact folder-name match; normally there should be only one.
        print(f"Warning: multiple folders named '{folder_name}' found. Using: {matches[0]}")
    return matches[0]

def find_csv(folder: Path, preferred_name: str):
    path = folder / preferred_name
    if path.exists():
        return path
    raise FileNotFoundError(f"Missing {preferred_name} in {folder}")

def read_metrics(folder: Path, is_baseline: bool):
    filename = (
        "transfer_learning_metrics_summary.csv"
        if is_baseline else "metrics_summary.csv"
    )
    path = find_csv(folder, filename)
    df = pd.read_csv(path)
    required = set(METRIC_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing metric columns: {sorted(missing)}")
    return df.iloc[0].to_dict()

def read_history(folder: Path):
    # Baseline filenames differ by model; EfficientNet also has a double .csv suffix.
    matches = sorted(folder.glob("*training_history.csv*"))
    if not matches:
        raise FileNotFoundError(f"No training-history CSV found in {folder}")
    df = pd.read_csv(matches[0])

    # The uploaded files use these exact column headings.
    required = ["Epoch", "Train Loss", "Validation Loss"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{matches[0]} is missing columns: {missing}")
    return df

def save_metric_bar_chart(comparison, metric, title, filename):
    models = list(EXPERIMENTS.keys())
    experiments = ["Baseline", "Rotation", "Translation", "Scaling", "Brightness/Contrast"]
    x = np.arange(len(models))
    width = 0.15

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, experiment in enumerate(experiments):
        values = []
        for model in models:
            row = comparison[
                (comparison["Model"] == model) &
                (comparison["Experiment"] == experiment)
            ]
            values.append(float(row.iloc[0][metric]) if not row.empty else np.nan)

        ax.bar(x + (i - 2) * width, values, width, label=experiment)

    ax.set_title(title)
    ax.set_xlabel("CNN Model")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=300)
    plt.close(fig)

def save_loss_comparison(model, augmentation, baseline_history, augmented_history):
    out_dir = RESULTS_DIR / "loss_comparisons" / model.replace("-", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(baseline_history["Epoch"], baseline_history["Train Loss"],
            marker="o", label="Baseline - Train Loss")
    ax.plot(baseline_history["Epoch"], baseline_history["Validation Loss"],
            marker="o", label="Baseline - Validation Loss")
    ax.plot(augmented_history["Epoch"], augmented_history["Train Loss"],
            marker="s", label=f"{augmentation} - Train Loss")
    ax.plot(augmented_history["Epoch"], augmented_history["Validation Loss"],
            marker="s", label=f"{augmentation} - Validation Loss")

    ax.set_title(f"{model}: Baseline vs {augmentation} - Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xticks(sorted(set(baseline_history["Epoch"]).union(
        set(augmented_history["Epoch"]))))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    safe_name = augmentation.lower().replace("/", "_").replace(" ", "_")
    fig.savefig(out_dir / f"baseline_vs_{safe_name}_loss.png", dpi=300)
    plt.close(fig)

def main():
    

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    histories = {}

    for model, experiments in EXPERIMENTS.items():
        for experiment, folder_name in experiments.items():
            folder = find_experiment_folder(folder_name)

            is_baseline = experiment == "Baseline"
            metrics = read_metrics(folder, is_baseline)
            row = {
                "Model": model,
                "Experiment": experiment,
                "Test Images": metrics.get("test_images", np.nan),
                "Test Loss": metrics.get("test_loss", np.nan),
            }
            for column in METRIC_COLUMNS:
                row[column] = metrics[column]
            rows.append(row)

            histories[(model, experiment)] = read_history(folder)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(RESULTS_DIR / "overall_classification_comparison.csv", index=False)

    save_metric_bar_chart(
        comparison, "accuracy",
        "Test Accuracy: Baseline vs Data Augmentation",
        "augmentation_accuracy_comparison.png"
    )
    save_metric_bar_chart(
        comparison, "f1_macro",
        "Macro F1-Score: Baseline vs Data Augmentation",
        "augmentation_macro_f1_comparison.png"
    )

    # 4 models x 4 augmentations = 16 loss comparison plots.
    for model, experiments in EXPERIMENTS.items():
        baseline_history = histories[(model, "Baseline")]
        for augmentation in ["Rotation", "Translation", "Scaling", "Brightness/Contrast"]:
            save_loss_comparison(
                model, augmentation, baseline_history,
                histories[(model, augmentation)]
            )

    print("\nCOMPLETED SUCCESSFULLY")
    print(f"Combined CSV: {RESULTS_DIR / 'overall_classification_comparison.csv'}")
    print(f"Accuracy plot: {RESULTS_DIR / 'augmentation_accuracy_comparison.png'}")
    print(f"Macro F1 plot: {RESULTS_DIR / 'augmentation_macro_f1_comparison.png'}")
    print(f"Loss plots: {RESULTS_DIR / 'loss_comparisons'}")
    print("\nTotal expected outputs: 1 combined CSV + 2 comparison plots + 16 loss plots.")

if __name__ == "__main__":
    main()
