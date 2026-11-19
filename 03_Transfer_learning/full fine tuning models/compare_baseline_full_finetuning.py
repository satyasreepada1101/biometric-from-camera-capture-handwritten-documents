
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# FOLDER PATHS
# --------------------------------------------------

# Script is inside:
# 03_Transfer_learning/full fine tuning models/

TRANSFER_DIR = Path(__file__).resolve().parent

BASELINE_DIR = TRANSFER_DIR.parent / "baseline models"
FULL_FINETUNING_DIR = TRANSFER_DIR

OUTPUT_DIR = TRANSFER_DIR / "comparison_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# MODEL FOLDERS
# --------------------------------------------------

models = {
    "AlexNet": "alexnet",
    "EfficientNet-B0": "efficientnet_b0",
    "MobileNetV2": "mobilenetv2",
    "ResNet50": "resnet50"
}

# --------------------------------------------------
# FUNCTION TO EXTRACT ACCURACY
# --------------------------------------------------

def get_accuracy(model_folder):

    output_folder = model_folder / "output"

    csv_path = output_folder / "transfer_learning_metrics_summary.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Evaluation CSV not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    if "model" not in df.columns or "accuracy" not in df.columns:
        raise ValueError(
            f"Required columns 'model' and 'accuracy' "
            f"not found in {csv_path}"
        )

    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

    return df


def find_model_accuracy(model_folder, model_label):

    df = get_accuracy(model_folder)

    matching_rows = df[
        df["model"].astype(str).str.fullmatch(
            model_label, case=False, na=False
        )
    ]

    if matching_rows.empty:
        raise ValueError(
            f"Result '{model_label}' not found in "
            f"{model_folder / 'output'}"
        )

    # Get the latest matching evaluation
    accuracy = float(matching_rows.iloc[-1]["accuracy"])

    # Convert decimal accuracy to percentage
    if accuracy <= 1:
        accuracy *= 100

    return accuracy


# --------------------------------------------------
# EXTRACT BASELINE AND FULL FT ACCURACIES
# --------------------------------------------------

baseline_accuracies = []
full_finetuning_accuracies = []

for model_name, folder_name in models.items():

    print(f"\nProcessing {model_name}...")

    baseline_folder = BASELINE_DIR / folder_name
    full_folder = FULL_FINETUNING_DIR / folder_name

    baseline_accuracy = find_model_accuracy(
        baseline_folder,
        model_name
    )

    full_accuracy = find_model_accuracy(
        full_folder,
        f"{model_name} Full Fine-Tuning"
    )

    baseline_accuracies.append(baseline_accuracy)
    full_finetuning_accuracies.append(full_accuracy)

    print(f"Baseline Accuracy: {baseline_accuracy:.2f}%")
    print(f"Full Fine-Tuning Accuracy: {full_accuracy:.2f}%")


# --------------------------------------------------
# CREATE COMPARISON GRAPH
# --------------------------------------------------

model_names = list(models.keys())

x = np.arange(len(model_names))
width = 0.35

plt.figure(figsize=(12, 7))

plt.bar(
    x - width / 2,
    baseline_accuracies,
    width,
    label="Baseline Transfer Learning"
)

plt.bar(
    x + width / 2,
    full_finetuning_accuracies,
    width,
    label="Full Fine-Tuning"
)

plt.xlabel("CNN Models", fontsize=12)
plt.ylabel("Test Accuracy (%)", fontsize=12)

plt.title(
    "Baseline Transfer Learning vs Full Fine-Tuning",
    fontsize=14,
    fontweight="bold"
)

plt.xticks(x, model_names)
plt.ylim(0, 100)

plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Display accuracy values
for i in range(len(model_names)):

    plt.text(
        x[i] - width / 2,
        baseline_accuracies[i] + 1,
        f"{baseline_accuracies[i]:.2f}%",
        ha="center",
        fontsize=9
    )

    plt.text(
        x[i] + width / 2,
        full_finetuning_accuracies[i] + 1,
        f"{full_finetuning_accuracies[i]:.2f}%",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

# --------------------------------------------------
# SAVE GRAPH
# --------------------------------------------------

graph_path = (
    OUTPUT_DIR /
    "baseline_vs_full_finetuning_accuracy.png"
)

plt.savefig(graph_path, dpi=300)
plt.show()

# --------------------------------------------------
# PRINT COMPARISON RESULTS
# --------------------------------------------------

print("\n========================================")
print("BASELINE VS FULL FINE-TUNING COMPARISON")
print("========================================")

for i, model in enumerate(model_names):

    difference = (
        full_finetuning_accuracies[i] -
        baseline_accuracies[i]
    )

    print(f"\n{model}")
    print(f"Baseline: {baseline_accuracies[i]:.2f}%")
    print(f"Full Fine-Tuning: {full_finetuning_accuracies[i]:.2f}%")
    print(f"Difference: {difference:+.2f}%")

print("\nGraph saved at:")
print(graph_path)
