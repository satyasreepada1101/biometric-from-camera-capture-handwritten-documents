
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# PROJECT FOLDERS
# --------------------------------------------------

# This script is inside:
# 03_Transfer_learning/partial fine tuning models/

TRANSFER_DIR = Path(__file__).resolve().parent.parent

BASELINE_DIR = TRANSFER_DIR / "baseline models"
FINETUNING_DIR = TRANSFER_DIR / "partial fine tuning models"

# Model folder names
models = {
    "AlexNet": "alexnet",
    "EfficientNet-B0": "efficientnet_b0",
    "MobileNetV2": "mobilenetv2",
    "ResNet50": "resnet50"
}

# --------------------------------------------------
# FUNCTION TO READ ACCURACY
# --------------------------------------------------

def get_accuracy(model_folder):

    output_folder = model_folder / "output"

    # Find the evaluation summary CSV
    csv_files = list(
        output_folder.rglob("*metrics_summary*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No metrics summary CSV found in: {output_folder}"
        )

    # Use the most recently modified summary file
    csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)

    df = pd.read_csv(csv_path)

    if "accuracy" not in df.columns:
        raise ValueError(
            f"'accuracy' column not found in {csv_path}"
        )

    if df.empty:
        raise ValueError(f"The CSV is empty: {csv_path}")

    # Get the latest evaluation result
    accuracy = float(df.iloc[-1]["accuracy"])

    # Convert decimal accuracy to percentage if necessary
    if accuracy <= 1:
        accuracy *= 100

    print(f"Read: {csv_path}")
    print(f"Accuracy: {accuracy:.2f}%\n")

    return accuracy


# --------------------------------------------------
# EXTRACT BASELINE AND FINE-TUNING ACCURACIES
# --------------------------------------------------

baseline_accuracies = []
finetuned_accuracies = []

for model_name, folder_name in models.items():

    print(f"Processing {model_name}...")

    baseline_folder = BASELINE_DIR / folder_name
    finetuning_folder = FINETUNING_DIR / folder_name

    baseline_accuracy = get_accuracy(baseline_folder)
    finetuned_accuracy = get_accuracy(finetuning_folder)

    baseline_accuracies.append(baseline_accuracy)
    finetuned_accuracies.append(finetuned_accuracy)


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
    finetuned_accuracies,
    width,
    label="Partial Fine-Tuning"
)

# Graph labels
plt.xlabel("CNN Models", fontsize=12)
plt.ylabel("Test Accuracy (%)", fontsize=12)

plt.title(
    "Baseline Transfer Learning vs Partial Fine-Tuning",
    fontsize=14,
    fontweight="bold"
)

plt.xticks(x, model_names)
plt.ylim(0, 100)

plt.legend(fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Display accuracy values above each bar
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
        finetuned_accuracies[i] + 1,
        f"{finetuned_accuracies[i]:.2f}%",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

# --------------------------------------------------
# SAVE GRAPH
# --------------------------------------------------

graph_path = (
    TRANSFER_DIR /
    "baseline_vs_partial_finetuning_accuracy.png"
)

plt.savefig(graph_path, dpi=300)
plt.show()

print("----------------------------------------")
print("ACCURACY COMPARISON")
print("----------------------------------------")

for i, model in enumerate(model_names):
    print(f"\n{model}")
    print(f"Baseline: {baseline_accuracies[i]:.2f}%")
    print(f"Partial Fine-Tuning: {finetuned_accuracies[i]:.2f}%")
    print(
        f"Difference: "
        f"{finetuned_accuracies[i] - baseline_accuracies[i]:+.2f}%"
    )

print("\nGraph saved at:")
print(graph_path)
