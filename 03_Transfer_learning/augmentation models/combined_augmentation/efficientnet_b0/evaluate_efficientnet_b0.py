
# ============================================================
# EFFICIENTNET-B0 - COMBINED AUGMENTATION EVALUATION
# Original, unaugmented test images
# ============================================================

import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        ".."
    )
)

CSV_PATH = os.path.join(
    PROJECT_ROOT, "patch_dataset", "metadata.csv"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(
    OUTPUT_DIR, "efficientnet_b0_combined_augmented.pth"
)

BATCH_SIZE = 16

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# CHECK CHECKPOINT
# ============================================================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}. "
        "Run train.py first."
    )


# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(CSV_PATH)

df = df[df["kept"] == 1].copy()

df["filepath"] = df["patch_path"].apply(
    lambda p: os.path.normpath(
        os.path.join(PROJECT_ROOT, p)
    )
)

df = df[df["filepath"].apply(os.path.isfile)].copy()

test_df = df[df["split"] == "test"].copy()

if test_df.empty:
    raise ValueError("No test images found.")


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

writer_to_idx = checkpoint["writer_to_idx"]

idx_to_writer = {
    int(k): v
    for k, v in checkpoint["idx_to_writer"].items()
}

NUM_CLASSES = checkpoint["num_classes"]

print("Number of writers:", NUM_CLASSES)
print("Test patches:", len(test_df))
print("Best epoch:", checkpoint["best_epoch"])


# ============================================================
# TEST TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

class WriterDataset(Dataset):

    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image = Image.open(
            row["filepath"]
        ).convert("RGB")

        image = self.transform(image)

        label = writer_to_idx[row["writer"]]

        return image, torch.tensor(
            label, dtype=torch.long
        )


test_dataset = WriterDataset(
    test_df,
    test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD EFFICIENTNET-B0
# ============================================================

model = models.efficientnet_b0(weights=None)

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    NUM_CLASSES
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()


# ============================================================
# TESTING
# ============================================================

criterion = nn.CrossEntropyLoss()

all_predictions = []
all_labels = []

running_loss = 0.0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predictions = torch.max(outputs, 1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        total += labels.size(0)


# ============================================================
# METRICS
# ============================================================

test_loss = running_loss / total

test_accuracy = accuracy_score(
    all_labels, all_predictions
)

precision_macro = precision_score(
    all_labels, all_predictions,
    average="macro", zero_division=0
)

recall_macro = recall_score(
    all_labels, all_predictions,
    average="macro", zero_division=0
)

f1_macro = f1_score(
    all_labels, all_predictions,
    average="macro", zero_division=0
)

precision_weighted = precision_score(
    all_labels, all_predictions,
    average="weighted", zero_division=0
)

recall_weighted = recall_score(
    all_labels, all_predictions,
    average="weighted", zero_division=0
)

f1_weighted = f1_score(
    all_labels, all_predictions,
    average="weighted", zero_division=0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========== EFFICIENTNET-B0 TEST RESULTS ==========")

print(f"Test Loss          : {test_loss:.4f}")
print(f"Accuracy           : {test_accuracy:.4f}")
print(f"Macro Precision    : {precision_macro:.4f}")
print(f"Macro Recall       : {recall_macro:.4f}")
print(f"Macro F1-score     : {f1_macro:.4f}")
print(f"Weighted Precision : {precision_weighted:.4f}")
print(f"Weighted Recall    : {recall_weighted:.4f}")
print(f"Weighted F1-score  : {f1_weighted:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

target_names = [
    idx_to_writer[i]
    for i in range(NUM_CLASSES)
]

report_text = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=target_names,
    zero_division=0
)

report_dict = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=target_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report_dict).transpose()

report_df.index.name = "class_or_average"

report_csv_path = os.path.join(
    OUTPUT_DIR,
    "efficientnet_b0_combined_classification_report.csv"
)

report_df.to_csv(report_csv_path)

report_txt_path = os.path.join(
    OUTPUT_DIR,
    "efficientnet_b0_combined_classification_report.txt"
)

with open(report_txt_path, "w", encoding="utf-8") as f:
    f.write(report_text)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual_{name}" for name in target_names],
    columns=[f"Predicted_{name}" for name in target_names]
)

cm_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "efficientnet_b0_combined_confusion_matrix.csv"
    )
)


# ============================================================
# CONFUSION MATRIX GRAPH
# ============================================================

plt.figure(figsize=(16, 14))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=target_names,
    yticklabels=target_names
)

plt.xlabel("Predicted Writer")
plt.ylabel("Actual Writer")

plt.title(
    "EfficientNet-B0 - Combined Augmentation Confusion Matrix"
)

plt.tight_layout()

cm_png_path = os.path.join(
    OUTPUT_DIR,
    "efficientnet_b0_combined_confusion_matrix.png"
)

plt.savefig(
    cm_png_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# SUMMARY CSV
# ============================================================

summary = pd.DataFrame([{
    "model": "EfficientNet-B0",
    "experiment": "Combined_Augmentation",
    "test_images": total,
    "test_loss": test_loss,
    "accuracy": test_accuracy,
    "precision_macro": precision_macro,
    "recall_macro": recall_macro,
    "f1_macro": f1_macro,
    "precision_weighted": precision_weighted,
    "recall_weighted": recall_weighted,
    "f1_weighted": f1_weighted
}])

summary_path = os.path.join(
    OUTPUT_DIR,
    "efficientnet_b0_combined_metrics_summary.csv"
)

summary.to_csv(summary_path, index=False)


# ============================================================
# COMPLETION
# ============================================================

print("\n========== EVALUATION COMPLETE ==========")
print("Metrics summary:", summary_path)
print("Classification report:", report_csv_path)
print("Confusion matrix:", cm_png_path)
print("Output folder:", OUTPUT_DIR)
