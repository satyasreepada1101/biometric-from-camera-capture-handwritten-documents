
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
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# ==========================================
# SETTINGS
# ==========================================

PROJECT_DIR = r"C:\Users\prave\Desktop\college projects\biometric-from-camera-capture-handwritten-documents"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

METADATA_PATH = os.path.join(PROJECT_DIR, "patch_dataset", "metadata.csv")

if not os.path.exists(METADATA_PATH):
    METADATA_PATH = os.path.join(PROJECT_DIR, "metadata.csv")

CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR, "efficientnet_b0_full_finetuned.pth"
)

BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# ==========================================
# LOAD CHECKPOINT
# ==========================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

writer_to_idx = checkpoint["writer_to_idx"]
idx_to_writer = checkpoint["idx_to_writer"]
NUM_CLASSES = checkpoint["num_classes"]

class_names = [str(idx_to_writer[i]) for i in range(NUM_CLASSES)]

# ==========================================
# LOAD TEST DATA
# ==========================================

df = pd.read_csv(METADATA_PATH)

df = df[
    df["kept"].astype(str).str.lower().isin(["1", "1.0", "true"])
].copy()

def resolve_path(path):
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_DIR, path)

df["filepath"] = df["patch_path"].apply(resolve_path)
df = df[df["filepath"].apply(os.path.exists)].copy()

test_df = df[df["split"] == "test"].copy()

if test_df.empty:
    raise ValueError("No test images found.")

print("Test patches:", len(test_df))

# ==========================================
# DATASET
# ==========================================

class HandwritingDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image = Image.open(row["filepath"]).convert("RGB")
        label = writer_to_idx[row["writer"]]

        if self.transform:
            image = self.transform(image)

        return image, label

# ==========================================
# TRANSFORMS AND DATALOADER
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = HandwritingDataset(test_df, transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ==========================================
# LOAD EFFICIENTNET-B0
# ==========================================

model = models.efficientnet_b0(weights=None)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

criterion = nn.CrossEntropyLoss()

# ==========================================
# EVALUATION
# ==========================================

all_labels = []
all_predictions = []
total_loss = 0.0
total = 0

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

test_loss = total_loss / total
accuracy = accuracy_score(all_labels, all_predictions)

precision_macro = precision_score(
    all_labels, all_predictions, average="macro", zero_division=0
)
recall_macro = recall_score(
    all_labels, all_predictions, average="macro", zero_division=0
)
f1_macro = f1_score(
    all_labels, all_predictions, average="macro", zero_division=0
)

precision_weighted = precision_score(
    all_labels, all_predictions, average="weighted", zero_division=0
)
recall_weighted = recall_score(
    all_labels, all_predictions, average="weighted", zero_division=0
)
f1_weighted = f1_score(
    all_labels, all_predictions, average="weighted", zero_division=0
)

print("\n===================================")
print("EFFICIENTNET-B0 FULL FINE-TUNING RESULTS")
print("===================================")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Macro Precision: {precision_macro:.4f}")
print(f"Macro Recall: {recall_macro:.4f}")
print(f"Macro F1-Score: {f1_macro:.4f}")
print(f"Weighted Precision: {precision_weighted:.4f}")
print(f"Weighted Recall: {recall_weighted:.4f}")
print(f"Weighted F1-Score: {f1_weighted:.4f}")

# ==========================================
# CLASSIFICATION REPORT
# ==========================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    os.path.join(
        OUTPUT_DIR,
        "efficientnet_b0_full_finetuning_classification_report.csv"
    )
)

print("Classification report saved!")

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

plt.figure(figsize=(16, 14))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Writer")
plt.ylabel("Actual Writer")
plt.title("EfficientNet-B0 Full Fine-Tuning Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "efficientnet_b0_full_finetuning_confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("Confusion matrix saved!")

# ==========================================
# SAVE EVALUATION SUMMARY
# ==========================================

results = {
    "model": "EfficientNet-B0 Full Fine-Tuning",
    "test_images": len(test_df),
    "test_loss": test_loss,
    "accuracy": accuracy,
    "precision_macro": precision_macro,
    "recall_macro": recall_macro,
    "f1_macro": f1_macro,
    "precision_weighted": precision_weighted,
    "recall_weighted": recall_weighted,
    "f1_weighted": f1_weighted
}

summary_path = os.path.join(
    OUTPUT_DIR,
    "transfer_learning_metrics_summary.csv"
)

results_df = pd.DataFrame([results])

if os.path.exists(summary_path):
    old_df = pd.read_csv(summary_path)
    old_df = old_df[
        old_df["model"] != "EfficientNet-B0 Full Fine-Tuning"
    ]
    results_df = pd.concat([old_df, results_df], ignore_index=True)

results_df.to_csv(summary_path, index=False)

print("\nEvaluation completed successfully!")
print("Results saved in:", OUTPUT_DIR)
