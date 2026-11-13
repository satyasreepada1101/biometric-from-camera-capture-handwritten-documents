import os
import pandas as pd
import torch
import torch.nn as nn
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
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# MODEL: MobileNetV2
# AUGMENTATION TYPE: Brightness and Contrast (0.1)
# Test images are evaluated WITHOUT augmentation.
# ============================================================

PROJECT_ROOT = r"C:\Users\prave\Desktop\college projects\biometric-from-camera-capture-handwritten-documents"
CSV_PATH = os.path.join(PROJECT_ROOT, "patch_dataset", "metadata.csv")
BATCH_SIZE = 16
MODEL_NAME = "MobileNetV2"
AUGMENTATION_NAME = "Brightness and Contrast (0.1)"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
MODEL_PATH = os.path.join(
    OUTPUT_DIR, "mobilenetv2_brightness_contrast_writer_classifier.pth"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------------------------------------------------
# LOAD ORIGINAL TEST SET
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df = df[df["kept"] == 1].copy()

df["filepath"] = df["patch_path"].apply(
    lambda p: os.path.normpath(os.path.join(PROJECT_ROOT, p))
)
df = df[df["filepath"].apply(os.path.isfile)].copy()

test_df = df[df["split"] == "test"].copy()

if test_df.empty:
    raise ValueError("No test images found. Check metadata paths.")

# ------------------------------------------------------------
# LOAD TRAINED CHECKPOINT
# ------------------------------------------------------------
checkpoint = torch.load(MODEL_PATH, map_location=device)

writer_to_idx = checkpoint["writer_to_idx"]
idx_to_writer = {
    int(k): v for k, v in checkpoint["idx_to_writer"].items()
}
NUM_CLASSES = checkpoint["num_classes"]

# Test images are not augmented.
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class WriterDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        with Image.open(row["filepath"]) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = writer_to_idx[row["writer"]]
        return image, torch.tensor(label, dtype=torch.long)


test_dataset = WriterDataset(test_df, transform=test_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ------------------------------------------------------------
# RECREATE MOBILENETV2 AND LOAD CHECKPOINT
# ------------------------------------------------------------
model = models.mobilenet_v2(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, NUM_CLASSES)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# ------------------------------------------------------------
# TEST EVALUATION
# ------------------------------------------------------------
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
        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total += labels.size(0)

test_loss = running_loss / total
test_accuracy = accuracy_score(all_labels, all_predictions)
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

print("\n========== TEST RESULTS ==========")
print("Augmentation       :", AUGMENTATION_NAME)
print("Test images        :", total)
print(f"Test Loss          : {test_loss:.4f}")
print(f"Accuracy           : {test_accuracy:.4f}")
print(f"Macro Precision    : {precision_macro:.4f}")
print(f"Macro Recall       : {recall_macro:.4f}")
print(f"Macro F1-score     : {f1_macro:.4f}")
print(f"Weighted Precision : {precision_weighted:.4f}")
print(f"Weighted Recall    : {recall_weighted:.4f}")
print(f"Weighted F1-score  : {f1_weighted:.4f}")

# ------------------------------------------------------------
# SAVE CLASSIFICATION REPORT
# ------------------------------------------------------------
target_names = [
    idx_to_writer[i] for i in range(NUM_CLASSES)
]

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=target_names,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    os.path.join(OUTPUT_DIR, "classification_report.csv"),
    index_label="class_or_average"
)

report_text = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=target_names,
    zero_division=0
)

with open(
    os.path.join(OUTPUT_DIR, "classification_report.txt"),
    "w",
    encoding="utf-8"
) as f:
    f.write(f"Model: {MODEL_NAME}\n")
    f.write(f"Augmentation: {AUGMENTATION_NAME}\n\n")
    f.write(report_text)

# ------------------------------------------------------------
# SAVE CONFUSION MATRIX
# ------------------------------------------------------------
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
cm_df.to_csv(os.path.join(OUTPUT_DIR, "confusion_matrix.csv"))

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
plt.title(f"MobileNetV2 - {AUGMENTATION_NAME} - Confusion Matrix")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# ------------------------------------------------------------
# SAVE METRICS SUMMARY
# ------------------------------------------------------------
summary = pd.DataFrame([{
    "model": MODEL_NAME,
    "experiment": AUGMENTATION_NAME,
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

summary.to_csv(
    os.path.join(OUTPUT_DIR, "metrics_summary.csv"),
    index=False
)

print("\nEvaluation outputs saved in:", OUTPUT_DIR)
