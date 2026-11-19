
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


# ==========================================
# SETTINGS
# ==========================================

PROJECT_ROOT = r"C:\Users\prave\Desktop\college projects\biometric-from-camera-capture-handwritten-documents"

CSV_PATH = os.path.join(
    PROJECT_ROOT, "patch_dataset", "metadata.csv"
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "resnet50_writer_classifier.pth"
)

BATCH_SIZE = 16
MODEL_NAME = "ResNet50"


# ==========================================
# OUTPUT DIRECTORY
# ==========================================

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "transfer_learning_metrics_summary.csv"
)

CLASS_REPORT_CSV = os.path.join(
    OUTPUT_DIR,
    "resnet50_classification_report.csv"
)

CLASS_REPORT_TXT = os.path.join(
    OUTPUT_DIR,
    "resnet50_classification_report.txt"
)

CONFUSION_MATRIX_CSV = os.path.join(
    OUTPUT_DIR,
    "resnet50_confusion_matrix.csv"
)

CONFUSION_MATRIX_PNG = os.path.join(
    OUTPUT_DIR,
    "resnet50_confusion_matrix.png"
)


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==========================================
# LOAD CSV
# ==========================================

df = pd.read_csv(CSV_PATH)

# Keep only patches that were actually saved
df = df[df["kept"] == 1].copy()

df["filepath"] = df["patch_path"].apply(
    lambda p: os.path.normpath(
        os.path.join(PROJECT_ROOT, p)
    )
)

# Keep only image files that exist
df = df[
    df["filepath"].apply(os.path.isfile)
].copy()


# ==========================================
# SELECT TEST DATA
# ==========================================

test_df = df[
    df["split"] == "test"
].copy()

if test_df.empty:
    raise ValueError(
        "No test images found. Check the metadata paths."
    )

print("\n========================================")
print("TEST DATASET")
print("========================================")

print("Total saved patches:", len(df))
print("Test patches:", len(test_df))


# ==========================================
# LOAD MODEL CHECKPOINT
# ==========================================

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

writer_to_idx = checkpoint["writer_to_idx"]

idx_to_writer = {
    int(k): v
    for k, v in checkpoint["idx_to_writer"].items()
}

NUM_CLASSES = checkpoint["num_classes"]

print("Number of classes:", NUM_CLASSES)


# ==========================================
# TRANSFORM
# ==========================================

# No augmentation is applied to test images
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# DATASET
# ==========================================

class WriterDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image = Image.open(
            row["filepath"]
        ).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = writer_to_idx[row["writer"]]

        return (
            image,
            torch.tensor(label, dtype=torch.long)
        )


# ==========================================
# DATALOADER
# ==========================================

test_dataset = WriterDataset(
    test_df,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ==========================================
# RECREATE RESNET50 MODEL
# ==========================================

model = models.resnet50(
    weights=None
)

# Replace final classification layer
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    NUM_CLASSES
)


# ==========================================
# LOAD TRAINED WEIGHTS
# ==========================================

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


# ==========================================
# TEST
# ==========================================

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

        loss = criterion(
            outputs,
            labels
        )

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predictions = torch.max(
            outputs,
            1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        total += labels.size(0)


# ==========================================
# METRICS
# ==========================================

if total == 0:
    raise ValueError("The test dataset contains no samples.")

test_loss = running_loss / total

test_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision_macro = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

recall_macro = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

f1_macro = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

precision_weighted = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

recall_weighted = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

f1_weighted = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


# ==========================================
# PRINT TEST RESULTS
# ==========================================

print("\n========================================")
print("RESNET50 TEST RESULTS")
print("========================================")

print(f"Test Loss          : {test_loss:.4f}")
print(f"Accuracy           : {test_accuracy:.4f}")
print(f"Macro Precision    : {precision_macro:.4f}")
print(f"Macro Recall       : {recall_macro:.4f}")
print(f"Macro F1-score     : {f1_macro:.4f}")
print(f"Weighted Precision : {precision_weighted:.4f}")
print(f"Weighted Recall    : {recall_weighted:.4f}")
print(f"Weighted F1-score  : {f1_weighted:.4f}")


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

target_names = [
    str(idx_to_writer[i])
    for i in range(NUM_CLASSES)
]

report_text = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=target_names,
    zero_division=0
)

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(report_text)

# Save TXT report
with open(CLASS_REPORT_TXT, "w") as file:
    file.write(report_text)

# Save CSV report
report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=target_names,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

report_df.index.name = "class_or_average"

report_df.to_csv(
    CLASS_REPORT_CSV
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"Actual_{name}" for name in target_names
    ],
    columns=[
        f"Predicted_{name}" for name in target_names
    ]
)

cm_df.to_csv(
    CONFUSION_MATRIX_CSV
)


# ==========================================
# CONFUSION MATRIX PLOT
# ==========================================

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
    f"{MODEL_NAME} Writer Classification Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ==========================================
# SUMMARY METRICS CSV
# ==========================================

summary = pd.DataFrame([{
    "model": MODEL_NAME,
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


# ==========================================
# APPEND RESULTS
# ==========================================

# Replace an existing ResNet50 result
# instead of creating duplicate entries.
if os.path.exists(SUMMARY_CSV):

    previous = pd.read_csv(SUMMARY_CSV)

    previous = previous[
        previous["model"] != MODEL_NAME
    ]

    summary = pd.concat(
        [previous, summary],
        ignore_index=True
    )

summary.to_csv(
    SUMMARY_CSV,
    index=False
)


# ==========================================
# EVALUATION COMPLETE
# ==========================================

print("\n========================================")
print("EVALUATION COMPLETE")
print("========================================")

print("\nFiles saved successfully:")

print("Summary metrics CSV:", SUMMARY_CSV)
print("Classification report CSV:", CLASS_REPORT_CSV)
print("Classification report TXT:", CLASS_REPORT_TXT)
print("Confusion matrix CSV:", CONFUSION_MATRIX_CSV)
print("Confusion matrix PNG:", CONFUSION_MATRIX_PNG)
