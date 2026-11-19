
import os
import copy
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

PROJECT_DIR = r"C:\Users\prave\Desktop\college projects\biometric-from-camera-capture-handwritten-documents"

BATCH_SIZE = 16
NUM_EPOCHS = 10

FEATURE_LEARNING_RATE = 0.00001
CLASSIFIER_LEARNING_RATE = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(
    OUTPUT_DIR, "efficientnet_b0_full_finetuned.pth"
)

print("Using device:", DEVICE)

# --------------------------------------------------
# LOAD METADATA
# --------------------------------------------------

METADATA_PATH = os.path.join(
    PROJECT_DIR, "patch_dataset", "metadata.csv"
)

if not os.path.exists(METADATA_PATH):
    METADATA_PATH = os.path.join(PROJECT_DIR, "metadata.csv")

if not os.path.exists(METADATA_PATH):
    raise FileNotFoundError("metadata.csv not found. Check METADATA_PATH.")

df = pd.read_csv(METADATA_PATH)

df = df[df["kept"].astype(str).str.lower().isin(
    ["1", "1.0", "true"]
)].copy()

def resolve_path(path):
    path = str(path)

    if os.path.isabs(path):
        return path

    return os.path.join(PROJECT_DIR, path)

df["filepath"] = df["patch_path"].apply(resolve_path)
df = df[df["filepath"].apply(os.path.exists)].copy()

# --------------------------------------------------
# TRAIN / VALIDATION SPLIT
# --------------------------------------------------

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

train_pages = []
validation_pages = []

for writer in sorted(train_df["writer"].unique()):

    writer_pages = train_df[
        train_df["writer"] == writer
    ]["page"].unique()

    if len(writer_pages) < 2:
        raise ValueError(
            f"Not enough training pages for writer {writer}"
        )

    writer_train_pages, writer_val_pages = train_test_split(
        writer_pages,
        test_size=1,
        random_state=42
    )

    train_pages.extend(
        [(writer, page) for page in writer_train_pages]
    )

    validation_pages.extend(
        [(writer, page) for page in writer_val_pages]
    )

train_df = train_df[
    train_df.apply(
        lambda row: (row["writer"], row["page"]) in train_pages,
        axis=1
    )
].copy()

val_df = df[df["split"] == "train"].copy()

val_df = val_df[
    val_df.apply(
        lambda row: (row["writer"], row["page"]) in validation_pages,
        axis=1
    )
].copy()

print("Training patches:", len(train_df))
print("Validation patches:", len(val_df))
print("Test patches:", len(test_df))

# --------------------------------------------------
# WRITER LABELS
# --------------------------------------------------

writers = sorted(df["writer"].unique())

writer_to_idx = {
    writer: idx for idx, writer in enumerate(writers)
}

idx_to_writer = {
    idx: writer for writer, idx in writer_to_idx.items()
}

NUM_CLASSES = len(writer_to_idx)

print("Number of writers:", NUM_CLASSES)

# --------------------------------------------------
# DATASET
# --------------------------------------------------

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

# --------------------------------------------------
# TRANSFORMS
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_dataset = HandwritingDataset(train_df, transform)
val_dataset = HandwritingDataset(val_df, transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# --------------------------------------------------
# LOAD PRETRAINED EFFICIENTNET-B0
# --------------------------------------------------

model = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
)

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    NUM_CLASSES
)

model = model.to(DEVICE)

# --------------------------------------------------
# FULL FINE-TUNING
# --------------------------------------------------

# Unfreeze ALL parameters
for param in model.parameters():
    param.requires_grad = True

print("\nFull Fine-Tuning: ALL EfficientNet-B0 layers are trainable.")

optimizer = optim.Adam([
    {
        "params": model.features.parameters(),
        "lr": FEATURE_LEARNING_RATE
    },
    {
        "params": model.classifier.parameters(),
        "lr": CLASSIFIER_LEARNING_RATE
    }
])

criterion = nn.CrossEntropyLoss()

# --------------------------------------------------
# TRAINING
# --------------------------------------------------

history = {
    "epoch": [],
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

best_val_accuracy = 0.0

for epoch in range(NUM_EPOCHS):

    # ---------------- TRAINING ----------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / total
    train_accuracy = 100 * correct / total

    # ---------------- VALIDATION ----------------

    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss_total / val_total
    val_accuracy = 100 * val_correct / val_total

    # ---------------- SAVE HISTORY ----------------

    history["epoch"].append(epoch + 1)
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

    print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # ---------------- SAVE BEST MODEL ----------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save({
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "writer_to_idx": writer_to_idx,
            "idx_to_writer": idx_to_writer,
            "num_classes": NUM_CLASSES,
            "model_name": "EfficientNet-B0 Full Fine-Tuning",
            "best_val_accuracy": best_val_accuracy
        }, CHECKPOINT_PATH)

        print("Best model saved!")

# --------------------------------------------------
# SAVE TRAINING HISTORY
# --------------------------------------------------

history_df = pd.DataFrame(history)

history_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "efficientnet_b0_full_finetuning_history.csv"
    ),
    index=False
)

# --------------------------------------------------
# PLOT TRAINING CURVES
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(history["epoch"], history["train_loss"], label="Training Loss")
plt.plot(history["epoch"], history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("EfficientNet-B0 Full Fine-Tuning - Loss")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "efficientnet_b0_full_finetuning_loss.png"
    ),
    dpi=300
)

plt.show()

plt.figure(figsize=(8, 5))

plt.plot(history["epoch"], history["train_accuracy"], label="Training Accuracy")
plt.plot(history["epoch"], history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("EfficientNet-B0 Full Fine-Tuning - Accuracy")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "efficientnet_b0_full_finetuning_accuracy.png"
    ),
    dpi=300
)

plt.show()

print("\nTraining completed!")
print("Best Validation Accuracy:", best_val_accuracy)
print("Checkpoint saved at:", CHECKPOINT_PATH)
