
# ============================================================
# EFFICIENTNET-B0 - COMBINED DATA AUGMENTATION
# Original + Rotation + Translation + Scaling + Brightness
# Five samples per original training patch
# ============================================================

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = r"C:\Users\prave\Desktop\college projects\biometric-from-camera-capture-handwritten-documents"

CSV_PATH = os.path.join(
    PROJECT_ROOT, "patch_dataset", "metadata.csv"
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(
    OUTPUT_DIR, "efficientnet_b0_combined_augmented.pth"
)

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001

NUM_WORKERS = 0
RANDOM_SEED = 42

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


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

if df.empty:
    raise ValueError("No valid image patches found.")

print("\nTotal saved patches:", len(df))


# ============================================================
# WRITER MAPPING
# ============================================================

writers = sorted(df["writer"].unique())

writer_to_idx = {
    writer: idx
    for idx, writer in enumerate(writers)
}

idx_to_writer = {
    idx: writer
    for writer, idx in writer_to_idx.items()
}

NUM_CLASSES = len(writers)

print("Number of writers:", NUM_CLASSES)


# ============================================================
# TRANSFORMS
# ============================================================

NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

original_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZE
])

rotation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    NORMALIZE
])

translation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.03, 0.03)
    ),
    transforms.ToTensor(),
    NORMALIZE
])

scaling_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(
        degrees=0,
        scale=(0.95, 1.05)
    ),
    transforms.ToTensor(),
    NORMALIZE
])

brightness_contrast_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1
    ),
    transforms.ToTensor(),
    NORMALIZE
])

AUGMENTATION_TRANSFORMS = [
    original_transform,
    rotation_transform,
    translation_transform,
    scaling_transform,
    brightness_contrast_transform
]


# ============================================================
# DATASETS
# ============================================================

class CombinedAugmentationDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df) * 5

    def __getitem__(self, index):

        original_index = index // 5
        augmentation_index = index % 5

        row = self.df.iloc[original_index]

        image = Image.open(
            row["filepath"]
        ).convert("RGB")

        image = AUGMENTATION_TRANSFORMS[
            augmentation_index
        ](image)

        label = writer_to_idx[row["writer"]]

        return image, torch.tensor(
            label, dtype=torch.long
        )


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


# ============================================================
# PAGE-LEVEL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

train_pool = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

if train_pool.empty or test_df.empty:
    raise ValueError(
        "Training or test split is empty. Check metadata.csv."
    )

train_parts = []
val_parts = []

for writer, group in train_pool.groupby("writer"):

    pages = group["page"].drop_duplicates().tolist()

    if len(pages) < 2:
        raise ValueError(
            f"Writer {writer} has fewer than 2 training pages."
        )

    train_pages, val_pages = train_test_split(
        pages,
        test_size=1,
        random_state=RANDOM_SEED
    )

    train_parts.append(
        group[group["page"].isin(train_pages)]
    )

    val_parts.append(
        group[group["page"].isin(val_pages)]
    )

train_df = pd.concat(train_parts, ignore_index=True)
val_df = pd.concat(val_parts, ignore_index=True)

print("\n========== DATASET SPLIT ==========")
print("Original training patches:", len(train_df))
print("Expanded training patches:", len(train_df) * 5)
print("Validation patches:", len(val_df))
print("Test patches:", len(test_df))


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = CombinedAugmentationDataset(train_df)

eval_transform = original_transform

val_dataset = WriterDataset(
    val_df, eval_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# LOAD PRETRAINED EFFICIENTNET-B0
# ============================================================

print("\nLoading pretrained EfficientNet-B0...")

weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(weights=weights)


# Freeze feature extraction layers
for param in model.features.parameters():
    param.requires_grad = False


# Replace classifier
num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    NUM_CLASSES
)

model = model.to(device)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING HISTORY
# ============================================================

history = {
    "Epoch": [],
    "Train Loss": [],
    "Validation Loss": [],
    "Train Accuracy": [],
    "Validation Accuracy": []
}

best_val_accuracy = -1.0
best_epoch = 0


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n========== STARTING EFFICIENTNET-B0 TRAINING ==========")

for epoch in range(NUM_EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

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


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_running_loss += (
                loss.item() * images.size(0)
            )

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

    val_loss = val_running_loss / val_total
    val_accuracy = 100 * val_correct / val_total


    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    history["Epoch"].append(epoch + 1)
    history["Train Loss"].append(train_loss)
    history["Validation Loss"].append(val_loss)
    history["Train Accuracy"].append(train_accuracy)
    history["Validation Accuracy"].append(val_accuracy)


    # --------------------------------------------------------
    # SAVE BEST CHECKPOINT
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "writer_to_idx": writer_to_idx,
                "idx_to_writer": idx_to_writer,
                "num_classes": NUM_CLASSES,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_accuracy
            },
            MODEL_SAVE_PATH
        )


    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.2f}% "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_accuracy:.2f}%"
    )


# ============================================================
# SAVE HISTORY CSV
# ============================================================

history_df = pd.DataFrame(history)

history_path = os.path.join(
    OUTPUT_DIR,
    "efficientnet_b0_combined_training_history.csv"
)

history_df.to_csv(history_path, index=False)


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    history_df["Epoch"],
    history_df["Train Loss"],
    marker="o",
    label="Training Loss"
)

plt.plot(
    history_df["Epoch"],
    history_df["Validation Loss"],
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("EfficientNet-B0 Combined Augmentation - Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "efficientnet_b0_combined_loss.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    history_df["Epoch"],
    history_df["Train Accuracy"],
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    history_df["Epoch"],
    history_df["Validation Accuracy"],
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("EfficientNet-B0 Combined Augmentation - Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "efficientnet_b0_combined_accuracy.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# COMPLETION
# ============================================================

print("\n========== TRAINING COMPLETE ==========")
print("Best epoch:", best_epoch)
print("Best validation accuracy:", best_val_accuracy)
print("Best model saved:", MODEL_SAVE_PATH)
print("Training history:", history_path)
print("Output folder:", OUTPUT_DIR)
