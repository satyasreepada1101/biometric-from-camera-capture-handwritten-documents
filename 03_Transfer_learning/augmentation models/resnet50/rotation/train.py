import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============================================================
# MODEL: ResNet50
# AUGMENTATION TYPE: Rotation (±5 degrees)
# Only this augmentation is applied to training images.
# ============================================================

PROJECT_ROOT = r"C:\Users\prave\Desktop\college projects\biometric-from-camera-capture-handwritten-documents"
CSV_PATH = os.path.join(PROJECT_ROOT, "patch_dataset", "metadata.csv")

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
RANDOM_SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(
    OUTPUT_DIR, "resnet50_rotation_writer_classifier.pth"
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Augmentation:", "Rotation (±5 degrees)")

# ------------------------------------------------------------
# LOAD METADATA
# ------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
df = df[df["kept"] == 1].copy()

df["filepath"] = df["patch_path"].apply(
    lambda p: os.path.normpath(os.path.join(PROJECT_ROOT, p))
)
df = df[df["filepath"].apply(os.path.isfile)].copy()

if df.empty:
    raise ValueError("No patch images found. Check PROJECT_ROOT and metadata.csv.")

# ------------------------------------------------------------
# WRITER MAPPING
# ------------------------------------------------------------
writers = sorted(df["writer"].unique())
writer_to_idx = {writer: idx for idx, writer in enumerate(writers)}
idx_to_writer = {idx: writer for writer, idx in writer_to_idx.items()}
NUM_CLASSES = len(writers)

# ------------------------------------------------------------
# PAGE-LEVEL SPLIT
# Keep original test pages untouched.
# Split each writer's original train pages into train/validation.
# ------------------------------------------------------------
train_pool = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

train_parts, val_parts = [], []

for writer, group in train_pool.groupby("writer"):
    pages = group["page"].drop_duplicates().tolist()

    if len(pages) < 2:
        raise ValueError(f"Writer {writer} has fewer than 2 training pages.")

    train_pages, val_pages = train_test_split(
        pages, test_size=1, random_state=RANDOM_SEED
    )

    train_parts.append(group[group["page"].isin(train_pages)])
    val_parts.append(group[group["page"].isin(val_pages)])

train_df = pd.concat(train_parts, ignore_index=True)
val_df = pd.concat(val_parts, ignore_index=True)

if train_df.empty or val_df.empty or test_df.empty:
    raise ValueError("Train, validation, or test split is empty.")

print("\n========================================")
print("DATASET")
print("========================================")
print("Training   :", len(train_df))
print("Validation :", len(val_df))
print("Testing    :", len(test_df))
print("Writers    :", NUM_CLASSES)

# ------------------------------------------------------------
# IMAGE TRANSFORMS
# Augmentation is applied ONLY to training images.
# ------------------------------------------------------------
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    normalize
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

# ------------------------------------------------------------
# DATASET
# ------------------------------------------------------------
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


train_dataset = WriterDataset(train_df, transform=train_transform)
val_dataset = WriterDataset(val_df, transform=eval_transform)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# ------------------------------------------------------------
# PRETRAINED RESNET50
# Same setup as your baseline:
# freeze pretrained layers and train only the final fc layer.
# ------------------------------------------------------------
print("\nLoading pretrained ResNet50...")
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, NUM_CLASSES)

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE
)

# ------------------------------------------------------------
# TRAINING
# ------------------------------------------------------------
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

print("\nStarting training...")

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
        predictions = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    epoch_train_loss = running_loss / total
    epoch_train_accuracy = 100 * correct / total

    # VALIDATION
    model.eval()
    running_val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            val_total += labels.size(0)
            val_correct += (predictions == labels).sum().item()

    epoch_val_loss = running_val_loss / val_total
    epoch_val_accuracy = 100 * val_correct / val_total

    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    train_accuracies.append(epoch_train_accuracy)
    val_accuracies.append(epoch_val_accuracy)

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"| Train Loss: {epoch_train_loss:.4f} "
        f"| Train Acc: {epoch_train_accuracy:.2f}% "
        f"| Val Loss: {epoch_val_loss:.4f} "
        f"| Val Acc: {epoch_val_accuracy:.2f}%"
    )

# ------------------------------------------------------------
# SAVE TRAINING HISTORY AND CURVES
# ------------------------------------------------------------
history_df = pd.DataFrame({
    "Epoch": range(1, NUM_EPOCHS + 1),
    "Train Loss": train_losses,
    "Validation Loss": val_losses,
    "Train Accuracy": train_accuracies,
    "Validation Accuracy": val_accuracies
})

history_path = os.path.join(OUTPUT_DIR, "training_history.csv")
history_df.to_csv(history_path, index=False)

plt.figure(figsize=(8, 5))
plt.plot(history_df["Epoch"], history_df["Train Loss"],
         marker="o", label="Training Loss")
plt.plot(history_df["Epoch"], history_df["Validation Loss"],
         marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet50 - Rotation (±5 degrees) - Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curve.png"), dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history_df["Epoch"], history_df["Train Accuracy"],
         marker="o", label="Training Accuracy")
plt.plot(history_df["Epoch"], history_df["Validation Accuracy"],
         marker="o", label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("ResNet50 - Rotation (±5 degrees) - Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_curve.png"), dpi=300)
plt.close()

# ------------------------------------------------------------
# SAVE MODEL CHECKPOINT
# ------------------------------------------------------------
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "writer_to_idx": writer_to_idx,
        "idx_to_writer": idx_to_writer,
        "num_classes": NUM_CLASSES,
        "model_name": "ResNet50",
        "augmentation": "Rotation (±5 degrees)",
        "random_seed": RANDOM_SEED
    },
    MODEL_SAVE_PATH
)

print("\nTraining complete.")
print("Training history:", history_path)
print("Model saved to:", MODEL_SAVE_PATH)
