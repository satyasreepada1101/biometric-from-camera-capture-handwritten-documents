import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split


#settings
PROJECT_ROOT = r"C:\Users\prave\Desktop\college projects\biometric-from-camera-capture-handwritten-documents"
CSV_PATH = r"C:\Users\prave\Desktop\college projects\biometric-from-camera-capture-handwritten-documents\patch_dataset\metadata.csv"
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "alexnet_writer_classifier.pth"
)


# DEVICE
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# LOAD CSV
df = pd.read_csv(CSV_PATH)

# Keep only patches that were actually saved
df = df[df["kept"] == 1].copy()

# Convert patch_path to the filepath expected by WriterDataset
df["filepath"] = df["patch_path"].apply(
    lambda p: os.path.join(PROJECT_ROOT, p)
)

# Keep only paths that exist
df = df[df["filepath"].apply(os.path.isfile)].copy()

print("\nTotal saved patches:", len(df))

# WRITER MAPPING
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



# IMAGE TRANSFORMS


train_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# DATASET


class WriterDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.df = dataframe.reset_index(drop=True)

        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = row["filepath"]

        writer = row["writer"]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform:

            image = self.transform(image)

        label = writer_to_idx[writer]

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label




# SPLIT DATA BY PAGE



# Keep the original test pages untouched
train_pool = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()

train_parts = []
val_parts = []

# Split each writer's training pages into train and validation
for writer, group in train_pool.groupby("writer"):

    pages = group["page"].drop_duplicates().tolist()

    if len(pages) < 2:
        raise ValueError(
            f"Writer {writer} has fewer than 2 training pages."
        )

    train_pages, val_pages = train_test_split(
        pages,
        test_size=1,
        random_state=42
    )

    train_parts.append(
        group[group["page"].isin(train_pages)]
    )

    val_parts.append(
        group[group["page"].isin(val_pages)]
    )

train_df = pd.concat(train_parts, ignore_index=True)
val_df = pd.concat(val_parts, ignore_index=True)

print("\n========================================")
print("DATASET")
print("========================================")
print("Training   :", len(train_df))
print("Validation :", len(val_df))
print("Testing    :", len(test_df))




# CREATE DATASETS


train_dataset = WriterDataset(
    train_df,
    transform=train_transform
)

val_dataset = WriterDataset(
    val_df,
    transform=test_transform
)

test_dataset = WriterDataset(
    test_df,
    transform=test_transform
)



# CREATE DATALOADERS


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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)



# LOAD PRETRAINED ALEXNET


print("\nLoading pretrained AlexNet...")

weights = models.AlexNet_Weights.DEFAULT

model = models.alexnet(
    weights=weights
)



# FREEZE FEATURE EXTRACTION LAYERS


for param in model.features.parameters():

    param.requires_grad = False



# REPLACE FINAL CLASSIFIER


num_features = model.classifier[6].in_features

model.classifier[6] = nn.Linear(
    num_features,
    NUM_CLASSES
)



# MOVE MODEL TO DEVICE


model = model.to(device)



# LOSS FUNCTION


criterion = nn.CrossEntropyLoss()



# OPTIMIZER


optimizer = optim.Adam(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=LEARNING_RATE
)



# TRAINING HISTORY


train_losses = []

val_losses = []

train_accuracies = []

val_accuracies = []



# TRAINING LOOP


print("\n========================================")
print("STARTING TRAINING")
print("========================================")


for epoch in range(NUM_EPOCHS):

    
    # TRAIN
    

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0


    for images, labels in train_loader:

        images = images.to(device)

        labels = labels.to(device)


        # Clear gradients

        optimizer.zero_grad()


        # Forward pass

        outputs = model(images)


        # Calculate loss

        loss = criterion(
            outputs,
            labels
        )


        # Backpropagation

        loss.backward()


        # Update weights

        optimizer.step()


        # Statistics

        running_loss += (
            loss.item() * images.size(0)
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    epoch_train_loss = (
        running_loss / total
    )

    epoch_train_accuracy = (
        100 * correct / total
    )


    
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


            loss = criterion(
                outputs,
                labels
            )


            running_val_loss += (
                loss.item() * images.size(0)
            )


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    epoch_val_loss = (
        running_val_loss / val_total
    )

    epoch_val_accuracy = (
        100 * val_correct / val_total
    )


    
    # SAVE HISTORY
    

    train_losses.append(
        epoch_train_loss
    )

    val_losses.append(
        epoch_val_loss
    )

    train_accuracies.append(
        epoch_train_accuracy
    )

    val_accuracies.append(
        epoch_val_accuracy
    )


    
    # PRINT RESULTS
    

    print(
        f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
        f"| Train Loss: {epoch_train_loss:.4f} "
        f"| Train Acc: {epoch_train_accuracy:.2f}% "
        f"| Val Loss: {epoch_val_loss:.4f} "
        f"| Val Acc: {epoch_val_accuracy:.2f}%"
    )



# SAVE MODEL


torch.save(

    {
        "model_state_dict": model.state_dict(),

        "writer_to_idx": writer_to_idx,

        "idx_to_writer": idx_to_writer,

        "num_classes": NUM_CLASSES
    },

    MODEL_SAVE_PATH
)


print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    "Model saved to:",
    MODEL_SAVE_PATH
)