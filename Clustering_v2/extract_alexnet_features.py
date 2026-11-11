# ============================================================
# 02_extract_alexnet_features.py
#
# Purpose:
#   Use PRETRAINED AlexNet as a feature extractor.
#
# IMPORTANT:
#   AlexNet is NOT trained.
#   No optimizer.
#   No loss.
#   No backward().
#   No modification of AlexNet weights.
#
# Input:
#   patch_dataset/train
#   patch_dataset/test
#
# Output:
#   features/X_train.npy
#   features/y_train.npy
#   features/X_test.npy
#   features/y_test.npy
#   features/train_metadata.csv
#   features/test_metadata.csv
# ============================================================

from pathlib import Path
import csv

import numpy as np

from PIL import Image

import torch
import torch.nn as nn

from torchvision import models, transforms


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_DATASET_DIR = Path("patch_dataset")

FEATURE_DIR = Path(__file__).resolve().parent / "features"

IMAGE_SIZE = 224

BATCH_SIZE = 32


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("ALEXNET FEATURE EXTRACTION")
print("=" * 60)

print(
    f"\nUsing device: {DEVICE}"
)


# ============================================================
# LOAD PRETRAINED ALEXNET
# ============================================================

print(
    "\nLoading pretrained AlexNet..."
)

weights = models.AlexNet_Weights.DEFAULT

alexnet = models.alexnet(
    weights=weights
)

alexnet = alexnet.to(DEVICE)

# VERY IMPORTANT:
# Evaluation mode because we are only extracting features.
alexnet.eval()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

    # Every patch becomes exactly 224 x 224.
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    # Convert PIL image to tensor.
    transforms.ToTensor(),

    # ImageNet normalization used by the pretrained model.
    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# GET WRITER DIRECTORIES
# ============================================================

def get_writer_dirs(split_dir):

    return sorted(
        [
            p for p in split_dir.iterdir()
            if p.is_dir()
        ],
        key=lambda x: x.name
    )


# ============================================================
# GET PATCH FILES
# ============================================================

def get_patch_files(split_dir):

    writer_dirs = get_writer_dirs(
        split_dir
    )

    samples = []

    for writer_id, writer_dir in enumerate(writer_dirs):

        writer_name = writer_dir.name

        patch_files = sorted(
            [
                p for p in writer_dir.iterdir()
                if p.suffix.lower()
                in {".jpg", ".jpeg", ".png"}
            ]
        )

        for patch_path in patch_files:

            samples.append(
                {
                    "path": patch_path,
                    "writer_id": writer_id,
                    "writer": writer_name
                }
            )

    return samples


# ============================================================
# EXTRACT FEATURES FROM ONE BATCH
# ============================================================

def extract_batch_features(images):

    """
    images:
        Tensor [batch, 3, 224, 224]

    Returns:
        Tensor [batch, 4096]
    """

    images = images.to(DEVICE)

    with torch.no_grad():

        # --------------------------------------------
        # AlexNet convolutional layers
        # --------------------------------------------

        x = alexnet.features(images)

        # --------------------------------------------
        # Adaptive average pooling
        # --------------------------------------------

        x = alexnet.avgpool(x)

        # --------------------------------------------
        # Flatten
        # --------------------------------------------

        x = torch.flatten(
            x,
            start_dim=1
        )

        # --------------------------------------------
        # AlexNet classifier EXCEPT final layer
        #
        # The final layer gives 1000 ImageNet classes.
        # We don't want that.
        #
        # classifier[:-1] gives the 4096-dimensional
        # representation before the final classification.
        # --------------------------------------------

        x = alexnet.classifier[:-1](x)

    return x.cpu()


# ============================================================
# EXTRACT FEATURES FOR A SPLIT
# ============================================================

def extract_features_for_split(split_name):

    split_dir = (
        PATCH_DATASET_DIR /
        split_name
    )

    print("\n")
    print("=" * 60)
    print(
        f"Processing {split_name.upper()}"
    )
    print("=" * 60)

    samples = get_patch_files(
        split_dir
    )

    print(
        f"\nNumber of patches: "
        f"{len(samples)}"
    )

    if len(samples) == 0:

        raise RuntimeError(
            f"No patches found in "
            f"{split_dir}"
        )

    all_features = []

    all_labels = []

    all_metadata = []

    # --------------------------------------------------------
    # Process batches manually
    # --------------------------------------------------------

    for start in range(
        0,
        len(samples),
        BATCH_SIZE
    ):

        batch_samples = samples[
            start:start + BATCH_SIZE
        ]

        batch_images = []

        for sample in batch_samples:

            image = Image.open(
                sample["path"]
            ).convert("RGB")

            image = transform(
                image
            )

            batch_images.append(
                image
            )

        # --------------------------------------------
        # Stack batch
        # --------------------------------------------

        batch_tensor = torch.stack(
            batch_images
        )

        # --------------------------------------------
        # Extract features
        # --------------------------------------------

        batch_features = (
            extract_batch_features(
                batch_tensor
            )
        )

        batch_features = (
            batch_features.numpy()
        )

        # --------------------------------------------
        # Store
        # --------------------------------------------

        all_features.append(
            batch_features
        )

        for i, sample in enumerate(
            batch_samples
        ):

            all_labels.append(
                sample["writer_id"]
            )

            all_metadata.append(
                {
                    "writer_id":
                        sample["writer_id"],

                    "writer":
                        sample["writer"],

                    "patch_path":
                        str(sample["path"])
                }
            )

        # --------------------------------------------
        # Progress
        # --------------------------------------------

        processed = min(
            start + BATCH_SIZE,
            len(samples)
        )

        print(
            f"Processed "
            f"{processed}/{len(samples)} patches"
        )

    # ========================================================
    # COMBINE FEATURES
    # ========================================================

    X = np.concatenate(
        all_features,
        axis=0
    )

    y = np.array(
        all_labels,
        dtype=np.int64
    )

    print(
        f"\nRaw patch feature shape: "
        f"{X.shape}"
    )

    print(
        "Expected second dimension: 4096"
    )

    # ========================================================
    # SAVE
    # ========================================================

    FEATURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        FEATURE_DIR /
        f"X_{split_name}_patches.npy",
        X
    )

    np.save(
        FEATURE_DIR /
        f"y_{split_name}_patches.npy",
        y
    )

    metadata_path = (
        FEATURE_DIR /
        f"{split_name}_patch_metadata.csv"
    )

    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "writer_id",
            "writer",
            "patch_path"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            all_metadata
        )

    print(
        f"\nSaved:"
    )

    print(
        FEATURE_DIR /
        f"X_{split_name}_patches.npy"
    )

    print(
        FEATURE_DIR /
        f"y_{split_name}_patches.npy"
    )

    return X, y, all_metadata


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check patch dataset
    # --------------------------------------------------------

    if not PATCH_DATASET_DIR.exists():

        raise FileNotFoundError(
            "patch_dataset does not exist.\n"
            "Run 01_create_patches.py first."
        )

    # --------------------------------------------------------
    # Train features
    # --------------------------------------------------------

    X_train_patches, y_train_patches, train_metadata = (
        extract_features_for_split(
            "train"
        )
    )

    # --------------------------------------------------------
    # Test features
    # --------------------------------------------------------

    X_test_patches, y_test_patches, test_metadata = (
        extract_features_for_split(
            "test"
        )
    )

    # --------------------------------------------------------
    # SAVE PATCH-LEVEL FEATURES
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        "\nTrain patch features:",
        X_train_patches.shape
    )

    print(
        "Test patch features:",
        X_test_patches.shape
    )

    print(
        "\nAlexNet was used ONLY for feature extraction."
    )

    print(
        "AlexNet weights were NOT trained or modified."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()