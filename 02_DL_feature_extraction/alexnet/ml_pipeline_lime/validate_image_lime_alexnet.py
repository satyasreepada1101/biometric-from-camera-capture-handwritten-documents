import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from PIL import Image
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_DIR = (
    PROJECT_ROOT
    / "02_DL_feature_extraction"
    / "alexnet"
    / "features"
)

ML_DATASET_DIR = (
    PROJECT_ROOT
    / "ML_dataset"
)

PATCH_DATASET_DIR = (
    PROJECT_ROOT
    / "patch_dataset"
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("ALEXNET IMAGE LIME FEATURE VALIDATION")
print("=" * 60)

print(
    f"\nUsing device: {DEVICE}"
)


# ============================================================
# LOAD SAVED PAGE-LEVEL FEATURES
# ============================================================

print(
    "\nLoading saved page-level AlexNet features..."
)

X_test = np.load(
    FEATURE_DIR
    / "X_test_page.npy"
)

y_test = np.load(
    FEATURE_DIR
    / "y_test_page.npy"
)

print(
    "Saved X_test shape:",
    X_test.shape
)

print(
    "Saved y_test shape:",
    y_test.shape
)


# ============================================================
# LOAD TEST METADATA
# ============================================================

metadata = pd.read_csv(
    FEATURE_DIR
    / "test_patch_metadata.csv"
)


# ============================================================
# PREPARE METADATA
# ============================================================

metadata = metadata.copy()

metadata["patch_number"] = (
    metadata["patch_path"]
    .apply(
        lambda x:
        int(
            re.search(
                r"_patch_(\d+)",
                Path(x).name
            ).group(1)
        )
    )
)

metadata["page_key"] = (
    metadata["patch_path"]
    .apply(
        lambda x:
        Path(x).name.rsplit(
            "_patch_",
            1
        )[0]
    )
)


# ============================================================
# GROUP PATCHES INTO PAGES
# ============================================================

grouped_test = metadata.groupby(
    [
        "writer_id",
        "writer",
        "page_key"
    ],
    sort=False
)

test_page_information = []

for (
    writer_id,
    writer,
    page_key
), group in grouped_test:

    test_page_information.append(
        {
            "writer_id": int(writer_id),

            "writer": writer,

            "page_key": page_key,

            "patch_numbers":
                sorted(
                    group[
                        "patch_number"
                    ].tolist()
                )
        }
    )


print(
    "\nNumber of test pages:",
    len(test_page_information)
)


# ============================================================
# CHECK PAGE ORDER
# ============================================================

if len(test_page_information) != len(X_test):

    raise RuntimeError(
        "\nMismatch between page metadata "
        "and X_test_page.npy\n"
        f"Metadata pages: "
        f"{len(test_page_information)}\n"
        f"X_test pages: "
        f"{len(X_test)}"
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

alexnet = alexnet.to(
    DEVICE
)

alexnet.eval()


# ============================================================
# SAME ALEXNET PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    ),

    transforms.ToTensor(),

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
# EXTRACT ALEXNET FEATURES
# ============================================================

def extract_alexnet_features(images):

    tensors = []

    for image in images:

        image = image.convert(
            "RGB"
        )

        image = transform(
            image
        )

        tensors.append(
            image
        )

    if len(tensors) == 0:

        return np.empty(
            (
                0,
                4096
            ),
            dtype=np.float32
        )

    all_features = []

    for start in range(
        0,
        len(tensors),
        BATCH_SIZE
    ):

        batch = torch.stack(
            tensors[
                start:
                start + BATCH_SIZE
            ]
        )

        batch = batch.to(
            DEVICE
        )

        with torch.no_grad():

            x = alexnet.features(
                batch
            )

            x = alexnet.avgpool(
                x
            )

            x = torch.flatten(
                x,
                start_dim=1
            )

            x = alexnet.classifier[
                :-1
            ](
                x
            )

        all_features.append(
            x.cpu().numpy()
        )

    return np.concatenate(
        all_features,
        axis=0
    )


# ============================================================
# RECONSTRUCT ONE PAGE FEATURE
# USING THE EXACT SAVED PATCH JPEG FILES
# ============================================================

def reconstruct_page_feature(
    writer,
    page_key,
    patch_numbers
):

    patch_images = []

    # --------------------------------------------------------
    # Make sure page name has an extension
    # --------------------------------------------------------

    page_filename = page_key

    if not page_filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp")
    ):
        page_filename += ".jpg"

    # --------------------------------------------------------
    # Test patch directory
    # --------------------------------------------------------

    patch_writer_dir = (
        PATCH_DATASET_DIR
        / "test"
        / writer
    )

    # --------------------------------------------------------
    # Load EXACT saved patch JPEG files
    #
    # Dataset contains two naming conventions:
    #
    #   page_1_patch_01.jpg
    #   page_5.jpg_patch_01.jpg
    #
    # Try both automatically.
    # --------------------------------------------------------

    for patch_number in patch_numbers:

        page_stem = Path(page_filename).stem

        candidate_paths = [
            patch_writer_dir / (
                f"{page_stem}_patch_{patch_number:02d}.jpg"
            ),
            patch_writer_dir / (
                f"{page_filename}_patch_{patch_number:02d}.jpg"
            ),
        ]

        patch_path = None

        for candidate in candidate_paths:

            if candidate.exists():
                patch_path = candidate
                break

        if patch_path is None:

            raise FileNotFoundError(
                "Patch not found. Tried:\n"
                + "\n".join(
                    str(p)
                    for p in candidate_paths
                )
            )

        # This is the exact JPEG file that the
        # original AlexNet feature extraction used.

        patch = Image.open(
            patch_path
        ).convert(
            "RGB"
        )

        patch_images.append(
            patch
        )

    # --------------------------------------------------------
    # Extract AlexNet features from all retained patches
    # --------------------------------------------------------

    patch_features = (
        extract_alexnet_features(
            patch_images
        )
    )

    # --------------------------------------------------------
    # Exact same page-level aggregation
    # --------------------------------------------------------

    page_feature = (
        patch_features.mean(
            axis=0
        )
    )

    return page_feature
# ============================================================
# VALIDATE ALL TEST PAGES
# ============================================================

print(
    "\n" +
    "=" * 60
)

print(
    "VALIDATING PAGE-LEVEL FEATURES"
)

print(
    "=" * 60
)

max_difference = 0.0

all_close = True

for index, page_info in enumerate(
    test_page_information
):

    writer = page_info[
        "writer"
    ]

    page_key = page_info[
        "page_key"
    ]

    patch_numbers = page_info[
        "patch_numbers"
    ]

    print(
        f"\n[{index + 1:02d}/"
        f"{len(test_page_information):02d}] "
        f"{writer}/{page_key}"
    )

    print(
        "Retained patches:",
        patch_numbers
    )

    reconstructed_feature = (
        reconstruct_page_feature(
            writer,
            page_key,
            patch_numbers
        )
    )

    saved_feature = (
        X_test[index]
    )

    difference = np.max(
        np.abs(
            saved_feature -
            reconstructed_feature
        )
    )

    mean_difference = np.mean(
        np.abs(
            saved_feature -
            reconstructed_feature
        )
    )

    is_same = np.allclose(
        saved_feature,
        reconstructed_feature,
        rtol=1e-4,
        atol=1e-5
    )

    print(
        "Maximum difference:",
        f"{difference:.10f}"
    )

    print(
        "Mean difference:",
        f"{mean_difference:.10f}"
    )

    print(
        "MATCH:",
        is_same
    )

    if difference > max_difference:

        max_difference = (
            difference
        )

    if not is_same:

        all_close = False


# ============================================================
# FINAL RESULT
# ============================================================

print(
    "\n" +
    "=" * 60
)

print(
    "VALIDATION COMPLETE"
)

print(
    "=" * 60
)

print(
    "\nMaximum feature difference:",
    f"{max_difference:.10f}"
)

if all_close:

    print(
        "\nSUCCESS!"
    )

    print(
        "The reconstructed features "
        "match the saved page-level features."
    )

    print(
        "\nTherefore:"
    )

    print(
        "Saved patch JPEGs"
    )

    print(
        "→ same AlexNet preprocessing"
    )

    print(
        "→ same 4096-D features"
    )

    print(
        "→ same mean pooling"
    )

    print(
        "→ same page-level representation"
    )

    print(
        "\nThe Image LIME pipeline is "
        "consistent with the original ML pipeline."
    )

else:

    print(
        "\nWARNING!"
    )

    print(
        "The reconstructed features do NOT "
        "match X_test_page.npy."
    )

    print(
        "\nDo NOT proceed to the other models yet."
    )

    print(
        "The original feature extraction pipeline "
        "needs to be checked."
    )