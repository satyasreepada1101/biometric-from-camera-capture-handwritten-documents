# ============================================================
# image_lime_alexnet.py
#
# Purpose:
#   Contrastive Image-level LIME explanation for the
#   AlexNet + Random Forest pipeline.
#
# Green = favors actual writer
# Red   = favors predicted/wrong writer
#
# Contrastive score:
#   actual-class LIME weight - predicted-class LIME weight
#
# Positive score -> favors actual writer
# Negative score -> favors predicted writer
# ============================================================

from pathlib import Path
import re
import io

import numpy as np
import pandas as pd

from PIL import Image

import torch
from torchvision import models, transforms

from sklearn.ensemble import RandomForestClassifier

from lime.lime_image import LimeImageExplainer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

ALEXNET_DIR = SCRIPT_DIR.parent

PROJECT_ROOT = ALEXNET_DIR.parents[1]

FEATURE_DIR = ALEXNET_DIR / "features"

ML_DATASET_DIR = PROJECT_ROOT / "ML_dataset"

RESULTS_DIR = (
    SCRIPT_DIR /
    "results" /
    "image_lime"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224

BATCH_SIZE = 32

RANDOM_STATE = 42

# Number of LIME perturbation samples
LIME_NUM_SAMPLES = 100


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 60)
print("ALEXNET CONTRASTIVE IMAGE LIME")
print("=" * 60)

print(
    f"\nUsing device: {DEVICE}"
)


# ============================================================
# LOAD PAGE-LEVEL FEATURES
# ============================================================

print("\nLoading page-level AlexNet features...")


X_train = np.load(
    FEATURE_DIR /
    "X_train_page.npy"
)

y_train = np.load(
    FEATURE_DIR /
    "y_train_page.npy"
)

X_test = np.load(
    FEATURE_DIR /
    "X_test_page.npy"
)

y_test = np.load(
    FEATURE_DIR /
    "y_test_page.npy"
)


print(
    "\nTraining feature shape:",
    X_train.shape
)

print(
    "Training label shape:",
    y_train.shape
)

print(
    "Testing feature shape:",
    X_test.shape
)

print(
    "Testing label shape:",
    y_test.shape
)


# ============================================================
# LOAD TEST METADATA
# ============================================================

test_metadata = pd.read_csv(
    FEATURE_DIR /
    "test_patch_metadata.csv"
)

train_metadata = pd.read_csv(
    FEATURE_DIR /
    "train_patch_metadata.csv"
)


# ============================================================
# PREPARE METADATA
# ============================================================

def prepare_metadata(metadata):

    metadata = metadata.copy()

    # --------------------------------------------------------
    # Extract patch number
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Extract original page name
    # --------------------------------------------------------

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

    return metadata


test_metadata = prepare_metadata(
    test_metadata
)

train_metadata = prepare_metadata(
    train_metadata
)


# ============================================================
# CREATE PAGE INFORMATION
# ============================================================

grouped_test = test_metadata.groupby(
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
            "writer_id": writer_id,
            "writer": writer,
            "page_key": page_key,
            "patch_numbers": sorted(
                group[
                    "patch_number"
                ].tolist()
            )
        }
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
# ALEXNET PREPROCESSING
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

def extract_alexnet_features(
    images
):

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
            (0, 4096),
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
# EXACT PATCH SEGMENTATION
# ============================================================

def create_patch_segments(
    image
):

    """
    Create the exact 4 x 4 segmentation used by
    01_create_patches.py.
    """

    height, width = image.shape[:2]

    row_height = (
        height // 4
    )

    standard_patch_width = (
        row_height
    )

    segments = np.zeros(
        (
            height,
            width
        ),
        dtype=np.int32
    )

    segment_number = 0

    for row in range(4):

        top = (
            row *
            row_height
        )

        if row < 3:

            bottom = (
                (row + 1) *
                row_height
            )

        else:

            bottom = height

        for col in range(4):

            left = (
                col *
                standard_patch_width
            )

            if col < 3:

                right = (
                    (col + 1) *
                    standard_patch_width
                )

            else:

                right = width

            if left >= width:
                continue

            if right > width:
                right = width

            if top >= height:
                continue

            if bottom > height:
                bottom = height

            segments[
                top:bottom,
                left:right
            ] = segment_number

            segment_number += 1

    return segments


# ============================================================
# CROP THE EXACT 16 PATCHES
# ============================================================

def crop_page_into_patches(
    image
):

    """
    Reproduce the exact patches created by
    01_create_patches.py.
    """

    width, height = image.size

    row_height = height // 4

    standard_patch_width = row_height

    patches = []

    patch_number = 1

    for row in range(4):

        top = row * row_height

        if row < 3:

            bottom = (
                (row + 1) *
                row_height
            )

        else:

            bottom = height

        for col in range(4):

            left = (
                col *
                standard_patch_width
            )

            if col < 3:

                right = (
                    (col + 1) *
                    standard_patch_width
                )

            else:

                right = width

            if left >= width:
                continue

            if right > width:
                right = width

            if top >= height:
                continue

            if bottom > height:
                bottom = height

            patch = image.crop(
                (
                    left,
                    top,
                    right,
                    bottom
                )
            )

            patches.append(
                (
                    patch_number,
                    patch
                )
            )

            patch_number += 1

    return patches


# ============================================================
# REPRODUCE SAVED PATCH
# ============================================================

def reproduce_saved_patch(
    patch
):

    """
    Reproduce the exact image that AlexNet originally saw.

    patch
        -> JPEG quality=95
        -> Image.open()
        -> RGB
    """

    buffer = io.BytesIO()

    patch.save(
        buffer,
        format="JPEG",
        quality=95
    )

    buffer.seek(0)

    reproduced_patch = Image.open(
        buffer
    ).convert(
        "RGB"
    )

    return reproduced_patch


# ============================================================
# PAGE PREDICTION FUNCTION FOR LIME
# ============================================================

def make_prediction_function(
    classifier,
    kept_patch_numbers
):

    """
    Exact original project pipeline:

        LIME page
            ↓
        exact 4x4 patches
            ↓
        retained patches
            ↓
        JPEG quality=95
            ↓
        AlexNet
            ↓
        4096-D patch features
            ↓
        mean pooling
            ↓
        Random Forest
    """

    def predict(images):

        page_features = []

        for image_array in images:

            image = Image.fromarray(
                image_array.astype(
                    np.uint8
                )
            ).convert(
                "RGB"
            )

            patches = crop_page_into_patches(
                image
            )

            selected_patches = []

            for (
                patch_number,
                patch
            ) in patches:

                if (
                    patch_number
                    in kept_patch_numbers
                ):

                    patch = (
                        reproduce_saved_patch(
                            patch
                        )
                    )

                    selected_patches.append(
                        patch
                    )

            if len(
                selected_patches
            ) == 0:

                raise RuntimeError(
                    "No retained patches found "
                    "for this page."
                )

            patch_features = (
                extract_alexnet_features(
                    selected_patches
                )
            )

            page_feature = (
                patch_features.mean(
                    axis=0
                )
            )

            page_features.append(
                page_feature
            )

        page_features = np.asarray(
            page_features,
            dtype=np.float32
        )

        probabilities = (
            classifier.predict_proba(
                page_features
            )
        )

        return probabilities

    return predict


# ============================================================
# TRAIN RANDOM FOREST
# ============================================================

print(
    "\nTraining Random Forest..."
)

random_forest = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

random_forest.fit(
    X_train,
    y_train
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

predictions = (
    random_forest.predict(
        X_test
    )
)

accuracy = np.mean(
    predictions == y_test
)

print(
    f"\nRandom Forest accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# FIND MISCLASSIFIED PAGES
# ============================================================

error_indices = np.where(
    predictions != y_test
)[0]

print(
    "\nNumber of misclassified "
    f"pages: {len(error_indices)}"
)

if len(error_indices) == 0:

    raise RuntimeError(
        "No misclassified test pages found."
    )


# ============================================================
# CREATE LIME EXPLAINER
# ============================================================

explainer = LimeImageExplainer(
    random_state=RANDOM_STATE
)


# ============================================================
# RUN CONTRASTIVE IMAGE LIME
# ============================================================

print(
    "\n" +
    "=" * 60
)

print(
    "RUNNING CONTRASTIVE IMAGE LIME"
)

print(
    "=" * 60
)

print(
    f"\nTotal misclassified pages: "
    f"{len(error_indices)}"
)


# ============================================================
# PROCESS EACH MISCLASSIFIED PAGE
# ============================================================

for error_number, sample_index in enumerate(
    error_indices,
    start=1
):

    print(
        "\n" +
        "=" * 60
    )

    print(
        f"PROCESSING ERROR "
        f"{error_number}/{len(error_indices)}"
    )

    print(
        "=" * 60
    )

    # ========================================================
    # GET PAGE INFORMATION
    # ========================================================

    sample_info = (
        test_page_information[
            sample_index
        ]
    )

    actual_writer_id = int(
        y_test[
            sample_index
        ]
    )

    predicted_writer_id = int(
        predictions[
            sample_index
        ]
    )

    actual_writer = (
        sample_info[
            "writer"
        ]
    )

    page_key = (
        sample_info[
            "page_key"
        ]
    )

    kept_patch_numbers = (
        sample_info[
            "patch_numbers"
        ]
    )

    print(
        f"\nTest sample index : "
        f"{sample_index}"
    )

    print(
        f"Actual writer     : "
        f"{actual_writer}"
    )

    print(
        f"Actual writer ID  : "
        f"{actual_writer_id}"
    )

    print(
        f"Predicted writer ID: "
        f"{predicted_writer_id}"
    )

    print(
        f"Page              : "
        f"{page_key}.jpg"
    )

    print(
        f"Used patch numbers: "
        f"{kept_patch_numbers}"
    )


    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    error_dir = (
        RESULTS_DIR /
        f"error_{error_number:02d}_"
        f"{actual_writer}_"
        f"{page_key}"
    )

    error_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # LOAD ORIGINAL PAGE
    # ========================================================

    page_filename = page_key

    if not page_filename.lower().endswith(
        (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp"
        )
    ):

        page_filename += ".jpg"

    page_path = (
        ML_DATASET_DIR /
        actual_writer /
        page_filename
    )

    if not page_path.exists():

        print(
            "\nWARNING: Original page not found:"
        )

        print(
            page_path
        )

        print(
            "Skipping this error."
        )

        continue


    original_image = (
        Image.open(
            page_path
        ).convert(
            "RGB"
        )
    )

    original_array = (
        np.asarray(
            original_image
        )
    )

    print(
        "\nOriginal page:"
    )

    print(
        f"Path: {page_path}"
    )

    print(
        f"Size: {original_image.size}"
    )


    # ========================================================
    # CREATE LIME SEGMENTATION
    # ========================================================

    segments = (
        create_patch_segments(
            original_array
        )
    )

    number_of_regions = len(
        np.unique(
            segments
        )
    )

    print(
        "\nNumber of LIME regions:",
        number_of_regions
    )


    # ========================================================
    # CREATE PAGE PREDICTION FUNCTION
    # ========================================================

    prediction_function = (
        make_prediction_function(
            random_forest,
            kept_patch_numbers
        )
    )


    # ========================================================
    # VERIFY ORIGINAL PAGE PREDICTION
    # ========================================================

    original_probabilities = (
        prediction_function(
            [
                original_array
            ]
        )
    )

    actual_class_index = (
        list(
            random_forest.classes_
        ).index(
            actual_writer_id
        )
    )

    predicted_class_index = (
        list(
            random_forest.classes_
        ).index(
            predicted_writer_id
        )
    )

    actual_probability = (
        original_probabilities[
            0,
            actual_class_index
        ]
    )

    predicted_probability = (
        original_probabilities[
            0,
            predicted_class_index
        ]
    )

    print(
        "\nOriginal page prediction:"
    )

    print(
        f"Actual writer       : "
        f"{actual_writer}"
    )

    print(
        f"Predicted writer    : "
        f"{predicted_writer_id}"
    )

    print(
        f"Actual probability  : "
        f"{actual_probability:.4f}"
    )

    print(
        f"Predicted probability: "
        f"{predicted_probability:.4f}"
    )


    # ========================================================
    # RUN LIME FOR BOTH CLASSES
    # ========================================================

    print(
        "\nGenerating Contrastive Image LIME..."
    )

    print(
        "This may take some time..."
    )

    explanation = (
    explainer.explain_instance(

        image=original_array,

        classifier_fn=(
            prediction_function
        ),

        labels=[
            actual_class_index,
            predicted_class_index
        ],

        top_labels=None,

        hide_color=255,

        num_features=(
            number_of_regions
        ),

        num_samples=(
            LIME_NUM_SAMPLES
        ),

        batch_size=10,

        segmentation_fn=(
            create_patch_segments
        ),

        random_seed=(
            RANDOM_STATE
        )
    )
)


    # ========================================================
    # GET LIME WEIGHTS
    # ========================================================

    actual_local_explanation = (
        explanation.local_exp[
            actual_class_index
        ]
    )

    predicted_local_explanation = (
        explanation.local_exp[
            predicted_class_index
        ]
    )

    if actual_class_index not in explanation.local_exp:
        raise RuntimeError(
            f"Actual class {actual_class_index} was not explained. "
            f"Available labels: {list(explanation.local_exp.keys())}"
        )

    if predicted_class_index not in explanation.local_exp:
        raise RuntimeError(
            f"Predicted class {predicted_class_index} was not explained. "
            f"Available labels: {list(explanation.local_exp.keys())}"
        )


    actual_weights = dict(
        explanation.local_exp[
            actual_class_index
        ]
    )

    predicted_weights = dict(
        explanation.local_exp[
            predicted_class_index
        ]
    )


    # ========================================================
    # CALCULATE CONTRASTIVE WEIGHTS
    # ========================================================

    all_segment_ids = sorted(
        set(actual_weights) |
        set(predicted_weights)
    )

    contrastive_weights = {
        segment_id:
        actual_weights.get(
            segment_id,
            0.0
        )
        -
        predicted_weights.get(
            segment_id,
            0.0
        )

        for segment_id
        in all_segment_ids
    }


    # ========================================================
    # SAVE TEXT EXPLANATION
    # ========================================================

    text_path = (
        error_dir /
        "image_lime_error_analysis.txt"
    )

    with open(
        text_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "ALEXNET CONTRASTIVE IMAGE LIME "
            "ERROR ANALYSIS\n"
        )

        f.write(
            "=" * 60 +
            "\n\n"
        )

        f.write(
            f"Error number: "
            f"{error_number}\n"
        )

        f.write(
            f"Test sample index: "
            f"{sample_index}\n"
        )

        f.write(
            f"Original page: "
            f"{page_path}\n"
        )

        f.write(
            f"Actual writer: "
            f"{actual_writer}\n"
        )

        f.write(
            f"Actual writer ID: "
            f"{actual_writer_id}\n"
        )

        f.write(
            f"Predicted writer ID: "
            f"{predicted_writer_id}\n"
        )

        f.write(
            f"Actual probability: "
            f"{actual_probability:.6f}\n"
        )

        f.write(
            f"Predicted probability: "
            f"{predicted_probability:.6f}\n"
        )

        f.write(
            f"Number of LIME regions: "
            f"{number_of_regions}\n"
        )

        f.write(
            f"Used patch numbers: "
            f"{kept_patch_numbers}\n\n"
        )

        f.write(
            "CONTRASTIVE EXPLANATION\n"
        )

        f.write(
            "-" * 60 +
            "\n"
        )

        f.write(
            "GREEN = favors actual writer\n"
        )

        f.write(
            "RED = favors predicted writer\n"
        )

        f.write(
            "Contrastive score = "
            "actual LIME weight - "
            "predicted LIME weight\n\n"
        )

        f.write(
            "CONTRASTIVE LIME REGION CONTRIBUTIONS\n"
        )

        f.write(
            "-" * 60 +
            "\n"
        )

        sorted_weights = sorted(
            contrastive_weights.items(),
            key=lambda x: abs(
                x[1]
            ),
            reverse=True
        )

        for (
            segment_id,
            contrastive_weight
        ) in sorted_weights:

            patch_number = (
                segment_id + 1
            )

            actual_weight = (
                actual_weights.get(
                    segment_id,
                    0.0
                )
            )

            predicted_weight = (
                predicted_weights.get(
                    segment_id,
                    0.0
                )
            )

            if contrastive_weight > 0:

                effect = (
                    "GREEN - "
                    "favors actual writer"
                )

            elif contrastive_weight < 0:

                effect = (
                    "RED - "
                    "favors predicted writer"
                )

            else:

                effect = (
                    "NEUTRAL"
                )

            f.write(
                f"Patch "
                f"{patch_number:02d}: "
                f"contrastive="
                f"{contrastive_weight:+.6f} | "
                f"actual="
                f"{actual_weight:+.6f} | "
                f"predicted="
                f"{predicted_weight:+.6f} "
                f"({effect})\n"
            )


    # ========================================================
    # CREATE VISUALIZATION ARRAYS
    # ========================================================

    height, width = (
        original_array.shape[:2]
    )

    positive_overlay = np.zeros(
        (
            height,
            width,
            4
        ),
        dtype=np.float32
    )

    negative_overlay = np.zeros(
        (
            height,
            width,
            4
        ),
        dtype=np.float32
    )

    combined_overlay = np.zeros(
        (
            height,
            width,
            4
        ),
        dtype=np.float32
    )


    # ========================================================
    # FIND MAXIMUM CONTRASTIVE WEIGHT
    # ========================================================

    max_weight = max(
        [
            abs(weight)
            for weight
            in contrastive_weights.values()
        ]
        or [1.0]
    )


    # ========================================================
    # APPLY GREEN / RED COLORS
    # ========================================================

    for (
        segment_id,
        weight
    ) in contrastive_weights.items():

        mask = (
            segments ==
            segment_id
        )

        alpha = (
            0.20 +
            0.55 *
            (
                abs(weight) /
                max_weight
            )
        )

        # ----------------------------------------------------
        # GREEN = favors actual writer
        # ----------------------------------------------------

        if weight > 0:

            positive_overlay[
                mask
            ] = [
                0.0,
                1.0,
                0.0,
                alpha
            ]

            combined_overlay[
                mask
            ] = [
                0.0,
                1.0,
                0.0,
                alpha
            ]

        # ----------------------------------------------------
        # RED = favors predicted writer
        # ----------------------------------------------------

        elif weight < 0:

            negative_overlay[
                mask
            ] = [
                1.0,
                0.0,
                0.0,
                alpha
            ]

            combined_overlay[
                mask
            ] = [
                1.0,
                0.0,
                0.0,
                alpha
            ]


    # ========================================================
    # SAVE ORIGINAL PAGE
    # ========================================================

    original_output = (
        error_dir /
        "original_page.png"
    )

    plt.figure(
        figsize=(10, 12)
    )

    plt.imshow(
        original_array
    )

    plt.axis(
        "off"
    )

    plt.title(
        f"Original Page\n"
        f"Actual: {actual_writer} | "
        f"Predicted: {predicted_writer_id}"
    )

    plt.tight_layout()

    plt.savefig(
        original_output,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # SAVE COMBINED CONTRASTIVE EXPLANATION
    # ========================================================

    lime_output = (
        error_dir /
        "lime_contrastive_red_green.png"
    )

    plt.figure(
        figsize=(10, 12)
    )

    plt.imshow(
        original_array
    )

    plt.imshow(
        combined_overlay
    )

    plt.axis(
        "off"
    )

    plt.title(
        "Contrastive Image LIME\n"
        "Green = favors actual writer | "
        "Red = favors predicted writer"
    )

    plt.tight_layout()

    plt.savefig(
        lime_output,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # SAVE GREEN REGIONS
    # ========================================================

    positive_output = (
        error_dir /
        "lime_actual_writer_green.png"
    )

    plt.figure(
        figsize=(10, 12)
    )

    plt.imshow(
        original_array
    )

    plt.imshow(
        positive_overlay
    )

    plt.axis(
        "off"
    )

    plt.title(
        "LIME Regions Favoring Actual Writer"
    )

    plt.tight_layout()

    plt.savefig(
        positive_output,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # SAVE RED REGIONS
    # ========================================================

    negative_output = (
        error_dir /
        "lime_predicted_writer_red.png"
    )

    plt.figure(
        figsize=(10, 12)
    )

    plt.imshow(
        original_array
    )

    plt.imshow(
        negative_overlay
    )

    plt.axis(
        "off"
    )

    plt.title(
        "LIME Regions Favoring Predicted Writer"
    )

    plt.tight_layout()

    plt.savefig(
        negative_output,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # PRINT COMPLETION
    # ========================================================

    print(
        "\nContrastive Image LIME complete."
    )

    print(
        f"Results saved to:"
    )

    print(
        error_dir
    )


# ============================================================
# ALL ERRORS COMPLETE
# ============================================================

print(
    "\n" +
    "=" * 60
)

print(
    "ALL CONTRASTIVE IMAGE LIME ERROR ANALYSIS COMPLETE"
)

print(
    "=" * 60
)

print(
    f"\nProcessed "
    f"{len(error_indices)} "
    f"misclassified pages."
)

print(
    "\nResults directory:"
)

print(
    RESULTS_DIR
)

print(
    "\nEach error has its own folder containing:"
)

print(
    "1. original_page.png"
)

print(
    "2. lime_contrastive_red_green.png"
)

print(
    "3. lime_actual_writer_green.png"
)

print(
    "4. lime_predicted_writer_red.png"
)

print(
    "5. image_lime_error_analysis.txt"
)