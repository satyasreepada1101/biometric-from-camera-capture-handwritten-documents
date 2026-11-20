
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, transforms
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries


# ============================================================
# SETTINGS
# ============================================================

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))

CSV_PATH = os.path.join(
    PROJECT_ROOT, "patch_dataset", "metadata.csv"
)

OUTPUT_DIR = os.path.join(
    HERE, "lime_results", "efficientnet_b0"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_TEST_IMAGES = 5
LIME_NUM_SAMPLES = 1000
NUM_FEATURES = 10
BATCH_SIZE = 32
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# PREPROCESSING
# ============================================================

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


# ============================================================
# CHECKPOINT PATHS
# ============================================================

CHECKPOINTS = {

    "Baseline": os.path.join(
        HERE, "baseline models", "efficientnet_b0",
        "efficientnet_b0_writer_classifier.pth"
    ),

    "Partial_Fine_Tuning": os.path.join(
        HERE, "partial fine tuning models",
        "efficientnet_b0",
        "efficientnet_b0_partial_finetuned.pth"
    ),

    "Full_Fine_Tuning": os.path.join(
        HERE, "full fine tuning models",
        "efficientnet_b0", "output",
        "efficientnet_b0_full_finetuned.pth"
    ),

    "Rotation": os.path.join(
        HERE, "augmentation models",
        "efficientnet_b0", "rotation",
        "efficientnet_b0_rotation_writer_classifier.pth"
    ),

    "Translation": os.path.join(
        HERE, "augmentation models",
        "efficientnet_b0", "translation",
        "efficientnet_b0_translation_writer_classifier.pth"
    ),

    "Scaling": os.path.join(
        HERE, "augmentation models",
        "efficientnet_b0", "scaling",
        "efficientnet_b0_scaling_writer_classifier.pth"
    ),

    "Brightness_Contrast": os.path.join(
        HERE, "augmentation models",
        "efficientnet_b0", "brightness_contrast",
        "efficientnet_b0_brightness_contrast_writer_classifier.pth"
    ),

    "Combined_Augmentation": os.path.join(
        HERE, "augmentation models",
        "combined_augmentation", "efficientnet_b0",
        "efficientnet_b0_combined_augmented.pth"
    )
}


# ============================================================
# LOAD TEST DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

df = df[df["kept"] == 1].copy()

df["filepath"] = df["patch_path"].apply(
    lambda p: os.path.normpath(
        os.path.join(PROJECT_ROOT, p)
    )
)

df = df[df["filepath"].apply(os.path.isfile)].copy()

test_df = df[df["split"] == "test"].copy()

if test_df.empty:
    raise ValueError("No test images found.")

# Use the same five test images for all experiments.
test_df = (
    test_df.sort_values(["writer", "page", "filepath"])
    .sample(
        n=min(NUM_TEST_IMAGES, len(test_df)),
        random_state=SEED
    )
    .reset_index(drop=True)
)

print("\nSelected test images:", len(test_df))


# ============================================================
# LOAD EFFICIENTNET-B0
# ============================================================

def load_model(checkpoint_path):

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    writer_to_idx = checkpoint["writer_to_idx"]
    num_classes = checkpoint["num_classes"]

    model = models.efficientnet_b0(weights=None)

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        num_classes
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    idx_to_writer = {
        int(index): writer
        for index, writer in checkpoint["idx_to_writer"].items()
    }

    return model, writer_to_idx, idx_to_writer


# ============================================================
# LIME PREDICTION FUNCTION
# ============================================================

def create_predict_function(model):

    def predict_fn(images):

        batch_predictions = []

        for start in range(0, len(images), BATCH_SIZE):

            batch_images = images[start:start + BATCH_SIZE]

            tensors = []

            for image in batch_images:

                image = np.asarray(
                    image, dtype=np.float32
                )

                image = np.clip(image, 0, 1)

                tensor = torch.from_numpy(
                    image.transpose(2, 0, 1)
                ).float()

                tensor = normalize(tensor)

                tensors.append(tensor)

            batch = torch.stack(tensors).to(device)

            with torch.no_grad():

                outputs = model(batch)

                probabilities = torch.softmax(
                    outputs, dim=1
                )

            batch_predictions.append(
                probabilities.cpu().numpy()
            )

        return np.concatenate(
            batch_predictions, axis=0
        )

    return predict_fn


# ============================================================
# GENERATE LIME EXPLANATIONS
# ============================================================

explainer = LimeImageExplainer(random_state=SEED)

all_results = []

for experiment, checkpoint_path in CHECKPOINTS.items():

    print("\n" + "=" * 60)
    print("EXPERIMENT:", experiment)
    print("=" * 60)

    model, writer_to_idx, idx_to_writer = load_model(
        checkpoint_path
    )

    predict_fn = create_predict_function(model)

    experiment_dir = os.path.join(
        OUTPUT_DIR, experiment
    )

    os.makedirs(experiment_dir, exist_ok=True)

    for image_number, (_, row) in enumerate(
        test_df.iterrows(), start=1
    ):

        image_path = row["filepath"]
        actual_writer = row["writer"]

        # Load original, unaugmented test image.
        with Image.open(image_path) as img:

            original_image = img.convert("RGB").resize(
                (224, 224)
            )

            image_array = (
                np.asarray(
                    original_image, dtype=np.float32
                ) / 255.0
            )

        # Model prediction.
        probabilities = predict_fn(
            np.expand_dims(image_array, axis=0)
        )[0]

        predicted_index = int(np.argmax(probabilities))

        predicted_writer = idx_to_writer[predicted_index]

        confidence = float(
            probabilities[predicted_index]
        )

        # Actual writer class index.
        if actual_writer in writer_to_idx:
            actual_index = writer_to_idx[actual_writer]

        elif str(actual_writer) in writer_to_idx:
            actual_index = writer_to_idx[str(actual_writer)]

        else:
            raise KeyError(
                f"Writer {actual_writer} not found "
                "in checkpoint mapping."
            )

        # Generate LIME explanation.
        explanation = explainer.explain_instance(
            image_array,
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=LIME_NUM_SAMPLES
        )

        lime_label = explanation.top_labels[0]

        explained_image, mask = (
            explanation.get_image_and_mask(
                lime_label,
                positive_only=False,
                num_features=NUM_FEATURES,
                hide_rest=False
            )
        )

        image_name = f"test_image_{image_number}"

        original_path = os.path.join(
            experiment_dir,
            image_name + "_original.png"
        )

        lime_path = os.path.join(
            experiment_dir,
            image_name + "_lime.png"
        )

        mask_path = os.path.join(
            experiment_dir,
            image_name + "_mask.png"
        )

        comparison_path = os.path.join(
            experiment_dir,
            image_name + "_comparison.png"
        )

        # Save original image.
        plt.imsave(original_path, image_array)

        # Save LIME explanation.
        lime_overlay = mark_boundaries(
            explained_image,
            mask
        )

        plt.imsave(lime_path, lime_overlay)

        # Save mask.
        plt.imsave(
            mask_path,
            mask,
            cmap="gray"
        )

        # Save side-by-side comparison.
        fig, axes = plt.subplots(
            1, 2, figsize=(10, 5)
        )

        axes[0].imshow(image_array)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(lime_overlay)
        axes[1].set_title("LIME Explanation")
        axes[1].axis("off")

        fig.suptitle(
            f"{experiment}\n"
            f"Actual: {actual_writer} | "
            f"Predicted: {predicted_writer}\n"
            f"Confidence: {confidence:.4f}"
        )

        plt.tight_layout()

        plt.savefig(
            comparison_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        # Store results.
        result = {
            "Experiment": experiment,
            "Image": image_name,
            "Image_Path": image_path,
            "Actual_Writer": actual_writer,
            "Predicted_Writer": predicted_writer,
            "Actual_Class_Index": actual_index,
            "Predicted_Class_Index": predicted_index,
            "Prediction_Confidence": confidence,
            "Correct": actual_index == predicted_index,
            "Original_Image": original_path,
            "LIME_Image": lime_path,
            "LIME_Mask": mask_path,
            "Comparison_Image": comparison_path
        }

        all_results.append(result)

        print(
            f"Image {image_number}: "
            f"Actual = {actual_writer} | "
            f"Predicted = {predicted_writer} | "
            f"Confidence = {confidence:.4f}"
        )

    # Release memory.
    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)

results_csv = os.path.join(
    OUTPUT_DIR,
    "efficientnet_b0_lime_results.csv"
)

results_df.to_csv(
    results_csv,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = (
    results_df.groupby("Experiment")
    .agg(
        Total_Images=("Image", "count"),
        Correct_Predictions=("Correct", "sum"),
        Mean_Confidence=("Prediction_Confidence", "mean")
    )
    .reset_index()
)

summary["Accuracy_Percent"] = (
    summary["Correct_Predictions"]
    / summary["Total_Images"] * 100
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "efficientnet_b0_lime_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)

print("\n" + "=" * 60)
print("EFFICIENTNET-B0 LIME COMPLETED")
print("=" * 60)

print("\nSUMMARY:")
print(summary.to_string(index=False))

print("\nResults saved:", results_csv)
print("Summary saved:", summary_path)
print("Output folder:", OUTPUT_DIR)
