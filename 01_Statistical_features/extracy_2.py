import cv2
import numpy as np
import pandas as pd
import os
import glob


def create_16_patches(img):
    """
    Split an image using the same layout as 01_create_patches(1).py.

    The source image is divided into 4 rows. Each row is height // 4
    pixels tall (except the last row, which includes any remainder).
    The first three columns have width equal to the row height; the
    fourth column contains the remaining image width.

    For a 2100 x 2400 image, this creates:
      - 12 patches of 600 x 600
      - 4 patches of 300 x 600

    OpenCV image dimensions are (height, width, channels).
    Returns a list of (patch_number, patch_image).
    """
    h, w = img.shape[:2]
    row_height = h // 4
    standard_patch_width = row_height

    patches = []
    patch_number = 1

    for row in range(4):
        top = row * row_height
        bottom = (row + 1) * row_height if row < 3 else h

        for col in range(4):
            left = col * standard_patch_width
            right = (col + 1) * standard_patch_width if col < 3 else w

            # Match the safety checks in 01_create_patches(1).py.
            if left >= w or top >= h:
                continue
            right = min(right, w)
            bottom = min(bottom, h)

            patch = img[top:bottom, left:right]
            if patch.size == 0:
                continue

            patches.append((patch_number, patch))
            patch_number += 1

    return patches


def calculate_patch_features(patch):
    """Calculate RGB mean and variance for one patch."""
    # OpenCV loads images in BGR order.
    b_channel = patch[:, :, 0]
    g_channel = patch[:, :, 1]
    r_channel = patch[:, :, 2]

    features = {}
    for channel_name, channel_data in [
        ("R", r_channel),
        ("G", g_channel),
        ("B", b_channel),
    ]:
        features[f"{channel_name}_mean"] = float(np.mean(channel_data))
        features[f"{channel_name}_var"] = float(np.var(channel_data))

    return features


def extract_features(image_path):
    """
    Read an image, split it into the same 4 x 4 patch layout as
    01_create_patches(1).py, and return one feature record per patch.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    patch_records = []
    for patch_number, patch in create_16_patches(img):
        features = calculate_patch_features(patch)
        patch_records.append({
            "patch_number": patch_number,
            "patch_height": int(patch.shape[0]),
            "patch_width": int(patch.shape[1]),
            **features,
        })

    return patch_records


def process_dataset(root_dir, output_csv="features_2.csv"):
    """
    Process images stored as root_dir/person_id/image files.

    Each source image produces up to 16 rows (one row per patch).
    The CSV records patch dimensions so the generated sizes can be checked.
    """
    all_rows = []
    supported_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")

    person_folders = sorted(glob.glob(os.path.join(root_dir, "*")))

    for person_folder in person_folders:
        if not os.path.isdir(person_folder):
            continue

        person_id = os.path.basename(person_folder)
        image_paths = []
        for extension in supported_extensions:
            image_paths.extend(glob.glob(os.path.join(person_folder, extension)))
        image_paths = sorted(image_paths)

        for img_path in image_paths:
            try:
                patch_records = extract_features(img_path)
            except ValueError as error:
                print(f"Skipping: {error}")
                continue

            for record in patch_records:
                row = {
                    "person_id": person_id,
                    "image_name": os.path.basename(img_path),
                    **record,
                }
                all_rows.append(row)

            print(
                f"Processed: {person_id}/{os.path.basename(img_path)} "
                f"({len(patch_records)} patches)"
            )

    columns = [
        "person_id", "image_name", "patch_number",
        "patch_height", "patch_width",
        "R_mean", "R_var", "G_mean", "G_var", "B_mean", "B_var",
    ]
    df = pd.DataFrame(all_rows, columns=columns)
    df.to_csv(output_csv, index=False)

    print(f"\nSaved {len(df)} patch rows to {output_csv}")
    if not df.empty:
        print("Patch dimensions (height x width):")
        print(
            df[["patch_height", "patch_width"]]
            .drop_duplicates()
            .sort_values(["patch_height", "patch_width"])
            .to_string(index=False)
        )
    return df


if __name__ == "__main__":
    import sys

    root_dir = sys.argv[1] if len(sys.argv) > 1 else "ML_dataset"
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "features_2.csv"
    process_dataset(root_dir, output_csv)
