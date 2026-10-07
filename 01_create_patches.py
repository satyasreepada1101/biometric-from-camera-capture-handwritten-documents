# ============================================================
# 01_create_patches.py
#
# Purpose:
#   1. Read ML_dataset
#   2. Split pages into train/test PER WRITER
#   3. Divide each page into 4 x 4 patches
#   4. Remove mostly blank patches
#   5. Save patches + metadata
#
# NO MACHINE LEARNING HAPPENS IN THIS FILE.
# ============================================================

from pathlib import Path
from PIL import Image
import csv
import random
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("ML_dataset")

OUTPUT_DIR = Path("patch_dataset")

RANDOM_SEED = 42

# Each writer has 5 pages.
# We use 4 for training and 1 for testing.
TRAIN_PAGES_PER_WRITER = 4
TEST_PAGES_PER_WRITER = 1

# Minimum amount of dark content required.
#
# This is deliberately conservative.
# We can tune this later after inspecting the results.
MIN_INK_RATIO = 0.015


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(RANDOM_SEED)


# ============================================================
# FUNCTIONS
# ============================================================

def is_image_file(path):
    """
    Check whether a file is an image.
    """

    return path.suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp"
    }


def calculate_ink_ratio(image):
    """
    Estimate how much dark content exists in a patch.

    This is NOT meant to perfectly identify handwriting.
    It is only used to remove patches that are almost completely
    blank.

    Notebook lines / shadows may also contribute slightly.
    """

    gray = image.convert("L")

    array = np.asarray(gray)

    # Pixels below 180 are treated as dark.
    dark_pixels = array < 180

    ratio = dark_pixels.mean()

    return float(ratio)


def create_16_patches(image):
    """
    Create the exact 4 x 4 layout described in the project.

    For a 2100 x 2400 image:

        width  = 2100
        height = 2400

    We get:

        columns:
            600
            600
            600
            300

        rows:
            600
            600
            600
            600

    Therefore:

        12 patches = 600 x 600
        4 patches  = 300 x 600

    Returns:
        list of (patch_number, patch_image, ink_ratio)
    """

    width, height = image.size

    # --------------------------------------------------------
    # Height:
    # Divide into four equal horizontal bands.
    # --------------------------------------------------------

    row_height = height // 4

    # --------------------------------------------------------
    # Width:
    #
    # We want each of the first three columns to have the
    # same width as one row height.
    #
    # Example:
    #
    # height = 2400
    # row_height = 600
    #
    # width = 2100
    #
    # columns = 600, 600, 600, 300
    # --------------------------------------------------------

    standard_patch_width = row_height

    patches = []

    patch_number = 1

    for row in range(4):

        top = row * row_height

        if row < 3:
            bottom = (row + 1) * row_height
        else:
            bottom = height

        for col in range(4):

            left = col * standard_patch_width

            if col < 3:
                right = (col + 1) * standard_patch_width
            else:
                right = width

            # ------------------------------------------------
            # Safety check
            # ------------------------------------------------

            if left >= width:
                continue

            if right > width:
                right = width

            if top >= height:
                continue

            if bottom > height:
                bottom = height

            # ------------------------------------------------
            # Crop
            # ------------------------------------------------

            patch = image.crop(
                (left, top, right, bottom)
            )

            ink_ratio = calculate_ink_ratio(patch)

            patches.append(
                (
                    patch_number,
                    patch,
                    ink_ratio
                )
            )

            patch_number += 1

    return patches


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("PATCH CREATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset directory not found: "
            f"{DATASET_DIR}"
        )

    # --------------------------------------------------------
    # Find writer folders
    # --------------------------------------------------------

    writer_dirs = sorted(
        [
            p for p in DATASET_DIR.iterdir()
            if p.is_dir()
        ],
        key=lambda x: x.name
    )

    print(
        f"\nFound {len(writer_dirs)} writer folders."
    )

    if len(writer_dirs) != 40:

        print(
            f"WARNING: Expected 40 writers, "
            f"but found {len(writer_dirs)}."
        )

    # --------------------------------------------------------
    # Create output directories
    # --------------------------------------------------------

    train_output = OUTPUT_DIR / "train"

    test_output = OUTPUT_DIR / "test"

    train_output.mkdir(
        parents=True,
        exist_ok=True
    )

    test_output.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = []

    total_pages = 0

    total_kept_patches = 0

    total_removed_patches = 0

    # --------------------------------------------------------
    # Process each writer
    # --------------------------------------------------------

    for writer_id, writer_dir in enumerate(writer_dirs):

        writer_name = writer_dir.name

        print(
            f"\nWriter {writer_id:02d}: {writer_name}"
        )

        # ----------------------------------------------------
        # Find images
        # ----------------------------------------------------

        image_files = sorted(
            [
                p for p in writer_dir.iterdir()
                if p.is_file() and is_image_file(p)
            ]
        )

        print(
            f"  Found {len(image_files)} images."
        )

        if len(image_files) != 5:

            print(
                f"  WARNING: Expected 5 pages "
                f"but found {len(image_files)}."
            )

        # ----------------------------------------------------
        # Shuffle pages
        # ----------------------------------------------------

        shuffled_pages = image_files.copy()

        random.shuffle(shuffled_pages)

        # ----------------------------------------------------
        # Split pages
        # ----------------------------------------------------

        train_pages = shuffled_pages[
            :TRAIN_PAGES_PER_WRITER
        ]

        test_pages = shuffled_pages[
            TRAIN_PAGES_PER_WRITER:
            TRAIN_PAGES_PER_WRITER + TEST_PAGES_PER_WRITER
        ]

        # ----------------------------------------------------
        # Process train and test
        # ----------------------------------------------------

        split_information = [
            ("train", train_pages, train_output),
            ("test", test_pages, test_output)
        ]

        for split_name, pages, split_output in split_information:

            writer_output = (
                split_output / writer_name
            )

            writer_output.mkdir(
                parents=True,
                exist_ok=True
            )

            for page_path in pages:

                total_pages += 1

                print(
                    f"    {split_name}: "
                    f"{page_path.name}"
                )

                # --------------------------------------------
                # Open image
                # --------------------------------------------

                image = Image.open(
                    page_path
                ).convert("RGB")

                width, height = image.size

                print(
                    f"      size: "
                    f"{width} x {height}"
                )

                # --------------------------------------------
                # Create 16 patches
                # --------------------------------------------

                patches = create_16_patches(
                    image
                )

                # --------------------------------------------
                # Save patches
                # --------------------------------------------

                for (
                    patch_number,
                    patch,
                    ink_ratio
                ) in patches:

                    if ink_ratio < MIN_INK_RATIO:

                        total_removed_patches += 1

                        metadata.append(
                            {
                                "writer_id": writer_id,
                                "writer": writer_name,
                                "split": split_name,
                                "page": page_path.name,
                                "patch": patch_number,
                                "ink_ratio": ink_ratio,
                                "kept": 0,
                                "patch_path": ""
                            }
                        )

                        continue

                    # ----------------------------------------
                    # Patch filename
                    # ----------------------------------------

                    page_stem = page_path.stem

                    patch_filename = (
                        f"{page_stem}"
                        f"_patch_{patch_number:02d}.jpg"
                    )

                    patch_path = (
                        writer_output /
                        patch_filename
                    )

                    # ----------------------------------------
                    # Save
                    # ----------------------------------------

                    patch.save(
                        patch_path,
                        quality=95
                    )

                    total_kept_patches += 1

                    # ----------------------------------------
                    # Metadata
                    # ----------------------------------------

                    metadata.append(
                        {
                            "writer_id": writer_id,
                            "writer": writer_name,
                            "split": split_name,
                            "page": page_path.name,
                            "patch": patch_number,
                            "ink_ratio": ink_ratio,
                            "kept": 1,
                            "patch_path": str(patch_path)
                        }
                    )

    # ========================================================
    # SAVE METADATA CSV
    # ========================================================

    metadata_path = (
        OUTPUT_DIR / "metadata.csv"
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
            "split",
            "page",
            "patch",
            "ink_ratio",
            "kept",
            "patch_path"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(metadata)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")
    print("=" * 60)
    print("PATCH CREATION COMPLETE")
    print("=" * 60)

    print(
        f"Total pages processed: "
        f"{total_pages}"
    )

    print(
        f"Patches kept: "
        f"{total_kept_patches}"
    )

    print(
        f"Patches removed: "
        f"{total_removed_patches}"
    )

    print(
        f"Metadata saved to: "
        f"{metadata_path}"
    )

    print(
        f"\nPatch dataset saved to: "
        f"{OUTPUT_DIR}"
    )

    print("\nTrain structure:")
    print(
        "patch_dataset/train/<writer>/..."
    )

    print("\nTest structure:")
    print(
        "patch_dataset/test/<writer>/..."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()