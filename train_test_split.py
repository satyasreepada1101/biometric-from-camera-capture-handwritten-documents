"""
info:
image 1-4 is for train
image 5 is for test
all 5 images are split in 20 parts

everything is saved in "ML_dataset_patches" folder with "train" and "test" subfolders

the split is done by page, not by patch
this prevents patches from the same handwritten page appearing in both the training and testing datasets

to run: python train_test_split.py
"""

import os
import random
import re

import cv2
import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DATASET_DIR = os.path.join(PROJECT_ROOT, "ML_dataset")
PATCH_DATASET_DIR = os.path.join(PROJECT_ROOT, "ML_dataset_patches")
TRAIN_DIR = os.path.join(PATCH_DATASET_DIR, "train")
TEST_DIR = os.path.join(PATCH_DATASET_DIR, "test")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
TEST_PAGE_INDEX = 4       # page_5, because the list starts at index 0
PATCH_SIZE = 227
PATCHES_PER_PAGE = 20
MIN_INK_RATIO = 0.02
GRAYSCALE = False


def natural_sort_key(filename):
	# Sort page_2 before page_10 instead of alphabetically
	numbers = re.findall(r"\d+", filename)
	return int(numbers[0]) if numbers else filename


def list_writer_pages(writer_dir):
	# Return the writer's image pages in page-number order 
	files = [
		filename
		for filename in os.listdir(writer_dir)
		if filename.lower().endswith(IMAGE_EXTENSIONS)
	]
	files.sort(key=natural_sort_key)
	return [os.path.join(writer_dir, filename) for filename in files]


def has_enough_ink(patch_gray, min_ratio=MIN_INK_RATIO):
	# Check whether a candidate patch contains enough handwritten ink
	_, threshold = cv2.threshold(
		patch_gray,
		0,
		255,
		cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
	)
	ink_ratio = np.count_nonzero(threshold) / threshold.size
	return ink_ratio >= min_ratio


def extract_patches_from_page(
	image_path,
	num_patches=PATCHES_PER_PAGE,
	patch_size=PATCH_SIZE,
):
	# Extract exactly num_patches random crops from one page.
	image = cv2.imread(image_path)
	if image is None:
		print(f"  [WARN] could not read {image_path}, skipping")
		return []

	height, width = image.shape[:2]
	if height < patch_size or width < patch_size:
		image = cv2.resize(image, (max(width, patch_size), max(height, patch_size)))
		height, width = image.shape[:2]

	gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	patches = []
	attempts = 0
	max_attempts = num_patches * 10

	# Prefer crops that contain handwriting.
	while len(patches) < num_patches and attempts < max_attempts:
		attempts += 1
		x = random.randint(0, width - patch_size)
		y = random.randint(0, height - patch_size)
		candidate_gray = gray_image[y:y + patch_size, x:x + patch_size]

		if not has_enough_ink(candidate_gray):
			continue

		patch = image[y:y + patch_size, x:x + patch_size]
		if GRAYSCALE:
			patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
		patches.append(patch)

	# If the page is sparse, fill the remaining slots with random crops.
	while len(patches) < num_patches:
		x = random.randint(0, width - patch_size)
		y = random.randint(0, height - patch_size)
		patch = image[y:y + patch_size, x:x + patch_size]
		if GRAYSCALE:
			patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
		patches.append(patch)

	return patches


def save_page_patches(page_path, output_dir):
	# Extract and save one page's 20 patches.
	page_name = os.path.splitext(os.path.basename(page_path))[0]
	patches = extract_patches_from_page(page_path)

	for patch_index, patch in enumerate(patches):
		output_path = os.path.join(
			output_dir,
			f"{page_name}_p{patch_index:02d}.jpg",
		)
		cv2.imwrite(output_path, patch)

	return len(patches)


def build_patch_dataset():
	# Split pages and create 20 patches per page for every writer.
	random.seed(42)

	if not os.path.isdir(RAW_DATASET_DIR):
		raise FileNotFoundError(f"Dataset folder not found: {RAW_DATASET_DIR}")

	writers = sorted(
		writer
		for writer in os.listdir(RAW_DATASET_DIR)
		if os.path.isdir(os.path.join(RAW_DATASET_DIR, writer))
	)
	print(f"Found {len(writers)} writer folders: {writers}")

	os.makedirs(TRAIN_DIR, exist_ok=True)
	os.makedirs(TEST_DIR, exist_ok=True)

	for writer in writers:
		writer_dir = os.path.join(RAW_DATASET_DIR, writer)
		pages = list_writer_pages(writer_dir)

		if len(pages) < 2:
			print(f"  [WARN] {writer} has fewer than 2 pages, skipping")
			continue

		test_index = min(TEST_PAGE_INDEX, len(pages) - 1)
		train_pages = [page for index, page in enumerate(pages) if index != test_index]
		test_pages = [pages[test_index]]

		train_output = os.path.join(TRAIN_DIR, writer)
		test_output = os.path.join(TEST_DIR, writer)
		os.makedirs(train_output, exist_ok=True)
		os.makedirs(test_output, exist_ok=True)

		train_patch_count = sum(
			save_page_patches(page, train_output) for page in train_pages
		)
		test_patch_count = sum(
			save_page_patches(page, test_output) for page in test_pages
		)

		print(
			f"  {writer}: {len(train_pages)} train pages, "
			f"{len(test_pages)} test page -> "
			f"{train_patch_count} train patches, "
			f"{test_patch_count} test patches"
		)

	print("\nDone. Shared patch dataset at:", PATCH_DATASET_DIR)


if __name__ == "__main__":
	build_patch_dataset()
