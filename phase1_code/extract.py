import cv2
import numpy as np
import pandas as pd
import os
import glob

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image. Check the file path.")
        exit()
    h, w, _ = img.shape
    parts = {
        "top_left":     img[0:h//2, 0:w//2],
        "top_right":    img[0:h//2, w//2:w],
        "bottom_left":  img[h//2:h, 0:w//2],
        "bottom_right": img[h//2:h, w//2:w],
    }

    features = []
    feature_names = []

    for part_name, part in parts.items():
        b_channel = part[:, :, 0]
        g_channel = part[:, :, 1]
        r_channel = part[:, :, 2]

        for ch_name, ch_data in [("R", r_channel), ("G", g_channel), ("B", b_channel)]:
            features.append(np.mean(ch_data))
            features.append(np.var(ch_data))
            feature_names.append(f"{part_name}_{ch_name}_mean")
            feature_names.append(f"{part_name}_{ch_name}_var")

    return np.array(features), feature_names


def process_dataset(root_dir, output_csv="features.csv"):
    all_rows = []
    feature_names = None
    person_folders = sorted(glob.glob(os.path.join(root_dir, "*")))
    for person_folder in person_folders:
        if not os.path.isdir(person_folder):
            continue
        person_id = os.path.basename(person_folder)

        image_paths = sorted(
            glob.glob(os.path.join(person_folder, "*.jpg")) +
            glob.glob(os.path.join(person_folder, "*.jpeg")) +
            glob.glob(os.path.join(person_folder, "*.png"))
        )

        for img_path in image_paths:
            feats, feature_names = extract_features(img_path)
            row = {"person_id": person_id, "image_name": os.path.basename(img_path)}
            row.update({name: val for name, val in zip(feature_names, feats)})
            all_rows.append(row)
            print(f"Processed: {person_id}/{os.path.basename(img_path)}")

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} rows x {len(feature_names)} features to {output_csv}")
    return df


if __name__ == "__main__":
    import sys
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "ML_dataset"
    process_dataset(root_dir)