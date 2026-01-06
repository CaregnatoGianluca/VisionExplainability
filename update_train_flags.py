#!/usr/bin/env python3
"""
Script to update the 'train' flag in heatmap_scores.json files based on
whether the images are in the training or test datasets.

Just fix purposes.
"""

import json
import os
from pathlib import Path


def update_cub_train_flags():
    """Update train flags for CUB dataset."""
    print("Processing CUB dataset...")

    # Locate CUB dataset metadata (images.txt + train_test_split.txt)
    cub_base = Path.cwd() / "CUB" / "DATASET" / "CUB_200_2011"
    if not cub_base.exists():
        print(f"CUB dataset folder not found at {cub_base}, skipping CUB.")
        return

    # Load train_test_split.txt to determine which images are training
    train_ids = set()
    with open(cub_base / "train_test_split.txt", "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            image_id, is_train = parts[0], parts[1]
            if int(is_train) == 1:
                train_ids.add(int(image_id))

    print(f"Found {len(train_ids)} training images in CUB")

    # Find all heatmap_scores.json files under the repo that look like CUB outputs
    workspace = Path.cwd()
    cub_jsons = list(workspace.rglob("heatmap_scores.json"))

    updated_total = 0
    for cub_json_path in cub_jsons:
        # Only process files that contain a CUB directory in their path
        if "CUB" not in str(cub_json_path):
            continue

        with open(cub_json_path, "r") as f:
            try:
                cub_data = json.load(f)
            except Exception:
                continue

        updated_count = 0
        for key, entry in cub_data.items():
            idx = entry.get("index")
            if isinstance(idx, int):
                is_train = int(idx) in train_ids
                if entry.get("train") != is_train:
                    entry["train"] = is_train
                    updated_count += 1

        if updated_count > 0:
            with open(cub_json_path, "w") as f:
                json.dump(cub_data, f, indent=4)

        print(f"Updated {updated_count} entries in {cub_json_path}")
        updated_total += updated_count

    print(f"Total updated entries across CUB files: {updated_total}")


def update_cxr_train_flags():
    """Update train flags for CXR dataset."""
    print("\nProcessing CXR dataset...")

    cxr_base = Path.cwd() / "CXR"
    train_dir = cxr_base / "train"
    train_stems = set()

    if train_dir.exists():
        for p in train_dir.rglob("*"):
            if p.is_file():
                train_stems.add(p.stem)

    print(f"Found {len(train_stems)} training images in CXR")

    # Find all heatmap_scores.json files under the repo that look like CXR outputs
    workspace = Path.cwd()
    cxr_jsons = list(workspace.rglob("heatmap_scores.json"))

    updated_total = 0
    for cxr_json_path in cxr_jsons:
        if "CXR" not in str(cxr_json_path):
            continue

        with open(cxr_json_path, "r") as f:
            try:
                cxr_data = json.load(f)
            except Exception:
                continue

        updated_count = 0
        for key, entry in cxr_data.items():
            idx = entry.get("index")
            if isinstance(idx, str):
                # Compare by stem (filename without extension)
                is_train = idx in train_stems
                if entry.get("train") != is_train:
                    entry["train"] = is_train
                    updated_count += 1

        if updated_count > 0:
            with open(cxr_json_path, "w") as f:
                json.dump(cxr_data, f, indent=4)

        print(f"Updated {updated_count} entries in {cxr_json_path}")
        updated_total += updated_count

    print(f"Total updated entries across CXR files: {updated_total}")


if __name__ == "__main__":
    update_cub_train_flags()
    update_cxr_train_flags()
    print("\nDone! Train flags have been updated.")
