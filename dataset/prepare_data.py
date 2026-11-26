"""
Dataset partitioning script for State Farm Distracted Driver Detection.
Maps multi-class Kaggle dataset into a binary classification task:
  c0                  -> 'normal'
  c1, c2, c3, c4      -> 'using_phone'
Splits files into train (70%), val (15%), and test (15%) partitions with deterministic seeding.
"""

import os
import shutil
import random
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("mobile_detection.dataset")


def partition_binary_dataset(
    source_dir: Path,
    output_dir: Path,
    class_mapping: Optional[Dict[str, str]] = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Dict[str, int]]:
    """
    Partitions raw images into structured binary folders:
      output_dir/
        train/
          normal/
          using_phone/
        val/
          normal/
          using_phone/
        test/
          normal/
          using_phone/

    Args:
        source_dir: Directory containing c0, c1, c2, c3, c4 subdirectories.
        output_dir: Destination path for partitioned binary dataset.
        class_mapping: Mapping from folder name to binary label.
        train_ratio: Fraction for training split (default: 0.70).
        val_ratio: Fraction for validation split (default: 0.15).
        seed: Random seed for deterministic reproducibility.

    Returns:
        Dict detailing the count of images in each split and class.
    """
    random.seed(seed)
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if class_mapping is None:
        class_mapping = {
            "c0": "normal",
            "c1": "using_phone",
            "c2": "using_phone",
            "c3": "using_phone",
            "c4": "using_phone"
        }

    # 1. Create target folder structure
    for split in ["train", "val", "test"]:
        for label in set(class_mapping.values()):
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)

    # 2. Pool images by target binary class
    pools = {label: [] for label in set(class_mapping.values())}
    for subfolder, target_label in class_mapping.items():
        folder_path = source_dir / subfolder
        if folder_path.exists():
            images = [f for f in folder_path.glob("*.jpg")] + [f for f in folder_path.glob("*.png")]
            pools[target_label].extend(images)
            logger.info(f"Loaded {len(images)} images from '{subfolder}' mapped to '{target_label}'")
        else:
            logger.warning(f"Subfolder '{folder_path}' does not exist, skipping.")

    stats = {"train": {}, "val": {}, "test": {}}

    # 3. Partition deterministically and copy files
    for label, files in pools.items():
        if not files:
            logger.warning(f"No files found for label '{label}'.")
            continue

        random.shuffle(files)
        total = len(files)
        train_cut = int(total * train_ratio)
        val_cut = train_cut + int(total * val_ratio)

        subsets = {
            "train": files[:train_cut],
            "val": files[train_cut:val_cut],
            "test": files[val_cut:]
        }

        logger.info(f"[CLASS: {label}] Total: {total} images")
        for split_name, subset_files in subsets.items():
            stats[split_name][label] = len(subset_files)
            logger.info(f"  └── {split_name.capitalize():<5} Set: {len(subset_files)} images")
            for img in subset_files:
                dest = output_dir / split_name / label / img.name
                shutil.copyfile(img, dest)

    logger.info("Binary dataset partitioned successfully.")
    return stats


if __name__ == "__main__":
    import argparse
    from configs.config import dataset_cfg

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Partition State Farm dataset into binary splits.")
    parser.add_argument("--source", type=str, default=str(dataset_cfg.source_dir), help="Source directory containing c0-c4")
    parser.add_argument("--output", type=str, default=str(dataset_cfg.output_dir), help="Output binary dataset directory")
    args = parser.parse_args()

    partition_binary_dataset(Path(args.source), Path(args.output))
