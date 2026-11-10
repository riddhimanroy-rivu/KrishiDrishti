"""
Dataset Preparation, Splitting, and Loading Pipeline for KrishiDrishti.
Handles PlantVillage tomato classes, validation, stratified splitting,
data augmentation for training, and TensorFlow dataset loading.
"""

from pathlib import Path
import os
import shutil
import json
import logging
from typing import Tuple, List, Dict, Optional
import numpy as np
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

from config.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
    DEFAULT_TOMATO_CLASSES,
    CLASS_NAMES_PATH,
    ALLOWED_EXTENSIONS,
)
from src.preprocessing import validate_image_file, preprocess_for_mobilenet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_dataset_root(raw_dir: Path = RAW_DATA_DIR) -> Optional[Path]:
    """
    Locates the actual root folder containing tomato classes.
    Handles nested directory structures (e.g., extracted zip folders like
    'PlantVillage', 'plantvillage-dataset/color', etc.).
    """
    if not raw_dir.exists():
        return None

    # Check directly inside raw_dir
    direct_subdirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    tomato_dirs = [p for p in direct_subdirs if "tomato" in p.name.lower()]
    if tomato_dirs:
        return raw_dir

    # Search up to 3 levels deep
    for root, dirs, _ in os.walk(raw_dir):
        t_dirs = [d for d in dirs if "tomato" in d.lower()]
        if len(t_dirs) >= 2:
            return Path(root)

    return raw_dir if direct_subdirs else None


def discover_tomato_classes(dataset_root: Path) -> List[str]:
    """
    Dynamically identifies all available tomato class folders in the dataset.
    Prioritizes folders containing 'tomato' (case-insensitive).
    """
    if not dataset_root.exists():
        return []

    subdirs = [p for p in dataset_root.iterdir() if p.is_dir()]
    tomato_subdirs = [p.name for p in subdirs if "tomato" in p.name.lower()]

    if tomato_subdirs:
        return sorted(tomato_subdirs)

    # Fallback to all subdirectories if specific tomato naming isn't present
    return sorted([p.name for p in subdirs])


def clean_and_collect_images(class_dir: Path) -> List[Path]:
    """
    Scans a class directory, verifies each image's integrity,
    and returns a list of valid image paths.
    """
    valid_images = []
    for file_path in class_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
            is_valid, _ = validate_image_file(file_path)
            if is_valid:
                valid_images.append(file_path)
            else:
                logger.warning(f"Skipping corrupted or unreadable image: {file_path}")
    return valid_images


def split_and_organize_dataset(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    train_ratio: float = TRAIN_SPLIT,
    val_ratio: float = VAL_SPLIT,
    test_ratio: float = TEST_SPLIT,
    seed: int = RANDOM_SEED,
) -> Dict[str, int]:
    """
    Performs stratified 70/15/15 dataset splitting across discovered classes.
    Organizes files into processed/train, processed/val, and processed/test.
    Prevents data leakage by ensuring mutually exclusive physical sets.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Splits must sum to 1.0"

    root = find_dataset_root(raw_dir)
    if not root:
        raise FileNotFoundError(f"No valid dataset folder found in {raw_dir}")

    classes = discover_tomato_classes(root)
    if not classes:
        raise ValueError(f"No class subfolders detected in {root}")

    logger.info(f"Discovered {len(classes)} classes: {classes}")

    # Prepare processed directories
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    test_dir = processed_dir / "test"

    for split_dir in [train_dir, val_dir, test_dir]:
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    all_class_names = []

    for class_name in classes:
        src_class_path = root / class_name
        valid_files = clean_and_collect_images(src_class_path)

        if len(valid_files) < 3:
            logger.warning(f"Class '{class_name}' has insufficient images ({len(valid_files)}). Skipping.")
            continue

        all_class_names.append(class_name)

        # Create class subdirectories in each split
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)
        (test_dir / class_name).mkdir(parents=True, exist_ok=True)

        # Stratified / deterministic splitting
        train_val_files, test_files = train_test_split(
            valid_files,
            test_size=test_ratio,
            random_state=seed,
            shuffle=True
        )

        relative_val_ratio = val_ratio / (train_ratio + val_ratio)
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=relative_val_ratio,
            random_state=seed,
            shuffle=True
        )

        # Copy files to designated splits
        for f in train_files:
            shutil.copy2(f, train_dir / class_name / f.name)
        for f in val_files:
            shutil.copy2(f, val_dir / class_name / f.name)
        for f in test_files:
            shutil.copy2(f, test_dir / class_name / f.name)

        summary[class_name] = {
            "total": len(valid_files),
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        }
        logger.info(f"Class {class_name}: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # Save sorted class names to json
    with open(CLASS_NAMES_PATH, "w") as fp:
        json.dump(all_class_names, fp, indent=2)

    logger.info(f"Class names saved to {CLASS_NAMES_PATH}")
    return summary


def create_sample_dataset_if_empty(
    target_raw_dir: Path = RAW_DATA_DIR,
    samples_per_class: int = 15
) -> bool:
    """
    Creates a lightweight synthetic verification dataset with 10 tomato classes
    if no raw dataset is present. Allows immediate execution of tests, training,
    and end-to-end evaluation out of the box.
    """
    root = find_dataset_root(target_raw_dir)
    if root and discover_tomato_classes(root):
        logger.info("Existing raw dataset detected. Sample generation skipped.")
        return False

    logger.info("No raw dataset found. Generating verification dataset for testing and demonstration...")

    palette_map = {
        "Tomato___Bacterial_spot": (120, 100, 30),
        "Tomato___Early_blight": (140, 80, 20),
        "Tomato___Late_blight": (80, 60, 40),
        "Tomato___Leaf_Mold": (160, 180, 50),
        "Tomato___Septoria_leaf_spot": (110, 110, 80),
        "Tomato___Spider_mites Two-spotted_spider_mite": (200, 190, 70),
        "Tomato___Target_Spot": (130, 70, 40),
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (210, 200, 30),
        "Tomato___Tomato_mosaic_virus": (100, 170, 60),
        "Tomato___healthy": (34, 139, 34),
    }

    np.random.seed(RANDOM_SEED)

    for class_name in DEFAULT_TOMATO_CLASSES:
        class_folder = target_raw_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        base_color = palette_map.get(class_name, (50, 120, 50))

        for idx in range(samples_per_class):
            img = Image.new("RGB", (256, 256), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)

            # Draw leaf shape
            leaf_box = [40, 30, 216, 226]
            draw.ellipse(leaf_box, fill=base_color, outline=(20, 60, 20))

            # Draw central vein
            draw.line([(128, 40), (128, 220)], fill=(20, 70, 20), width=3)
            # Secondary veins
            for y_off in range(70, 200, 25):
                draw.line([(128, y_off), (80, y_off - 15)], fill=(25, 75, 25), width=2)
                draw.line([(128, y_off), (176, y_off - 15)], fill=(25, 75, 25), width=2)

            # Draw specific disease pattern
            if "spot" in class_name.lower():
                for _ in range(8):
                    rx = int(np.random.randint(60, 190))
                    ry = int(np.random.randint(50, 200))
                    draw.ellipse([rx, ry, rx + 10, ry + 10], fill=(40, 20, 10), outline=(220, 200, 50))
            elif "blight" in class_name.lower():
                for _ in range(3):
                    rx = int(np.random.randint(60, 170))
                    ry = int(np.random.randint(60, 170))
                    draw.ellipse([rx, ry, rx + 30, ry + 30], fill=(70, 35, 10))
            elif "curl" in class_name.lower():
                draw.rectangle([50, 50, 80, 200], fill=(240, 230, 70))
            elif "spider" in class_name.lower():
                for _ in range(20):
                    rx = int(np.random.randint(60, 190))
                    ry = int(np.random.randint(60, 190))
                    draw.point((rx, ry), fill=(250, 250, 180))

            sample_file = class_folder / f"sample_{idx:03d}.jpg"
            img.save(sample_file, "JPEG", quality=90)

    logger.info(f"Generated sample dataset with {len(DEFAULT_TOMATO_CLASSES)} classes at {target_raw_dir}")
    return True


def get_dataset_statistics(processed_dir: Path = PROCESSED_DATA_DIR) -> Dict[str, dict]:
    """
    Computes summary metrics for train, val, and test splits.
    """
    stats = {}
    for split in ["train", "val", "test"]:
        split_path = processed_dir / split
        if not split_path.exists():
            continue
        class_counts = {}
        total = 0
        for class_dir in split_path.iterdir():
            if class_dir.is_dir():
                count = len([f for f in class_dir.iterdir() if f.is_file()])
                class_counts[class_dir.name] = count
                total += count
        stats[split] = {"total": total, "classes": class_counts}
    return stats
