"""
Unit Tests for Dataset Preparation and Splitting.
"""

from pathlib import Path
import pytest
import shutil
from src.data_loader import (
    discover_tomato_classes,
    create_sample_dataset_if_empty,
    split_and_organize_dataset,
    get_dataset_statistics,
)
from config.config import DEFAULT_TOMATO_CLASSES


@pytest.fixture
def temp_dataset_dir(tmp_path):
    """Creates a temporary isolated test data directory."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    return raw_dir, processed_dir


def test_create_sample_dataset(temp_dataset_dir):
    """Verifies that sample verification dataset generates all 10 tomato classes."""
    raw_dir, _ = temp_dataset_dir
    created = create_sample_dataset_if_empty(target_raw_dir=raw_dir, samples_per_class=4)
    assert created is True

    classes = discover_tomato_classes(raw_dir)
    assert len(classes) == 10
    assert set(classes) == set(DEFAULT_TOMATO_CLASSES)


def test_split_and_organize_dataset(temp_dataset_dir):
    """Verifies stratified splitting into train, val, and test subsets without data leakage."""
    raw_dir, processed_dir = temp_dataset_dir
    create_sample_dataset_if_empty(target_raw_dir=raw_dir, samples_per_class=10)

    summary = split_and_organize_dataset(
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    assert len(summary) == 10
    for cls_name, counts in summary.items():
        assert counts["total"] == 10
        assert counts["train"] >= 5
        assert counts["val"] >= 1
        assert counts["test"] >= 1
        # No overlap / leakage
        assert counts["train"] + counts["val"] + counts["test"] == counts["total"]

    # Verify physical file layout
    train_files = list((processed_dir / "train").glob("*/*"))
    val_files = list((processed_dir / "val").glob("*/*"))
    test_files = list((processed_dir / "test").glob("*/*"))

    assert len(train_files) > 0
    assert len(val_files) > 0
    assert len(test_files) > 0
