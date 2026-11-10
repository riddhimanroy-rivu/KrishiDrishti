"""
Model Training Pipeline for KrishiDrishti.
Implements Phase 1 (Transfer Learning feature extraction) and Phase 2 (Fine-Tuning),
EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, history recording, and plotting.
"""

from pathlib import Path
import json
import logging
import argparse
from typing import Dict, Any, Tuple
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks

from config.config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    PREPROCESS_CONFIG_PATH,
    PLOTS_DIR,
    METRICS_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    INITIAL_EPOCHS,
    FINE_TUNE_EPOCHS,
    INITIAL_LEARNING_RATE,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_PATIENCE,
    RANDOM_SEED,
)
from src.model import build_transfer_learning_model, configure_fine_tuning
from src.data_loader import (
    split_and_organize_dataset,
    create_sample_dataset_if_empty,
    discover_tomato_classes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int = RANDOM_SEED):
    """Sets deterministic seeds across Python, NumPy, and TensorFlow."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def create_data_augmentation_pipeline() -> tf.keras.Sequential:
    """
    Creates data augmentation pipeline applied exclusively to training data.
    Simulates variations in orientation, zoom, and lighting.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical", seed=RANDOM_SEED),
            layers.RandomRotation(0.15, fill_mode="reflect", seed=RANDOM_SEED),
            layers.RandomZoom(0.1, fill_mode="reflect", seed=RANDOM_SEED),
            layers.RandomContrast(0.1, seed=RANDOM_SEED),
        ],
        name="data_augmentation",
    )


def load_tf_datasets(
    train_dir: Path = TRAIN_DIR,
    val_dir: Path = VAL_DIR,
    test_dir: Path = TEST_DIR,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list]:
    """
    Loads training, validation, and test datasets as tf.data.Dataset pipelines.
    Applies data augmentation to train set and MobileNetV2 rescaling to all sets.
    """
    logger.info(f"Loading training data from {train_dir}...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=RANDOM_SEED,
    )
    class_names = train_ds.class_names

    logger.info(f"Loading validation data from {val_dir}...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    logger.info(f"Loading test data from {test_dir}...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        label_mode="categorical",
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    # Rescaling layer: maps [0, 255] to [-1, 1] matching MobileNetV2 expectations
    rescale_layer = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)
    data_augmentation = create_data_augmentation_pipeline()

    # Apply augmentation ONLY to training data, then normalize
    train_ds = train_ds.map(
        lambda x, y: (rescale_layer(data_augmentation(x, training=True)), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.map(
        lambda x, y: (rescale_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    test_ds = test_ds.map(
        lambda x, y: (rescale_layer(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Prefetch for performance
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def plot_and_save_curves(history_data: Dict[str, list], output_dir: Path = PLOTS_DIR):
    """
    Plots and exports training vs validation accuracy and loss curves.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs_range = range(1, len(history_data.get("accuracy", [])) + 1)

    # Accuracy Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history_data["accuracy"], label="Training Accuracy", color="#2e7d32", lw=2)
    plt.plot(epochs_range, history_data["val_accuracy"], label="Validation Accuracy", color="#0288d1", lw=2, linestyle="--")
    plt.title("KrishiDrishti - Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    acc_path = output_dir / "accuracy_history.png"
    plt.savefig(acc_path, dpi=300)
    plt.close()
    logger.info(f"Accuracy curve saved to {acc_path}")

    # Loss Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history_data["loss"], label="Training Loss", color="#c62828", lw=2)
    plt.plot(epochs_range, history_data["val_loss"], label="Validation Loss", color="#f57c00", lw=2, linestyle="--")
    plt.title("KrishiDrishti - Training vs Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    loss_path = output_dir / "loss_history.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()
    logger.info(f"Loss curve saved to {loss_path}")


def train_model(
    initial_epochs: int = INITIAL_EPOCHS,
    fine_tune: bool = False,
    fine_tune_epochs: int = FINE_TUNE_EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> Dict[str, Any]:
    """
    Executes complete training routine:
    1. Verifies dataset splits; prepares them if missing.
    2. Builds MobileNetV2 transfer learning model.
    3. Runs Phase 1 training with frozen backbone.
    4. Optionally runs Phase 2 fine-tuning on top backbone layers.
    5. Saves the best model, class names, preprocessing config, and training curves.
    """
    set_seed(RANDOM_SEED)

    # Verify processed splits exist; if not, split or generate verification set
    if not (TRAIN_DIR.exists() and any(TRAIN_DIR.iterdir())):
        logger.info("Processed dataset directories empty. Checking raw dataset...")
        try:
            split_and_organize_dataset()
        except Exception as exc:
            logger.warning(f"Dataset split error: {exc}. Generating verification dataset...")
            create_sample_dataset_if_empty()
            split_and_organize_dataset()

    train_ds, val_ds, test_ds, class_names = load_tf_datasets(batch_size=batch_size)
    num_classes = len(class_names)
    logger.info(f"Initialized datasets for {num_classes} classes: {class_names}")

    # Save class names JSON
    CLASS_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_NAMES_PATH, "w") as fp:
        json.dump(class_names, fp, indent=2)

    # Save Preprocessing Config JSON
    PREPROCESS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    prep_config = {
        "model_architecture": "MobileNetV2 Transfer Learning",
        "input_shape": [IMAGE_SIZE[0], IMAGE_SIZE[1], 3],
        "normalization": "MobileNetV2 [-1.0, 1.0]",
        "num_classes": num_classes,
        "classes": class_names,
    }
    with open(PREPROCESS_CONFIG_PATH, "w") as fp:
        json.dump(prep_config, fp, indent=2)

    # Build model (Phase 1: Feature Extraction)
    logger.info("Building MobileNetV2 model with frozen backbone...")
    model = build_transfer_learning_model(num_classes=num_classes)

    # Callbacks
    checkpoint_cb = callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )
    early_stop_cb = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=REDUCE_LR_PATIENCE,
        min_lr=1e-7,
        verbose=1,
    )

    all_callbacks = [checkpoint_cb, early_stop_cb, reduce_lr_cb]

    logger.info(f"Commencing Phase 1 training for {initial_epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epochs,
        callbacks=all_callbacks,
    )

    combined_history = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    # Phase 2: Optional Fine-Tuning
    if fine_tune:
        logger.info(f"Unfreezing top backbone layers for Phase 2 fine-tuning ({fine_tune_epochs} epochs)...")
        model = configure_fine_tuning(model, unfreeze_from_layer=100)
        total_epochs = initial_epochs + fine_tune_epochs

        ft_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=total_epochs,
            initial_epoch=history.epoch[-1] + 1,
            callbacks=all_callbacks,
        )

        for k, vals in ft_history.history.items():
            combined_history[k].extend([float(v) for v in vals])

    # Plot & Save curves
    plot_and_save_curves(combined_history)

    # Save training history JSON
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    history_file = METRICS_DIR / "training_history.json"
    with open(history_file, "w") as fp:
        json.dump(combined_history, fp, indent=2)

    logger.info(f"Training history saved to {history_file}")
    logger.info(f"Best model saved to {BEST_MODEL_PATH}")

    return {
        "class_names": class_names,
        "history": combined_history,
        "best_model_path": str(BEST_MODEL_PATH),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KrishiDrishti Model Training")
    parser.add_argument("--epochs", type=int, default=INITIAL_EPOCHS, help="Number of initial epochs")
    parser.add_argument("--fine-tune", action="store_true", help="Perform backbone fine-tuning")
    parser.add_argument("--ft-epochs", type=int, default=FINE_TUNE_EPOCHS, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    train_model(
        initial_epochs=args.epochs,
        fine_tune=args.fine_tune,
        fine_tune_epochs=args.ft_epochs,
        batch_size=args.batch_size,
    )
