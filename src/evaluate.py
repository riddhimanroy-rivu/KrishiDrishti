"""
Model Evaluation and Metrics Module for KrishiDrishti.
Computes comprehensive test set evaluation: accuracy, precision, recall, F1,
confusion matrix plot, and detailed per-class classification reports.
"""

from pathlib import Path
import json
import logging
from typing import Dict, Any, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

from config.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    TEST_DIR,
    PLOTS_DIR,
    REPORTS_DIR,
    METRICS_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_test_data(
    test_dir: Path = TEST_DIR,
    class_names: list = None,
    image_size: Tuple[int, int] = IMAGE_SIZE,
    batch_size: int = BATCH_SIZE,
) -> Tuple[tf.data.Dataset, np.ndarray]:
    """
    Loads test dataset and extracts true class labels.
    """
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found at {test_dir}. Please run dataset preparation.")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,  # Essential for aligning predictions with true labels
    )

    # Extract true labels
    y_true_list = []
    for _, labels_batch in test_ds:
        y_true_list.append(np.argmax(labels_batch.numpy(), axis=1))
    y_true = np.concatenate(y_true_list, axis=0)

    # Normalize images for MobileNetV2
    rescale_layer = tf.keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1.0)
    norm_test_ds = test_ds.map(lambda x, y: (rescale_layer(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    norm_test_ds = norm_test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return norm_test_ds, y_true


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    output_path: Path = PLOTS_DIR / "confusion_matrix.png",
):
    """
    Renders and saves a high-resolution Seaborn confusion matrix heatmap.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Format class labels for readable display
    display_labels = [c.replace("Tomato___", "").replace("_", " ") for c in class_names]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar=True,
    )
    plt.title("KrishiDrishti - Test Set Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.ylabel("Ground Truth Class", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix plot saved to {output_path}")


def evaluate_model(
    model_path: Path = BEST_MODEL_PATH,
    test_dir: Path = TEST_DIR,
    class_names_path: Path = CLASS_NAMES_PATH,
) -> Dict[str, Any]:
    """
    Evaluates the trained model on test data, computing metrics and saving reports.

    Returns:
        Dictionary containing summary metrics and per-class reports.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")

    if not class_names_path.exists():
        raise FileNotFoundError(f"Class names file not found at {class_names_path}.")

    with open(class_names_path, "r") as fp:
        class_names = json.load(fp)

    logger.info(f"Loading model from {model_path} for evaluation...")
    model = tf.keras.models.load_model(model_path)

    logger.info("Loading test dataset...")
    test_ds, y_true = load_test_data(test_dir, class_names=class_names)

    logger.info("Running model inference on test dataset...")
    y_pred_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Core metrics
    acc = float(accuracy_score(y_true, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Detailed classification report
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_and_save_confusion_matrix(cm, class_names)

    # Save reports
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_txt_path = REPORTS_DIR / "classification_report.txt"
    with open(report_txt_path, "w") as fp:
        fp.write("KrishiDrishti - Test Set Classification Report\n")
        fp.write("=" * 60 + "\n\n")
        fp.write(report_text)

    # Save metrics JSON
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_summary = {
        "test_accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "weighted_f1": float(f1_weighted),
        "num_test_samples": int(len(y_true)),
        "num_classes": len(class_names),
        "class_names": class_names,
        "classification_report": report_dict,
    }

    metrics_json_path = METRICS_DIR / "evaluation_metrics.json"
    with open(metrics_json_path, "w") as fp:
        json.dump(metrics_summary, fp, indent=2)

    logger.info(f"Evaluation complete. Test Accuracy: {acc * 100:.2f}%")
    logger.info(f"Metrics saved to {metrics_json_path}")

    return metrics_summary


def load_cached_evaluation_metrics() -> Dict[str, Any]:
    """
    Loads saved evaluation metrics if available; returns empty dict if not.
    """
    metrics_json_path = METRICS_DIR / "evaluation_metrics.json"
    if metrics_json_path.exists():
        with open(metrics_json_path, "r") as fp:
            return json.load(fp)
    return {}


if __name__ == "__main__":
    evaluate_model()
