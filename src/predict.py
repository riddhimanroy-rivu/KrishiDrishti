"""
Inference and Real-Time Prediction Module for KrishiDrishti.
Loads the trained MobileNetV2 model, preprocesses uploaded images,
and produces actual predictions, confidence scores, top-k alternatives,
and mapped educational agricultural disease profiles.
"""

from pathlib import Path
import json
import logging
from typing import Dict, Any, List, Union, Optional
import io
import numpy as np
from PIL import Image
import tensorflow as tf

from config.config import (
    BEST_MODEL_PATH,
    CLASS_NAMES_PATH,
    DEFAULT_TOMATO_CLASSES,
)
from src.preprocessing import preprocess_input_image, ImageValidationError
from src.disease_info import get_disease_info, AGRICULTURAL_DISCLAIMER

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Module-level model cache to avoid repeated disk I/O in non-Streamlit contexts
_LOADED_MODEL: Optional[tf.keras.Model] = None
_LOADED_CLASSES: Optional[List[str]] = None


def is_model_available(model_path: Path = BEST_MODEL_PATH) -> bool:
    """Checks whether the trained model file exists on disk."""
    return Path(model_path).exists()


def load_class_names(class_names_path: Path = CLASS_NAMES_PATH) -> List[str]:
    """Loads class names from saved JSON; falls back to default classes if not found."""
    p = Path(class_names_path)
    if p.exists():
        try:
            with open(p, "r") as fp:
                return json.load(fp)
        except Exception as exc:
            logger.warning(f"Could not read class names from {p}: {exc}")
    return DEFAULT_TOMATO_CLASSES


def load_prediction_model(model_path: Path = BEST_MODEL_PATH) -> tf.keras.Model:
    """
    Loads and caches the trained Keras model for inference.
    Raises FileNotFoundError if model has not been trained yet.
    """
    global _LOADED_MODEL
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{p}'.\n"
            "Please train the model first by running:\n"
            "   python -m src.train\n"
            "or review the documentation."
        )

    if _LOADED_MODEL is None:
        logger.info(f"Loading model into memory from {p}...")
        _LOADED_MODEL = tf.keras.models.load_model(str(p), compile=False)

    return _LOADED_MODEL


def predict_crop_disease(
    image_input: Union[str, Path, io.BytesIO, bytes, Image.Image],
    model: Optional[tf.keras.Model] = None,
    class_names: Optional[List[str]] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    End-to-end real inference pipeline:
    1. Validates and preprocesses image input into MobileNetV2 normalized tensor (1, 224, 224, 3).
    2. Runs model.predict() to obtain actual class probabilities.
    3. Computes top prediction, confidence score, and top-k rankings.
    4. Attaches verified agricultural knowledge and symptoms.

    Returns:
        Structured prediction result dictionary.

    Raises:
        FileNotFoundError: If model is missing.
        ImageValidationError: If image cannot be read or is invalid.
    """
    # Load model if not provided
    if model is None:
        model = load_prediction_model()

    # Load class names if not provided
    if class_names is None:
        class_names = load_class_names()

    # Preprocess image into tensor
    input_tensor = preprocess_input_image(image_input, expand_dims=True)

    # Perform real forward pass
    raw_predictions = model.predict(input_tensor, verbose=0)
    probabilities = raw_predictions[0]

    num_model_classes = len(probabilities)
    if num_model_classes != len(class_names):
        logger.warning(
            f"Class count mismatch: model has {num_model_classes} outputs, but {len(class_names)} class names found. Adjusting."
        )
        if len(class_names) < num_model_classes:
            class_names = class_names + [f"Class_{i}" for i in range(len(class_names), num_model_classes)]
        else:
            class_names = class_names[:num_model_classes]

    # Best prediction
    best_idx = int(np.argmax(probabilities))
    best_class = class_names[best_idx]
    best_confidence = float(probabilities[best_idx])

    # Top-K predictions
    k = min(top_k, num_model_classes)
    top_indices = np.argsort(probabilities)[::-1][:k]

    top_predictions = []
    for idx in top_indices:
        c_name = class_names[idx]
        conf = float(probabilities[idx])
        top_predictions.append({
            "class_name": c_name,
            "display_name": c_name.replace("Tomato___", "").replace("_", " "),
            "confidence": conf,
            "confidence_percent": f"{conf * 100:.2f}%",
        })

    # Retrieve agricultural knowledge base profile
    disease_profile = get_disease_info(best_class)
    is_healthy = "healthy" in best_class.lower()

    return {
        "success": True,
        "predicted_class": best_class,
        "display_name": disease_profile.get("display_name", best_class),
        "status": "Healthy" if is_healthy else "Diseased",
        "confidence": best_confidence,
        "confidence_percent": f"{best_confidence * 100:.2f}%",
        "top_predictions": top_predictions,
        "causal_agent": disease_profile.get("causal_agent", "N/A"),
        "severity": disease_profile.get("severity", "N/A"),
        "description": disease_profile.get("description", ""),
        "symptoms": disease_profile.get("symptoms", []),
        "preventive_practices": disease_profile.get("preventive_practices", []),
        "recommendations": disease_profile.get("recommendations", []),
        "disclaimer": AGRICULTURAL_DISCLAIMER,
    }
