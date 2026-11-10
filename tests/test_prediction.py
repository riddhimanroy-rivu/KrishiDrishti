"""
Unit Tests for Prediction and Inference Pipeline.
"""

from pathlib import Path
import io
import pytest
import numpy as np
from PIL import Image
import tensorflow as tf

from src.predict import (
    predict_crop_disease,
    is_model_available,
    load_class_names,
)
from src.disease_info import get_disease_info
from config.config import DEFAULT_TOMATO_CLASSES


@pytest.fixture
def dummy_trained_model(tmp_path):
    """Creates a tiny dummy Keras model for inference testing."""
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


@pytest.fixture
def sample_leaf_image():
    """Generates a valid green leaf PIL image."""
    arr = np.full((224, 224, 3), 100, dtype=np.uint8)
    arr[:, :, 1] = 200  # Bright green
    return Image.fromarray(arr, "RGB")


def test_disease_info_retrieval():
    """Verifies that disease knowledge base retrieves valid profiles for all classes."""
    for c_name in DEFAULT_TOMATO_CLASSES:
        info = get_disease_info(c_name)
        assert "display_name" in info
        assert "status" in info
        assert "description" in info
        assert len(info["symptoms"]) > 0
        assert len(info["preventive_practices"]) > 0
        assert len(info["recommendations"]) > 0


def test_predict_crop_disease_structure(dummy_trained_model, sample_leaf_image):
    """Verifies complete output schema from real model inference."""
    result = predict_crop_disease(
        sample_leaf_image,
        model=dummy_trained_model,
        class_names=DEFAULT_TOMATO_CLASSES,
        top_k=3,
    )

    assert result["success"] is True
    assert "predicted_class" in result
    assert "display_name" in result
    assert result["status"] in ["Healthy", "Diseased"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert len(result["top_predictions"]) == 3
    assert "disclaimer" in result

    # Check top_predictions ordering
    confs = [p["confidence"] for p in result["top_predictions"]]
    assert confs == sorted(confs, reverse=True)


def test_is_model_available_nonexistent():
    """Verifies model availability check returns False when path does not exist."""
    assert is_model_available(Path("/nonexistent/model.keras")) is False
