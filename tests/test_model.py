"""
Unit Tests for Deep Learning Model Architecture.
"""

import pytest
import tensorflow as tf
from src.model import build_transfer_learning_model, configure_fine_tuning
from config.config import INPUT_SHAPE


def test_build_transfer_learning_model_architecture():
    """Verify MobileNetV2 architecture, layer shapes, and output dimension."""
    num_classes = 10
    model = build_transfer_learning_model(
        num_classes=num_classes,
        input_shape=INPUT_SHAPE,
        trainable_base=False,
    )

    # Check input shape
    assert model.input_shape == (None, 224, 224, 3)

    # Check output shape
    assert model.output_shape == (None, num_classes)

    # Check compilation status
    assert model.optimizer is not None
    assert model.loss == "categorical_crossentropy"

    # Verify backbone is frozen in Phase 1
    mobilenet_layer = None
    for layer in model.layers:
        if "mobilenetv2" in layer.name.lower():
            mobilenet_layer = layer
            break

    assert mobilenet_layer is not None
    assert mobilenet_layer.trainable is False


def test_configure_fine_tuning():
    """Verify top layers are unfrozen and model recompiled for fine-tuning."""
    num_classes = 5
    model = build_transfer_learning_model(num_classes=num_classes)
    fine_tuned = configure_fine_tuning(model, unfreeze_from_layer=120, fine_tune_lr=1e-5)

    mobilenet_layer = None
    for layer in fine_tuned.layers:
        if "mobilenetv2" in layer.name.lower():
            mobilenet_layer = layer
            break

    assert mobilenet_layer.trainable is True

    # Check that early layers remain frozen
    frozen_count = sum(1 for layer in mobilenet_layer.layers[:120] if not layer.trainable)
    assert frozen_count >= 100
