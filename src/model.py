"""
Deep Learning Model Architecture for KrishiDrishti.
Implements MobileNetV2 Transfer Learning with ImageNet pretrained weights,
Global Average Pooling, Dropout, and Softmax classification head.
Supports Phase 1 (feature extraction) and Phase 2 (fine-tuning).
"""

from pathlib import Path
from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers

from config.config import (
    INPUT_SHAPE,
    INITIAL_LEARNING_RATE,
    FINE_TUNE_LEARNING_RATE,
    DROPOUT_RATE,
)


def build_transfer_learning_model(
    num_classes: int,
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    dropout_rate: float = DROPOUT_RATE,
    trainable_base: bool = False,
    l2_reg: float = 1e-4,
) -> tf.keras.Model:
    """
    Constructs the KrishiDrishti CNN model using MobileNetV2 as the feature extractor.

    Architecture:
    1. Input Layer: (224, 224, 3)
    2. MobileNetV2 backbone (ImageNet weights, include_top=False)
    3. GlobalAveragePooling2D
    4. BatchNormalization (stabilizes activation distributions)
    5. Dense layer (256 units, ReLU, L2 regularized)
    6. Dropout layer (rate = dropout_rate)
    7. Dense Output layer (num_classes units, Softmax)

    Returns:
        Compiled tf.keras.Model
    """
    inputs = layers.Input(shape=input_shape, name="input_image")

    # Locate pre-downloaded ImageNet weights to avoid network queries in sandboxed/offline environments
    from config.config import BASE_DIR
    local_weights = BASE_DIR / ".cache" / "keras" / "models" / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
    home_weights = Path.home() / ".keras" / "models" / "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"

    weights_arg = "imagenet"
    if local_weights.exists():
        weights_arg = str(local_weights)
    elif home_weights.exists():
        weights_arg = str(home_weights)

    # MobileNetV2 pretrained backbone
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights_arg,
    )
    base_model.trainable = trainable_base

    # Forward pass through backbone
    x = base_model(inputs, training=False)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_batch_norm")(x)
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(l2_reg),
        name="dense_feature_extractor"
    )(x)
    x = layers.Dropout(dropout_rate, name="dropout_regularizer")(x)

    # Output Softmax Layer
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="disease_predictions"
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="KrishiDrishti_MobileNetV2")

    # Compile model with Adam optimizer and Categorical Crossentropy
    optimizer = optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def configure_fine_tuning(
    model: tf.keras.Model,
    unfreeze_from_layer: int = 100,
    fine_tune_lr: float = FINE_TUNE_LEARNING_RATE,
) -> tf.keras.Model:
    """
    Unfreezes top layers of the MobileNetV2 backbone for Phase 2 fine-tuning.
    Keeps BatchNormalization layers in inference mode to avoid distorting running stats.
    Recompiles model with a lower learning rate.
    """
    # Locate the MobileNetV2 layer within the functional model
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "mobilenetv2" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("MobileNetV2 backbone layer not detected in model hierarchy.")

    # Enable backbone training
    base_model.trainable = True

    # Freeze all layers before `unfreeze_from_layer`
    for i, layer in enumerate(base_model.layers):
        if i < unfreeze_from_layer or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    # Recompile with smaller learning rate for fine-tuning
    model.compile(
        optimizer=optimizers.Adam(learning_rate=fine_tune_lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_model_summary_str(model: tf.keras.Model) -> str:
    """Returns the string representation of model.summary()."""
    lines = []
    model.summary(print_fn=lambda x: lines.append(x))
    return "\n".join(lines)
