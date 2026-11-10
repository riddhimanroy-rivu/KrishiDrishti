"""
Unit Tests for Preprocessing and Image Validation Pipeline.
"""

import io
import pytest
import numpy as np
from PIL import Image

from src.preprocessing import (
    validate_image_file,
    load_and_prepare_image,
    preprocess_for_mobilenet,
    preprocess_input_image,
    ImageValidationError,
)
from config.config import IMAGE_SIZE


@pytest.fixture
def valid_pil_image():
    """Generates a valid 300x300 RGB PIL image."""
    arr = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def valid_image_bytes(valid_pil_image):
    """Generates valid JPEG image bytes in memory."""
    buf = io.BytesIO()
    valid_pil_image.save(buf, format="JPEG")
    buf.seek(0)
    return buf.getvalue()


def test_validate_image_file_success(valid_image_bytes):
    """Valid image bytes should pass validation with (True, None)."""
    is_valid, err = validate_image_file(valid_image_bytes)
    assert is_valid is True
    assert err is None


def test_validate_image_file_corrupted():
    """Corrupted / arbitrary non-image bytes should be rejected."""
    corrupted_bytes = b"NOT_A_VALID_JPEG_HEADER_CORRUPTED_DATA_12345"
    is_valid, err = validate_image_file(corrupted_bytes)
    assert is_valid is False
    assert err is not None
    assert "not a valid" in err.lower() or "corrupted" in err.lower()


def test_validate_image_file_too_small():
    """Tiny images under 10x10 px should be rejected as invalid."""
    tiny_img = Image.new("RGB", (4, 4), color="green")
    buf = io.BytesIO()
    tiny_img.save(buf, format="PNG")
    is_valid, err = validate_image_file(buf.getvalue())
    assert is_valid is False
    assert "too small" in err.lower()


def test_load_and_prepare_image_resizing(valid_pil_image):
    """Images should be converted to RGB and resized to exact target (224, 224)."""
    processed = load_and_prepare_image(valid_pil_image, target_size=IMAGE_SIZE)
    assert processed.size == IMAGE_SIZE
    assert processed.mode == "RGB"


def test_preprocess_for_mobilenet_range():
    """Normalization should scale pixel values strictly into [-1.0, 1.0]."""
    test_arr = np.array([[[0, 127.5, 255]]], dtype=np.float32)
    norm = preprocess_for_mobilenet(test_arr)
    assert np.isclose(norm[0, 0, 0], -1.0, atol=1e-4)
    assert np.isclose(norm[0, 0, 1], 0.0, atol=1e-4)
    assert np.isclose(norm[0, 0, 2], 1.0, atol=1e-4)


def test_preprocess_input_image_end_to_end(valid_image_bytes):
    """End-to-end preprocessing should produce tensor of shape (1, 224, 224, 3) with float32."""
    tensor = preprocess_input_image(valid_image_bytes, expand_dims=True)
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.float32
    assert tensor.min() >= -1.05
    assert tensor.max() <= 1.05


def test_preprocess_input_image_raises_on_invalid():
    """Invalid input should raise ImageValidationError without crashing application."""
    with pytest.raises(ImageValidationError):
        preprocess_input_image(b"broken_bytes")
