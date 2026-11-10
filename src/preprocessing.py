"""
Image Preprocessing and Validation Pipeline for KrishiDrishti.
Ensures identical preprocessing during dataset training and real-time inference.
"""

from pathlib import Path
import io
from typing import Tuple, Union, Optional
import numpy as np
from PIL import Image, UnidentifiedImageError
from config.config import IMAGE_SIZE, ALLOWED_EXTENSIONS


class ImageValidationError(Exception):
    """Raised when an uploaded or input image fails validation checks."""
    pass


def validate_image_file(
    image_source: Union[str, Path, io.BytesIO, bytes]
) -> Tuple[bool, Optional[str]]:
    """
    Validates whether an input file exists, has an allowed extension,
    and is a valid readable image.

    Returns:
        (True, None) if valid.
        (False, error_message) if invalid.
    """
    try:
        # Check file extension if path is provided
        if isinstance(image_source, (str, Path)):
            p = Path(image_source)
            if not p.exists():
                return False, f"File does not exist: {p.name}"
            if p.suffix.lower() not in ALLOWED_EXTENSIONS:
                return (
                    False,
                    f"Unsupported file format '{p.suffix}'. Supported formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
                )

        # Attempt to open and verify image bytes
        if isinstance(image_source, bytes):
            image_stream = io.BytesIO(image_source)
        elif isinstance(image_source, io.BytesIO):
            image_stream = image_source
            image_stream.seek(0)
        else:
            image_stream = open(image_source, "rb")

        try:
            with Image.open(image_stream) as img:
                img.verify()  # Verifies integrity without decoding entire pixel buffer
            
            # Reopen to check dimensions after verify() closed it
            if isinstance(image_source, (io.BytesIO, bytes)):
                image_stream.seek(0)
                with Image.open(image_stream) as img:
                    w, h = img.size
            else:
                with Image.open(image_source) as img:
                    w, h = img.size

            if w < 10 or h < 10:
                return False, f"Image resolution ({w}x{h}) is too small to be a valid leaf image."

            return True, None
        finally:
            if isinstance(image_source, (str, Path)):
                image_stream.close()

    except UnidentifiedImageError:
        return False, "File is not a valid or recognized image."
    except Exception as exc:
        return False, f"Corrupted or unreadable image file: {str(exc)}"


def load_and_prepare_image(
    image_source: Union[str, Path, io.BytesIO, bytes, Image.Image],
    target_size: Tuple[int, int] = IMAGE_SIZE
) -> Image.Image:
    """
    Opens an image safely, converts to 3-channel RGB, and resizes to target_size.
    Does not apply mathematical normalization yet, returning a clean PIL Image.
    """
    if isinstance(image_source, Image.Image):
        img = image_source
    elif isinstance(image_source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image_source))
    elif isinstance(image_source, io.BytesIO):
        image_source.seek(0)
        img = Image.open(image_source)
    else:
        img = Image.open(image_source)

    # Convert to RGB (handles RGBA, Palette, Grayscale, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize with high-quality resampling
    if img.size != target_size:
        img = img.resize(target_size, Image.Resampling.BILINEAR)

    return img


def preprocess_for_mobilenet(
    image_array: np.ndarray
) -> np.ndarray:
    """
    Normalizes image array pixels to the range [-1.0, 1.0] matching
    tf.keras.applications.mobilenet_v2.preprocess_input.
    Formula: (x / 127.5) - 1.0
    """
    x = image_array.astype(np.float32)
    return (x / 127.5) - 1.0


def preprocess_input_image(
    image_source: Union[str, Path, io.BytesIO, bytes, Image.Image],
    target_size: Tuple[int, int] = IMAGE_SIZE,
    expand_dims: bool = True
) -> np.ndarray:
    """
    End-to-end preprocessing pipeline for inference:
    1. Validates and loads image as RGB.
    2. Resizes to (224, 224).
    3. Normalizes using MobileNetV2 standard formula: [-1, 1].
    4. Optionally adds batch dimension (1, 224, 224, 3).

    Raises:
        ImageValidationError if image is invalid or unreadable.
    """
    # If not a PIL Image, validate first
    if not isinstance(image_source, Image.Image):
        is_valid, err = validate_image_file(image_source)
        if not is_valid:
            raise ImageValidationError(err)

    try:
        pil_img = load_and_prepare_image(image_source, target_size=target_size)
        arr = np.array(pil_img, dtype=np.float32)
        norm_arr = preprocess_for_mobilenet(arr)

        if expand_dims:
            norm_arr = np.expand_dims(norm_arr, axis=0)

        return norm_arr
    except Exception as exc:
        raise ImageValidationError(f"Error during image preprocessing: {str(exc)}") from exc
