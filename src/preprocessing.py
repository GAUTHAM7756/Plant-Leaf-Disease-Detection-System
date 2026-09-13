"""Image loading, resizing, RGB/LAB conversion, and preprocessing."""

from pathlib import Path

import cv2
import numpy as np

from .config import IMAGE_SIZE


def load_image(image_path: str | Path) -> np.ndarray:
    """Load an image with OpenCV and return it in RGB uint8 format."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        raise ValueError(f"Could not decode image or image is empty: {path}")

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_image(image_rgb: np.ndarray) -> np.ndarray:
    """Resize an RGB image to the project's fixed 256x256 size."""
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Cannot resize an empty image.")
    return cv2.resize(image_rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)


def rgb_to_lab(image_rgb: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 image to OpenCV's 8-bit LAB representation."""
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Cannot convert an empty image.")
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)


def preprocess_image(image_path: str | Path) -> dict:
    """Load, resize, and convert an image to LAB."""
    original_rgb = load_image(image_path)
    resized_rgb = resize_image(original_rgb)
    lab = rgb_to_lab(resized_rgb)
    return {
        "original_rgb": original_rgb,
        "resized_rgb": resized_rgb,
        "lab": lab,
    }
