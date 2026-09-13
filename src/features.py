"""Handcrafted color-moment and GLCM texture feature extraction."""

import numpy as np
from scipy.stats import skew
from skimage.feature import graycomatrix, graycoprops

from .config import (
    FEATURE_COLUMNS,
    GLCM_ANGLE,
    GLCM_DISTANCE,
    GLCM_LEVELS,
)


def _safe_values(values: np.ndarray) -> np.ndarray:
    """Return finite values or raise if no usable pixels exist."""
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite pixels available for feature extraction.")
    return values


def _safe_skewness(values: np.ndarray) -> float:
    """Calculate skewness while handling constant-valued channels."""
    values = _safe_values(values)
    if np.allclose(values, values[0]):
        return 0.0
    value = float(skew(values, bias=False, nan_policy="omit"))
    return 0.0 if not np.isfinite(value) else value


def color_moment_features(lab_image: np.ndarray, leaf_mask: np.ndarray) -> dict:
    """Extract mean, variance, and skewness for L*, a*, and b*."""
    if lab_image.shape[:2] != leaf_mask.shape:
        raise ValueError("LAB image and leaf mask dimensions do not match.")

    mask = leaf_mask.astype(bool)
    if int(mask.sum()) < 10:
        raise ValueError("Leaf mask contains too few pixels.")

    names = ["L", "a", "b"]
    result = {}
    for channel_index, name in enumerate(names):
        values = _safe_values(lab_image[:, :, channel_index][mask])
        result[f"{name}_mean"] = float(np.mean(values))
        result[f"{name}_variance"] = float(np.var(values))
        result[f"{name}_skewness"] = _safe_skewness(values)
    return result


def _quantize_grayscale(gray: np.ndarray, levels: int) -> np.ndarray:
    """Quantize uint8 grayscale to a smaller integer GLCM range."""
    return np.floor(gray.astype(np.float32) * levels / 256.0).astype(np.uint8)


def glcm_features(rgb_image: np.ndarray, leaf_mask: np.ndarray) -> dict:
    """
    Extract GLCM contrast, correlation, and entropy.

    The GLCM is built from a leaf-focused crop. Pixels outside the leaf are
    replaced with the median leaf intensity to avoid treating the background
    as a disease texture.
    """
    import cv2

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    mask = leaf_mask.astype(bool)
    ys, xs = np.where(mask)
    if len(xs) < 10:
        raise ValueError("Leaf mask contains too few pixels for GLCM.")

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop_gray = gray[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]

    leaf_values = crop_gray[crop_mask]
    fill_value = int(np.median(leaf_values))
    crop_gray[~crop_mask] = fill_value

    quantized = _quantize_grayscale(crop_gray, GLCM_LEVELS)

    glcm = graycomatrix(
        quantized,
        distances=[GLCM_DISTANCE],
        angles=[GLCM_ANGLE],
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    contrast = float(graycoprops(glcm, "contrast")[0, 0])
    correlation = float(graycoprops(glcm, "correlation")[0, 0])

    probabilities = glcm.astype(np.float64)
    entropy = float(
        -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))
    )

    return {
        "contrast": contrast,
        "correlation": correlation,
        "entropy": entropy,
    }


def extract_features(
    lab_image: np.ndarray,
    rgb_image: np.ndarray,
    leaf_mask: np.ndarray,
) -> np.ndarray:
    """Return exactly the 12 features in the project's fixed order."""
    color = color_moment_features(lab_image, leaf_mask)
    texture = glcm_features(rgb_image, leaf_mask)
    values = {**color, **texture}

    vector = np.array([values[name] for name in FEATURE_COLUMNS], dtype=np.float64)
    if vector.shape != (12,):
        raise ValueError(f"Expected 12 features, got shape {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"Non-finite feature vector: {vector}")
    return vector
