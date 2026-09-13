"""Lightweight tests for the core project components."""

import numpy as np
import cv2

from src.features import extract_features
from src.preprocessing import resize_image, rgb_to_lab
from src.segmentation import segment_leaf


def synthetic_leaf_image() -> np.ndarray:
    """Create a simple synthetic leaf-like RGB image for structural testing."""
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    image[:] = (35, 35, 35)
    cv2.ellipse(image, (150, 150), (90, 130), 20, 0, 360, (60, 150, 70), -1)
    cv2.circle(image, (130, 130), 25, (160, 90, 60), -1)
    return image


def test_preprocessing_shape_and_lab():
    image = synthetic_leaf_image()
    resized = resize_image(image)
    lab = rgb_to_lab(resized)
    assert resized.shape == (256, 256, 3)
    assert lab.shape == (256, 256, 3)


def test_segmentation_returns_mask():
    image = synthetic_leaf_image()
    lab = rgb_to_lab(resize_image(image))
    result = segment_leaf(lab)
    assert result["leaf_mask"].shape == (256, 256)


def test_feature_vector_has_exactly_12_values():
    image = synthetic_leaf_image()
    resized = resize_image(image)
    lab = rgb_to_lab(resized)
    segmentation = segment_leaf(lab)
    vector = extract_features(lab, resized, segmentation["leaf_mask"])
    assert vector.shape == (12,)
    assert np.isfinite(vector).all()
