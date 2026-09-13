"""K-Means image segmentation with border-based background identification."""

import cv2
import numpy as np

from .config import (
    KMEANS_ATTEMPTS,
    KMEANS_CLUSTERS,
    KMEANS_EPSILON,
    KMEANS_MAX_ITERATIONS,
    RANDOM_STATE,
)


def _border_indices(height: int, width: int) -> np.ndarray:
    """Return flattened indices for the image border."""
    top = np.arange(width)
    bottom = np.arange((height - 1) * width, height * width)
    left = np.arange(0, height * width, width)
    right = np.arange(width - 1, height * width, width)
    return np.unique(np.concatenate([top, bottom, left, right]))


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Remove small noise and fill modest holes using morphology."""
    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )
    if num_labels <= 1:
        return mask_u8 > 0

    # Keep components that are large enough relative to the image.
    min_area = max(100, int(mask_u8.size * 0.001))
    cleaned = np.zeros_like(mask_u8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255

    # If aggressive filtering removed everything, retain the morphology result.
    if not np.any(cleaned):
        cleaned = mask_u8
    return cleaned > 0


def segment_leaf(
    lab_image: np.ndarray,
    k: int = KMEANS_CLUSTERS,
    attempts: int = KMEANS_ATTEMPTS,
) -> dict:
    """
    Segment a leaf using K-Means in LAB space.

    Cluster IDs are not assumed to have semantic meaning. The cluster with
    the strongest presence on the image border is treated as background.
    All remaining clusters are candidate leaf regions.
    """
    if lab_image is None or lab_image.size == 0:
        raise ValueError("LAB image is empty.")

    height, width = lab_image.shape[:2]
    pixels = np.float32(lab_image.reshape(-1, 3))

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        KMEANS_MAX_ITERATIONS,
        KMEANS_EPSILON,
    )

    # OpenCV's kmeans is deterministic enough for a fixed seed while still
    # using K-Means++ initialization as required.
    cv2.setRNGSeed(RANDOM_STATE)
    compactness, labels, centers = cv2.kmeans(
        pixels,
        k,
        None,
        criteria,
        attempts,
        cv2.KMEANS_PP_CENTERS,
    )

    labels_2d = labels.reshape(height, width)
    centers = np.float32(centers)

    border = _border_indices(height, width)
    border_labels = labels[border].ravel()
    border_counts = np.bincount(border_labels, minlength=k)
    background_cluster = int(np.argmax(border_counts))

    # All clusters except the border-dominant background are candidates for leaf.
    leaf_mask = labels_2d != background_cluster
    leaf_mask = _clean_mask(leaf_mask)

    # In difficult images, preserve the dominant non-background region if the
    # morphology stage produces an implausibly tiny mask.
    leaf_fraction = float(np.mean(leaf_mask))
    if leaf_fraction < 0.01 or leaf_fraction > 0.98:
        non_background = [
            idx for idx in range(k) if idx != background_cluster
        ]
        if non_background:
            counts = np.bincount(
                labels.ravel(), minlength=k
            )
            best_leaf_cluster = max(non_background, key=lambda i: counts[i])
            fallback = labels_2d == best_leaf_cluster
            fallback = _clean_mask(fallback)
            if np.any(fallback):
                leaf_mask = fallback

    segmented_lab = centers[labels].reshape(height, width, 3).astype(np.uint8)
    segmented_rgb = cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2RGB)

    masked_leaf = np.zeros_like(segmented_rgb)
    masked_leaf[leaf_mask] = cv2.cvtColor(
        lab_image, cv2.COLOR_LAB2RGB
    )[leaf_mask]

    return {
        "labels": labels_2d,
        "centers": centers,
        "background_cluster": background_cluster,
        "leaf_mask": leaf_mask,
        "segmented_lab": segmented_lab,
        "segmented_rgb": segmented_rgb,
        "masked_leaf": masked_leaf,
        "compactness": float(compactness),
    }
