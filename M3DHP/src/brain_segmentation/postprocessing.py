"""Tumour-mask extraction from segmented MRI slices."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from .config import M3DHPConfig


def rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 image to grayscale uint8."""
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    return np.clip(gray, 0, 255).astype(np.uint8)


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a binary mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if num_labels <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return labels == largest


def estimate_brain_mask(image: np.ndarray) -> np.ndarray:
    """Estimate the brain region so background and borders are ignored."""
    gray = rgb_to_gray(image)
    if int(gray.max()) == 0:
        return np.zeros(gray.shape, dtype=bool)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = otsu > 0
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = largest_component(mask > 0)
    return binary_fill_holes(mask).astype(bool)


def clean_tumor_mask(mask: np.ndarray, brain_mask: np.ndarray, config: M3DHPConfig) -> np.ndarray:
    """Remove tiny specks and implausibly huge tumour candidates."""
    brain_area = max(int(brain_mask.sum()), 1)
    min_area = max(int(config.min_tumor_area_fraction * brain_area), 8)
    max_area = max(int(config.max_tumor_area_fraction * brain_area), min_area)

    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    clean = binary_fill_holes(clean > 0)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(clean.astype(np.uint8), 8)
    kept = np.zeros(mask.shape, dtype=bool)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            kept |= labels == label
    return kept & brain_mask


def fallback_bright_mask(
    image: np.ndarray,
    brain_mask: np.ndarray,
    config: M3DHPConfig,
) -> np.ndarray:
    """Fallback tumour candidate from the brightest tissue inside the brain."""
    gray = rgb_to_gray(image)
    brain_values = gray[brain_mask]
    if brain_values.size == 0:
        return np.zeros(gray.shape, dtype=bool)
    threshold = float(np.percentile(brain_values, 96))
    return clean_tumor_mask((gray >= threshold) & brain_mask, brain_mask, config)


def extract_tumor_mask(
    image: np.ndarray,
    label_map: np.ndarray,
    cluster_centers_rgb: np.ndarray,
    brain_mask: np.ndarray,
    label_hint: str,
    config: M3DHPConfig,
) -> np.ndarray:
    """Estimate a tumour mask from the paper's segmented clusters.

    M3DHP gives image segments, not a clinical tumour detector. This function
    converts those segments into a visible tumour mark by selecting bright,
    compact cluster regions inside the brain. If the source folder is "no",
    the mask is intentionally left empty by default.
    """
    if label_hint == "no" and config.mark_no_label_as_empty:
        return np.zeros(label_map.shape, dtype=bool)

    if not brain_mask.any():
        return np.zeros(label_map.shape, dtype=bool)

    gray = rgb_to_gray(image)
    brain_values = gray[brain_mask]
    brightness_cutoff = max(
        float(np.percentile(brain_values, 82)),
        float(brain_values.mean() + 0.35 * brain_values.std()),
    )

    brain_area = max(int(brain_mask.sum()), 1)
    min_area = max(int(config.min_tumor_area_fraction * brain_area), 8)
    max_area = max(int(config.max_tumor_area_fraction * brain_area), min_area)

    candidate = np.zeros(label_map.shape, dtype=bool)
    for cluster_id in range(len(cluster_centers_rgb)):
        cluster_mask = (label_map == cluster_id) & brain_mask
        area = int(cluster_mask.sum())
        if area < min_area or area > max_area:
            continue
        cluster_gray = gray[cluster_mask]
        if cluster_gray.size == 0:
            continue
        mean_intensity = float(cluster_gray.mean())
        p90_intensity = float(np.percentile(cluster_gray, 90))
        if mean_intensity >= brightness_cutoff or p90_intensity >= np.percentile(brain_values, 93):
            candidate |= cluster_mask

    cleaned = clean_tumor_mask(candidate, brain_mask, config)
    if cleaned.any():
        return cleaned
    return fallback_bright_mask(image, brain_mask, config)


def tumor_area_fraction(mask: np.ndarray, brain_mask: Optional[np.ndarray] = None) -> float:
    """Return tumour-mask area as a fraction of brain area."""
    denominator = int(brain_mask.sum()) if brain_mask is not None and brain_mask.any() else mask.size
    return float(mask.sum()) / max(denominator, 1)
