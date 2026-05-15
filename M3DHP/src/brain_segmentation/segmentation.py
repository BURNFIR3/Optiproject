"""Pixel assignment step for final image segmentation."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import M3DHPConfig
from .distance import non_euclidean_distance_matrix
from .histogram import dequantize_peaks


def assign_pixels_to_peaks(
    quantized_image: np.ndarray,
    dominant_peaks: np.ndarray,
    config: M3DHPConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign each pixel to its nearest dominant peak using NED.

    Returns the cluster-colour image, the integer label map, and RGB cluster
    centers. The output cluster image follows the paper: each pixel is replaced
    by the RGB value of its assigned histogram peak.
    """
    height, width, _ = quantized_image.shape
    pixels = quantized_image.reshape(-1, 3).astype(np.float32)
    labels = np.empty(len(pixels), dtype=np.int32)
    chunk_size = max(config.assignment_chunk_size, 1)

    for start in range(0, len(pixels), chunk_size):
        stop = min(start + chunk_size, len(pixels))
        distances = non_euclidean_distance_matrix(
            pixels[start:stop],
            dominant_peaks.astype(np.float32),
            normalizer=config.hist_bins - 1,
        )
        labels[start:stop] = np.argmin(distances, axis=1)

    centers_rgb = dequantize_peaks(dominant_peaks, config.hist_bins)
    cluster_pixels = centers_rgb[labels]
    clustered = cluster_pixels.reshape(height, width, 3).astype(np.uint8)
    label_map = labels.reshape(height, width)
    return clustered, label_map, centers_rgb.astype(np.uint8)
