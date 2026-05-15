"""3D RGB histogram construction and smoothing."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter


def quantize_rgb(image: np.ndarray, bins: int) -> np.ndarray:
    """Map RGB intensities from 0..255 into 0..bins-1 histogram bins."""
    if bins < 2:
        raise ValueError("histogram bins must be at least 2")
    scaled = np.floor(image.astype(np.float32) * bins / 256.0)
    return np.clip(scaled, 0, bins - 1).astype(np.int16)


def dequantize_peaks(peaks: np.ndarray, bins: int) -> np.ndarray:
    """Convert histogram-bin peak positions back to RGB intensity centers."""
    rgb = (peaks.astype(np.float32) + 0.5) * (256.0 / bins) - 0.5
    return np.clip(rgb, 0, 255)


def build_3d_rgb_histogram(image: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build the 3D RGB histogram used as the PSO objective function."""
    quantized = quantize_rgb(image, bins)
    flat = quantized.reshape(-1, 3)
    histogram = np.zeros((bins, bins, bins), dtype=np.float32)
    np.add.at(histogram, (flat[:, 0], flat[:, 1], flat[:, 2]), 1.0)
    return histogram, quantized


def smooth_3d_histogram(histogram: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth the 3D histogram to remove noisy and weak nearby peaks."""
    return gaussian_filter(histogram, sigma=sigma, mode="nearest")
