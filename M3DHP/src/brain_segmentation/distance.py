"""Distance metrics from the M3DHP methodology."""

from __future__ import annotations

import numpy as np


def non_euclidean_distance_matrix(
    a: np.ndarray,
    b: np.ndarray,
    normalizer: float,
) -> np.ndarray:
    """Compute the non-Euclidean distance used by the paper.

    Formula form: sum_d(1 - exp(-(x_d - c_d)^2)).
    RGB differences are normalized to keep the exponential numerically useful
    for 8-bit image intensities and quantized histogram coordinates.
    """
    scale = max(float(normalizer), 1.0)
    diff = (a[:, None, :] - b[None, :, :]) / scale
    return np.sum(1.0 - np.exp(-(diff * diff)), axis=2)
