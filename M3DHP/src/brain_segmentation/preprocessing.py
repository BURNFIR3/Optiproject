"""Preprocessing functions used before RGB histogram construction."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_preprocess_image(
    image: np.ndarray,
    sigma: float = 0.5,
    truncate: float = 1.0,
) -> np.ndarray:
    """Smooth each RGB channel with the Gaussian filter from the paper.

    The paper states sigma=0.5 and a 3x3 window before calculating the
    3D histogram. Applying the filter channel-wise preserves RGB structure.
    """
    image_f32 = image.astype(np.float32, copy=False)
    return gaussian_filter(
        image_f32,
        sigma=(sigma, sigma, 0.0),
        truncate=truncate,
        mode="nearest",
    )
