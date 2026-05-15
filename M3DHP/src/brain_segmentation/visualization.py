"""Visualization helpers for saved segmentation outputs."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .config import M3DHPConfig


def overlay_tumor_mark(
    image: np.ndarray,
    tumor_mask: np.ndarray,
    config: M3DHPConfig,
) -> np.ndarray:
    """Draw the tumour mask over the original MRI in red."""
    marked = image.astype(np.float32).copy()
    if not tumor_mask.any():
        return image.copy()

    color = np.asarray(config.tumor_color, dtype=np.float32)
    marked[tumor_mask] = (
        (1.0 - config.overlay_alpha) * marked[tumor_mask]
        + config.overlay_alpha * color
    )

    contours, _ = cv2.findContours(
        tumor_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    marked_u8 = np.clip(marked, 0, 255).astype(np.uint8)
    cv2.drawContours(marked_u8, contours, -1, tuple(int(v) for v in config.tumor_color), 2)
    return marked_u8


def render_3d_histogram_visualization(
    histogram: np.ndarray,
    smoothed_histogram: np.ndarray,
    dominant_peaks: np.ndarray,
    bins: int,
    image_size: Tuple[int, int] = (1240, 540),
    max_points_per_panel: int = 2600,
) -> np.ndarray:
    """Render raw and smoothed 3D RGB histograms as a side-by-side projection."""
    width, height = image_size
    gap = 22
    panel_width = (width - gap) // 2
    panel_size = (panel_width, height)

    raw_panel = _render_histogram_panel(
        histogram=histogram,
        dominant_peaks=None,
        bins=bins,
        title="Raw 3D RGB histogram",
        subtitle="Occupied RGB bins before smoothing",
        image_size=panel_size,
        max_points=max_points_per_panel,
    )
    smooth_panel = _render_histogram_panel(
        histogram=smoothed_histogram,
        dominant_peaks=dominant_peaks,
        bins=bins,
        title="Smoothed histogram + M3DHP peaks",
        subtitle="PSO dominant peaks marked with white rings",
        image_size=panel_size,
        max_points=max_points_per_panel,
    )

    canvas = Image.new("RGB", image_size, (10, 11, 20))
    canvas.paste(Image.fromarray(raw_panel), (0, 0))
    canvas.paste(Image.fromarray(smooth_panel), (panel_width + gap, 0))

    draw = ImageDraw.Draw(canvas)
    draw.line(
        [(panel_width + gap // 2, 30), (panel_width + gap // 2, height - 30)],
        fill=(42, 45, 72),
        width=1,
    )
    return np.asarray(canvas, dtype=np.uint8)


def _render_histogram_panel(
    histogram: np.ndarray,
    dominant_peaks: Optional[np.ndarray],
    bins: int,
    title: str,
    subtitle: str,
    image_size: Tuple[int, int],
    max_points: int,
) -> np.ndarray:
    width, height = image_size
    image = Image.new("RGB", image_size, (14, 16, 29))
    draw = ImageDraw.Draw(image, "RGBA")

    plot_bounds = (64, 92, width - 42, height - 58)
    _draw_histogram_cube(draw, bins, plot_bounds)

    coords, weights = _select_histogram_points(histogram, max_points=max_points)
    if len(coords) > 0:
        points, depth = _project_to_panel(coords.astype(np.float32), bins, plot_bounds)
        colors = _bin_centers_to_rgb(coords, bins)
        max_weight = float(weights.max()) if len(weights) else 1.0
        scaled = np.log1p(weights.astype(np.float32)) / max(np.log1p(max_weight), 1e-6)
        order = np.argsort(depth)

        for index in order:
            x, y = points[index]
            radius = int(1 + 4 * scaled[index])
            alpha = int(72 + 172 * scaled[index])
            r, g, b = (int(value) for value in colors[index])
            fill = (max(r, 36), max(g, 36), max(b, 36), alpha)
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=fill,
            )

    if dominant_peaks is not None and len(dominant_peaks) > 0:
        peak_points, peak_depth = _project_to_panel(
            dominant_peaks.astype(np.float32),
            bins,
            plot_bounds,
        )
        for index in np.argsort(peak_depth):
            x, y = peak_points[index]
            radius = 8
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=(255, 255, 255, 235),
                width=2,
            )
            draw.line((x - 5, y, x + 5, y), fill=(255, 255, 255, 210), width=1)
            draw.line((x, y - 5, x, y + 5), fill=(255, 255, 255, 210), width=1)

    occupied = int(np.count_nonzero(histogram))
    visible = min(occupied, max_points)
    max_count = float(histogram.max()) if histogram.size else 0.0

    draw.text((24, 22), title, fill=(232, 234, 246, 255))
    draw.text((24, 45), subtitle, fill=(159, 164, 196, 255))
    draw.text(
        (24, height - 33),
        f"bins: {bins} | occupied: {occupied:,} | shown: {visible:,} | max: {max_count:.1f}",
        fill=(129, 140, 248, 255),
    )
    return np.asarray(image, dtype=np.uint8)


def _select_histogram_points(histogram: np.ndarray, max_points: int) -> Tuple[np.ndarray, np.ndarray]:
    flat = histogram.reshape(-1)
    occupied = np.flatnonzero(flat > 0)
    if len(occupied) == 0:
        return np.empty((0, 3), dtype=np.int32), np.empty((0,), dtype=np.float32)

    if len(occupied) > max_points:
        values = flat[occupied]
        selected_index = np.argpartition(values, -max_points)[-max_points:]
        occupied = occupied[selected_index]

    weights = flat[occupied].astype(np.float32)
    coords = np.column_stack(np.unravel_index(occupied, histogram.shape)).astype(np.int32)
    return coords, weights


def _draw_histogram_cube(
    draw: ImageDraw.ImageDraw,
    bins: int,
    plot_bounds: Tuple[int, int, int, int],
) -> None:
    corners = np.asarray(
        [
            [0, 0, 0],
            [bins - 1, 0, 0],
            [0, bins - 1, 0],
            [0, 0, bins - 1],
            [bins - 1, bins - 1, 0],
            [bins - 1, 0, bins - 1],
            [0, bins - 1, bins - 1],
            [bins - 1, bins - 1, bins - 1],
        ],
        dtype=np.float32,
    )
    projected, _ = _project_to_panel(corners, bins, plot_bounds)
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
    for start, end in edges:
        draw.line(
            (tuple(projected[start]), tuple(projected[end])),
            fill=(72, 77, 116, 180),
            width=1,
        )

    labels = [("R", 1, (248, 113, 113)), ("G", 2, (74, 222, 128)), ("B", 3, (96, 165, 250))]
    for label, index, color in labels:
        x, y = projected[index]
        draw.text((x + 8, y - 8), label, fill=(*color, 255))


def _project_to_panel(
    coords: np.ndarray,
    bins: int,
    plot_bounds: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    left, top, right, bottom = plot_bounds
    corners = np.asarray(
        [
            [0, 0, 0],
            [bins - 1, 0, 0],
            [0, bins - 1, 0],
            [0, 0, bins - 1],
            [bins - 1, bins - 1, 0],
            [bins - 1, 0, bins - 1],
            [0, bins - 1, bins - 1],
            [bins - 1, bins - 1, bins - 1],
        ],
        dtype=np.float32,
    )
    corner_projection, _ = _project_isometric(corners, bins)
    projection, depth = _project_isometric(coords, bins)

    min_xy = corner_projection.min(axis=0)
    max_xy = corner_projection.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)
    scale = min((right - left) / span[0], (bottom - top) / span[1])

    projected = np.empty_like(projection)
    projected[:, 0] = left + (projection[:, 0] - min_xy[0]) * scale
    projected[:, 1] = top + (projection[:, 1] - min_xy[1]) * scale
    return projected, depth


def _project_isometric(coords: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
    normalizer = max(float(bins - 1), 1.0)
    rgb = coords.astype(np.float32) / normalizer
    red = rgb[:, 0]
    green = rgb[:, 1]
    blue = rgb[:, 2]

    x = (red - blue) * 0.82
    y = (red + blue) * 0.36 - green * 0.95
    depth = red + blue + green * 0.7
    return np.column_stack([x, y]), depth


def _bin_centers_to_rgb(coords: np.ndarray, bins: int) -> np.ndarray:
    rgb = (coords.astype(np.float32) + 0.5) * (256.0 / bins) - 0.5
    return np.clip(rgb, 0, 255).astype(np.uint8)
