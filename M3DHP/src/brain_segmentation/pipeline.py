"""End-to-end orchestration for the M3DHP MRI segmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import M3DHPConfig
from .histogram import build_3d_rgb_histogram, smooth_3d_histogram
from .io import (
    discover_images,
    infer_label_hint,
    load_rgb_image,
    save_mask,
    save_rgb_image,
    uniquify_stems,
    write_metadata_csv,
)
from .postprocessing import estimate_brain_mask, extract_tumor_mask, tumor_area_fraction
from .preprocessing import gaussian_preprocess_image
from .pso import filter_dominant_peaks, multimodal_pso_peaks
from .segmentation import assign_pixels_to_peaks
from .visualization import overlay_tumor_mark, render_3d_histogram_visualization


@dataclass(frozen=True)
class SegmentationResult:
    """All important arrays and metadata from a single MRI run."""

    original: np.ndarray
    preprocessed: np.ndarray
    clustered: np.ndarray
    label_map: np.ndarray
    brain_mask: np.ndarray
    tumor_mask: np.ndarray
    tumor_overlay: np.ndarray
    histogram: np.ndarray
    smoothed_histogram: np.ndarray
    dominant_peaks: np.ndarray
    cluster_centers_rgb: np.ndarray

    @property
    def n_clusters(self) -> int:
        return int(len(self.dominant_peaks))


def run_m3dhp_pipeline(
    image: np.ndarray,
    label_hint: str,
    config: M3DHPConfig,
) -> SegmentationResult:
    """Run every step of the paper-inspired pipeline on one RGB MRI image."""
    preprocessed = gaussian_preprocess_image(
        image,
        sigma=config.image_sigma,
        truncate=config.image_truncate,
    )
    histogram, quantized = build_3d_rgb_histogram(preprocessed, config.hist_bins)
    smoothed_histogram = smooth_3d_histogram(histogram, config.hist_smooth_sigma)
    pso_result = multimodal_pso_peaks(smoothed_histogram, config)
    dominant_peaks = filter_dominant_peaks(pso_result, config)
    clustered, label_map, centers_rgb = assign_pixels_to_peaks(quantized, dominant_peaks, config)

    brain_mask = estimate_brain_mask(image)
    tumor_mask = extract_tumor_mask(
        image=image,
        label_map=label_map,
        cluster_centers_rgb=centers_rgb,
        brain_mask=brain_mask,
        label_hint=label_hint,
        config=config,
    )
    tumor_overlay = overlay_tumor_mark(image, tumor_mask, config)

    return SegmentationResult(
        original=image,
        preprocessed=preprocessed,
        clustered=clustered,
        label_map=label_map,
        brain_mask=brain_mask,
        tumor_mask=tumor_mask,
        tumor_overlay=tumor_overlay,
        histogram=histogram,
        smoothed_histogram=smoothed_histogram,
        dominant_peaks=dominant_peaks,
        cluster_centers_rgb=centers_rgb,
    )


def process_image_file(
    image_path: Path,
    output_root: Path,
    config: M3DHPConfig,
    output_stem: Optional[str] = None,
) -> Dict[str, object]:
    """Run the pipeline for one image and save all requested outputs."""
    label_hint = infer_label_hint(image_path)
    stem = output_stem or image_path.stem
    original = load_rgb_image(image_path)
    result = run_m3dhp_pipeline(original, label_hint=label_hint, config=config)

    label_folder = label_hint if label_hint in {"yes", "no"} else "unknown"
    segmented_path = output_root / "segmented" / label_folder / f"{stem}_segmented.png"
    mask_path = output_root / "masks" / label_folder / f"{stem}_mask.png"
    cluster_path = output_root / "clusters" / label_folder / f"{stem}_clusters.png"
    brain_mask_path = output_root / "brain_masks" / label_folder / f"{stem}_brain_mask.png"
    histogram_path = output_root / "histograms" / label_folder / f"{stem}_histogram.jpg"

    # The user-facing segmented output is the MRI with the tumour marked.
    save_rgb_image(segmented_path, result.tumor_overlay)
    save_mask(mask_path, result.tumor_mask)
    save_rgb_image(cluster_path, result.clustered)
    save_mask(brain_mask_path, result.brain_mask)
    save_rgb_image(
        histogram_path,
        render_3d_histogram_visualization(
            result.histogram,
            result.smoothed_histogram,
            result.dominant_peaks,
            config.hist_bins,
        ),
    )

    height, width = original.shape[:2]
    return {
        "input_path": str(image_path),
        "label_hint": label_hint,
        "output_stem": stem,
        "width": width,
        "height": height,
        "clusters": result.n_clusters,
        "tumor_pixels": int(result.tumor_mask.sum()),
        "tumor_area_fraction": f"{tumor_area_fraction(result.tumor_mask, result.brain_mask):.6f}",
        "segmented_path": str(segmented_path),
        "mask_path": str(mask_path),
        "cluster_path": str(cluster_path),
        "histogram_path": str(histogram_path),
    }


def process_dataset(
    input_root: Path,
    output_root: Path,
    config: M3DHPConfig,
    limit: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Process every image under data/raw and write outputs to data/outputs."""
    images = discover_images(input_root)
    if limit is not None:
        images = images[:limit]

    stems = uniquify_stems(images)
    rows: List[Dict[str, object]] = []
    for index, image_path in enumerate(images, start=1):
        print(f"[{index:03d}/{len(images):03d}] {image_path.name}")
        rows.append(
            process_image_file(
                image_path=image_path,
                output_root=output_root,
                config=config,
                output_stem=stems[image_path],
            )
        )

    write_metadata_csv(output_root / "metadata.csv", rows)
    return rows


