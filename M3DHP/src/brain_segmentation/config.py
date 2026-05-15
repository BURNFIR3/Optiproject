"""Configuration for the M3DHP brain MRI segmentation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, Tuple


PresetName = Literal["balanced", "paper"]


@dataclass(frozen=True)
class M3DHPConfig:
    """All tunable parameters for the paper-inspired pipeline.

    The paper reports 350 particles, 350 iterations, w_max=0.98,
    w_min=0.30, C1=C2=2.15, and a dominant-peak distance limit of 80
    intensity units. The default "balanced" preset keeps the same pipeline
    while using a smaller RGB histogram for practical dataset processing.
    Use the "paper" preset from main.py for the full reported settings.
    """

    # Step 1: Gaussian preprocessing of the RGB MRI slice.
    image_sigma: float = 0.5
    image_truncate: float = 1.0  # radius 1 at sigma 0.5 gives a 3x3 window

    # Step 2-3: 3D RGB histogram construction and smoothing.
    hist_bins: int = 64
    hist_smooth_sigma: float = 1.2

    # Step 4: multimodal particle swarm optimization.
    population_size: int = 120
    iterations: int = 120
    w_max: float = 0.98
    w_min: float = 0.30
    c1: float = 2.15
    c2: float = 2.15
    alpha: float = 0.5
    random_seed: int = 42

    # Step 5: dominant peak pruning.
    peak_distance_limit: float = 20.0
    min_peak_relative_fitness: float = 0.02
    max_clusters: int = 18

    # Step 6: pixel assignment and memory control.
    assignment_chunk_size: int = 250_000

    # Tumour-marking layer applied after segmentation.
    mark_no_label_as_empty: bool = True
    min_tumor_area_fraction: float = 0.001
    max_tumor_area_fraction: float = 0.35
    overlay_alpha: float = 0.45
    tumor_color: Tuple[int, int, int] = (255, 35, 35)

    @classmethod
    def from_preset(cls, preset: PresetName) -> "M3DHPConfig":
        """Return a ready-to-use configuration preset."""
        if preset == "paper":
            return cls(
                hist_bins=256,
                hist_smooth_sigma=3.0,
                population_size=350,
                iterations=350,
                peak_distance_limit=80.0,
                min_peak_relative_fitness=0.01,
                max_clusters=32,
                assignment_chunk_size=120_000,
            )
        return cls()

    def with_overrides(self, **overrides: object) -> "M3DHPConfig":
        """Create a copy with selected fields overridden."""
        clean = {key: value for key, value in overrides.items() if value is not None}
        return replace(self, **clean)
