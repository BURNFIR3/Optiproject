"""Brain MRI segmentation package based on the M3DHP paper pipeline."""

from .config import M3DHPConfig
from .pipeline import process_dataset, process_image_file, run_m3dhp_pipeline

__all__ = [
    "M3DHPConfig",
    "process_dataset",
    "process_image_file",
    "run_m3dhp_pipeline",
]
