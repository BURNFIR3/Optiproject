"""Input discovery, file naming, and image save/load helpers."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_rgb_image(path: Path) -> np.ndarray:
    """Load an image as RGB uint8 with shape (height, width, 3)."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    """Save an RGB image array, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB").save(path)


def save_mask(path: Path, mask: np.ndarray) -> None:
    """Save a boolean mask as a black/white PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_u8 = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_u8, mode="L").save(path)


def discover_images(input_root: Path) -> List[Path]:
    """Return all supported image files under input_root in a stable order."""
    files = [
        path
        for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(files, key=lambda p: (p.parent.name.lower(), natural_key(p.name)))


def infer_label_hint(path: Path) -> str:
    """Infer the class hint from folder name; expected values are yes/no/unknown."""
    folder = path.parent.name.lower().strip()
    if folder in {"yes", "no"}:
        return folder
    return "unknown"


def natural_key(text: str) -> List[object]:
    """Sort helper that keeps no2 before no10."""
    pieces = re.split(r"(\d+)", text.lower())
    return [int(piece) if piece.isdigit() else piece for piece in pieces]


def make_output_stem(path: Path) -> str:
    """Create readable output names such as no1_segmented and yes108_mask.

    The dataset uses names like "1 no.jpeg", "no 923.jpg", "No11.jpg",
    and "Y108.jpg". This function turns those into class-first names while
    keeping a sanitized fallback for unusual filenames.
    """
    label = infer_label_hint(path)
    stem = path.stem.lower()
    numbers = re.findall(r"\d+", stem)

    if label in {"yes", "no"} and numbers:
        return f"{label}{numbers[0]}"

    sanitized = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return sanitized or "image"


def uniquify_stems(paths: Iterable[Path]) -> Dict[Path, str]:
    """Map each input file to a unique output stem."""
    used: Dict[str, int] = {}
    result: Dict[Path, str] = {}
    for path in paths:
        base = make_output_stem(path)
        key = f"{infer_label_hint(path)}:{base}"
        count = used.get(key, 0) + 1
        used[key] = count
        result[path] = base if count == 1 else f"{base}_{count}"
    return result


def write_metadata_csv(path: Path, rows: List[dict]) -> None:
    """Write a compact processing report for all images."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "input_path",
        "label_hint",
        "output_stem",
        "width",
        "height",
        "clusters",
        "tumor_pixels",
        "tumor_area_fraction",
        "segmented_path",
        "mask_path",
        "cluster_path",
        "histogram_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

