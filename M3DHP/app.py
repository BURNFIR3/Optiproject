"""Flask web server for the Brain MRI M3DHP segmentation portal."""

from __future__ import annotations

import base64
import io
import sys
import time
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brain_segmentation import M3DHPConfig, run_m3dhp_pipeline  # noqa: E402
from brain_segmentation.postprocessing import tumor_area_fraction  # noqa: E402
from brain_segmentation.metrics import compute_internal_metrics  # noqa: E402
from brain_segmentation.visualization import render_3d_histogram_visualization  # noqa: E402

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "templates"),
    static_folder=str(PROJECT_ROOT / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload


def numpy_to_base64_png(array: np.ndarray, mode: str = "RGB") -> str:
    """Convert a numpy image array to a base64-encoded PNG string."""
    img = Image.fromarray(array.astype(np.uint8), mode=mode)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.route("/")
def index():
    return render_template("segmentation.html")


@app.route("/api/segment", methods=["POST"])
def segment():
    """Accept an MRI image upload and return segmentation results."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    preset = request.form.get("preset", "balanced")
    label_hint = request.form.get("label_hint", "unknown")

    try:
        # Load the image
        pil_image = Image.open(file.stream).convert("RGB")
        image_array = np.asarray(pil_image, dtype=np.uint8)

        # Run the M3DHP pipeline
        config = M3DHPConfig.from_preset(preset)
        start_time = time.time()
        result = run_m3dhp_pipeline(image_array, label_hint=label_hint, config=config)
        elapsed = time.time() - start_time

        # Compute stats
        frac = tumor_area_fraction(result.tumor_mask, result.brain_mask)
        h, w = image_array.shape[:2]

        # Convert mask to visual
        mask_visual = (result.tumor_mask.astype(np.uint8) * 255)

        # Compute internal metrics
        internal_metrics = compute_internal_metrics(result.original, result.label_map)

        # Render the raw/smoothed 3D RGB histogram and dominant M3DHP peaks.
        histogram_visual = render_3d_histogram_visualization(
            result.histogram,
            result.smoothed_histogram,
            result.dominant_peaks,
            config.hist_bins,
        )

        response = {
            "original": numpy_to_base64_png(result.original),
            "preprocessed": numpy_to_base64_png(result.preprocessed.clip(0, 255).astype(np.uint8)),
            "clustered": numpy_to_base64_png(result.clustered),
            "tumor_overlay": numpy_to_base64_png(result.tumor_overlay),
            "tumor_mask": numpy_to_base64_png(mask_visual, mode="L"),
            "brain_mask": numpy_to_base64_png(result.brain_mask.astype(np.uint8) * 255, mode="L"),
            "histogram_3d": numpy_to_base64_png(histogram_visual),
            "stats": {
                "width": w,
                "height": h,
                "clusters": result.n_clusters,
                "tumor_pixels": int(result.tumor_mask.sum()),
                "tumor_area_fraction": round(frac, 6),
                "processing_time": round(elapsed, 2),
                "preset": preset,
                "hist_bins": config.hist_bins,
                "histogram_occupied_bins": int(np.count_nonzero(result.histogram)),
            },
            "metrics": {
                "F": round(internal_metrics["F"], 6),
                "F_prime": round(internal_metrics["F'"], 6),
                "Q": round(internal_metrics["Q"], 6),
            }
        }
        return jsonify(response)

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)


