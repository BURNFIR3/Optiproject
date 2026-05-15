from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brain_segmentation import M3DHPConfig, run_m3dhp_pipeline
from brain_segmentation.io import load_rgb_image, discover_images, infer_label_hint
from brain_segmentation.metrics import compute_internal_metrics

def main():
    input_root = PROJECT_ROOT / "data" / "raw"
    config = M3DHPConfig.from_preset("balanced")
    
    images = discover_images(input_root)
    # To keep the runtime reasonable, we'll process the first 15 images
    # If the user wants all 253, we can remove this slice, but for demonstration 15 is fast.
    images = images[:15]
    print(f"Processing {len(images)} images for the demonstration...")
    
    results = []
    
    for i, img_path in enumerate(images):
        try:
            image = load_rgb_image(img_path)
            label_hint = infer_label_hint(img_path)
            
            # Run pipeline
            seg_result = run_m3dhp_pipeline(image, label_hint, config)
            
            # Compute internal metrics
            metrics = compute_internal_metrics(seg_result.original, seg_result.label_map)
            
            results.append({
                "Image": img_path.name,
                "Label": label_hint,
                "F": metrics["F"],
                "F_prime": metrics["F'"],
                "Q": metrics["Q"]
            })
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    # Write output as a markdown artifact
    artifact_path = PROJECT_ROOT / "evaluation_results.md"
    
    with open(artifact_path, "w", encoding="utf-8") as f:
        f.write("# Segmentation Evaluation Metrics\n\n")
        f.write("Below are the internal clustering metrics ($F$, $F'$, and $Q$) for the selected images. **Lower values indicate better segmentation quality** (less over-segmentation and more homogeneous regions).\n\n")
        f.write("| Image | Label | $F$ Metric | $F'$ Metric | $Q$ Metric |\n")
        f.write("|-------|-------|------------|-------------|------------|\n")
        for res in results:
            f.write(f"| {res['Image']} | {res['Label']} | {res['F']:.6f} | {res['F_prime']:.6f} | {res['Q']:.6f} |\n")
            
        if results:
            avg_f = sum(r["F"] for r in results) / len(results)
            avg_f_p = sum(r["F_prime"] for r in results) / len(results)
            avg_q = sum(r["Q"] for r in results) / len(results)
            f.write(f"| **AVERAGE** | | **{avg_f:.6f}** | **{avg_f_p:.6f}** | **{avg_q:.6f}** |\n")

if __name__ == "__main__":
    main()
