from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

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
    print(f"Found {len(images)} images.")
    
    results = []
    
    for i, img_path in enumerate(images):
        print(f"[{i+1}/{len(images)}] Processing {img_path.name}...")
        
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
                "F'": metrics["F'"],
                "Q": metrics["Q"]
            })
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    # Display results in a table format
    print("\nEvaluation Metrics Results (Internal Clustering Metrics):")
    print("-" * 80)
    print(f"{'Image':<25} | {'Label':<10} | {'F':<12} | {'F_prime':<12} | {'Q':<12}")
    print("-" * 80)
    for res in results:
        print(f"{res['Image']:<25} | {res['Label']:<10} | {res['F']:<12.6f} | {res['F\'']:<12.6f} | {res['Q']:<12.6f}")
    print("-" * 80)

    # Calculate averages
    if results:
        avg_f = sum(r["F"] for r in results) / len(results)
        avg_f_p = sum(r["F'"] for r in results) / len(results)
        avg_q = sum(r["Q"] for r in results) / len(results)
        print(f"{'AVERAGE':<25} | {'':<10} | {avg_f:<12.6f} | {avg_f_p:<12.6f} | {avg_q:<12.6f}")

if __name__ == "__main__":
    main()
