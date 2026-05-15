import argparse
from pathlib import Path
import sys

# Add src to path so we can import the pipeline
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brain_segmentation import M3DHPConfig, run_m3dhp_pipeline
from brain_segmentation.io import load_rgb_image, infer_label_hint
from brain_segmentation.metrics import compute_internal_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate a single MRI image.")
    parser.add_argument("image_path", type=str, help="Path to the image (e.g., data/raw/yes/Y1.jpg)")
    args = parser.parse_args()

    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"Error: Could not find image at {img_path}")
        return

    print(f"Loading image: {img_path.name}")
    image = load_rgb_image(img_path)
    label_hint = infer_label_hint(img_path)
    
    print(f"Running M3DHP segmentation pipeline...")
    config = M3DHPConfig.from_preset("balanced")
    seg_result = run_m3dhp_pipeline(image, label_hint, config)
    
    print(f"Computing evaluation metrics...")
    metrics = compute_internal_metrics(seg_result.original, seg_result.label_map)
    
    print("-" * 40)
    print(f"Results for {img_path.name}:")
    print(f"  F Metric  : {metrics['F']:.6f}")
    print(f"  F' Metric : {metrics['F\'']:.6f}")
    print(f"  Q Metric  : {metrics['Q']:.6f}")
    print("-" * 40)

if __name__ == "__main__":
    main()
