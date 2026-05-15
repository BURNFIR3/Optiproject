"""Small entry point for the Brain MRI M3DHP segmentation project."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from brain_segmentation import M3DHPConfig, process_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brain MRI segmentation using M3DHP.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data" / "outputs")
    parser.add_argument("--preset", choices=["balanced", "paper"], default="balanced")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images.")
    parser.add_argument("--hist-bins", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = M3DHPConfig.from_preset(args.preset).with_overrides(
        hist_bins=args.hist_bins,
        population_size=args.population_size,
        iterations=args.iterations,
        random_seed=args.seed,
    )

    rows = process_dataset(args.input, args.output, config, limit=args.limit)
    print(f"\nDone. Processed {len(rows)} image(s).")
    print(f"Segmented MRI outputs: {args.output / 'segmented'}")
    print(f"Masks and metadata:    {args.output}")
    print(f"3D histogram JPGs:    {args.output / 'histograms'}")


if __name__ == "__main__":
    main()

