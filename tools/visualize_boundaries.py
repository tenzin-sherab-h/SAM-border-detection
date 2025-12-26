import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def _load_polygon(json_path: Path) -> List[List[float]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    boundary = data.get("boundary") or {}
    if boundary.get("type") != "polygon":
        raise ValueError("Unsupported boundary type")
    points = boundary.get("points")
    if not points:
        raise ValueError("Missing polygon points")
    return points


def _draw_polygon(image_path: Path, polygon: List[List[float]]) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    overlay = image.copy()
    cv2.polylines(overlay, [pts], isClosed=True, color=(0, 0, 255), thickness=3)
    return overlay


def _collect_pairs(job_dir: Path) -> List[Tuple[Path, Path]]:
    input_dir = job_dir / "input"
    borders_dir = job_dir / "borders"
    if not input_dir.exists() or not borders_dir.exists():
        raise FileNotFoundError("job_dir must contain input/ and borders/")

    pairs: List[Tuple[Path, Path]] = []
    for img_path in sorted(input_dir.iterdir()):
        if not img_path.is_file():
            continue
        stem = img_path.stem
        json_path = borders_dir / f"{stem}.json"
        if not json_path.exists():
            continue
        pairs.append((img_path, json_path))
    return pairs


def visualize(job_dir: Path, out_dir: Path, limit: int, sample: int | None, seed: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pairs = _collect_pairs(job_dir)
    if seed is not None:
        random.seed(seed)

    if sample is not None:
        selected = random.sample(all_pairs, k=min(sample, len(all_pairs)))
    else:
        selected = all_pairs[:limit]

    processed = 0
    for img_path, json_path in selected:
        try:
            polygon = _load_polygon(json_path)
            overlay = _draw_polygon(img_path, polygon)
            out_path = out_dir / img_path.name
            cv2.imwrite(str(out_path), overlay)
            processed += 1
        except Exception:
            continue

    print(f"Saved {processed} overlay(s) to {out_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize detected boundaries")
    parser.add_argument("--job", required=True, type=Path, help="Job directory with input/ and borders/")
    parser.add_argument("--out", required=True, type=Path, help="Output directory for overlays")
    parser.add_argument("--limit", type=int, default=5, help="Max number of images to visualize (ignored if --sample is set)")
    parser.add_argument("--sample", type=int, default=None, help="Randomly sample N images to visualize")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    visualize(args.job, args.out, args.limit, args.sample, args.seed)


if __name__ == "__main__":
    main()


