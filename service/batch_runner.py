import argparse
import json
from pathlib import Path
from typing import Dict, List

from service.detector import detect_page_boundary

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _iter_images(input_dir: Path) -> List[Path]:
    return sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )


def run_job(input_dir: Path, output_dir: Path) -> Dict[str, int]:
    """
    Iterates over images in input_dir.
    For each image:
      - calls detect_page_boundary
      - writes one JSON file per image to output_dir
      - isolates failures (does not stop batch)
    Returns a summary dict.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    images = _iter_images(input_dir)
    summary = {"processed": 0, "succeeded": 0, "failed": 0}

    for image_path in images:
        summary["processed"] += 1
        stem = image_path.stem
        out_path = output_dir / f"{stem}.json"
        err_path = output_dir / f"{stem}.error.json"

        try:
            result = detect_page_boundary(image_path)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            summary["succeeded"] += 1
        except Exception as exc:  # noqa: BLE001
            error_payload = {"image": image_path.name, "error": str(exc)}
            with err_path.open("w", encoding="utf-8") as f:
                json.dump(error_payload, f, ensure_ascii=False, indent=2)
            summary["failed"] += 1

    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch page boundary detection")
    parser.add_argument(
        "--input", required=True, type=Path, help="Input directory containing images"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output directory for JSON files"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_job(args.input, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


