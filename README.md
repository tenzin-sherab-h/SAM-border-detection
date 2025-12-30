# Physical Page Border Detection

Local, privacy-preserving pipeline for detecting full physical page boundaries from scanned book / pecha images. Built for batch processing without sending data to external services.

---

## Overview (Algorithm is unchanged)

Two-stage detection:
1) **GroundingDINO** for coarse localization (`"page"` prompt)  
2) **SAM + post-processing** for fine segmentation, convex hull geometry, and color/brightness gating with dual text-density paths (median/MAD for text-heavy, mean/std for light-text)

The algorithm matches the original `test_page_segmentation.py`: DINO → SAM → erosion → convex hull → color gating, with the hull taken from the final refined mask.

---

## Project Structure

```
.
├── service/
│   ├── detector.py         # single-image detection core (pure function)
│   ├── batch_runner.py     # directory-based batch runner (sequential)
│   └── __init__.py
│
├── tools/
│   └── visualize_boundaries.py  # optional dev-only polygon overlay tool
│
├── test_jobs/
│   └── job_001/
│       ├── input/          # place test images here
│       └── borders/        # JSON outputs here
│
├── test_page_segmentation.py    # legacy reference script (unchanged)
├── requirements.txt
├── README.md
├── GroundingDINO/          # git submodule
└── segment-anything/       # git submodule
```

`GroundingDINO` and `segment-anything` are git submodules (not pip-installed packages).

---

## Setup

Create a virtual environment and install deps:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Initialize submodules (if not already):
```bash
git submodule update --init --recursive
```

---

## Model Weights (Manual Download)

GroundingDINO:
- `groundingdino_swint_ogc.pth` → `GroundingDINO/weights/groundingdino_swint_ogc.pth`

Segment Anything:
- `sam_vit_h_4b8939.pth` → `segment-anything/checkpoints/sam_vit_h_4b8939.pth`

---

## Single-Image Detection (reference)

Legacy script (unchanged):
```bash
python test_page_segmentation.py
```

---

## Batch Detection (sequential, JSON outputs)

Run over a folder of images (`.jpg/.jpeg/.png/.tif/.tiff`):
```bash
python -m service.batch_runner \
  --input test_jobs/job_001/input \
  --output test_jobs/job_001/borders
```

Behavior:
- Uses `service.detector.detect_page_boundary` (pure function, no file writes).
- Writes `<stem>.json` per image; on failure writes `<stem>.error.json`.
- Sequential only; no multiprocessing/async.

Example JSON:
```json
{
  "image": "sample.jpg",
  "boundary": {
    "type": "polygon",
    "points": [[x1, y1], [x2, y2], ...]
  },
  "confidence": 0.92,
  "source": "auto"
}
```

---

## Optional Visualization (dev-only)

Overlay stored polygons onto images:
```bash
python tools/visualize_boundaries.py \
  --job test_jobs/job_001 \
  --out test_jobs/job_001/overlays \
  --limit 5
```

Random sampling:
```bash
python tools/visualize_boundaries.py \
  --job test_jobs/job_001 \
  --out test_jobs/job_001/overlays \
  --sample 3 \
  --seed 42   # optional for reproducibility
```

Notes:
- Skips missing/invalid JSON gracefully.
- Does not modify source images.

---

## Node Integration (paths to set)

Set two absolute paths when spawning Python from Node:
- `PYTHON_BIN`: your virtualenv Python, e.g. `/absolute/path/to/venv/bin/python`
- `PYTHON_APP`: repo root so imports like `service.batch_runner` resolve, e.g. `/absolute/path/to/samv1`

Then spawn:
```js
spawn(PYTHON_BIN, ["-m", "service.batch_runner", "--input", "...", "--output", "..."], {
  cwd: PYTHON_APP,
  env: { ...process.env, PYTHONPATH: PYTHON_APP },
});
```

The detector returns polygons. If your UI expects boxes, derive a box from the polygon points client-side; otherwise render the polygon directly.

---

## Notes

- All processing is local; no cloud calls.
- Algorithm thresholds and post-processing match the original script.
- Batch runner writes only JSON; overlays are opt-in via the visualization tool.
