# Physical Page Border Detection

Local, privacy-preserving pipeline for detecting full physical page boundaries from scanned book / pecha images. Built for batch processing without sending data to external services.

---

## Requirements

- Python 3.8+ (tested with Python 3.10)
- CUDA-capable GPU (optional, CPU mode supported)
- ~10GB disk space for models and dependencies

---

## Overview

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

### 1. Clone the repository with submodules

```bash
git clone --recursive https://github.com/tenzin-sherab-h/SAM-border-detection.git
cd <repo_name>
```

If you've already cloned without `--recursive`, initialize submodules:
```bash
git submodule update --init --recursive
```

### 2. Create virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip  # Ensure pip is up to date
pip install -r requirements.txt
```

### 3. Install submodules

**GroundingDINO:**
```bash
cd GroundingDINO
pip install -e . --no-build-isolation
cd ..
```

**Important:** The `--no-build-isolation` flag is required because GroundingDINO's setup.py imports torch during the build process, and pip's default build isolation doesn't have access to packages installed in your virtual environment.

**Note:** If you have CUDA and encounter compilation errors, ensure `CUDA_HOME` is set:
```bash
export CUDA_HOME=/path/to/cuda  # e.g., /usr/local/cuda
```

**Segment Anything:**
```bash
cd segment-anything
pip install -e .
cd ..
```

### 4. Download model weights

**GroundingDINO:**
```bash
mkdir -p GroundingDINO/weights
cd GroundingDINO/weights
curl -L -o groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../..
```

**Alternative (Linux with wget):**
```bash
mkdir -p GroundingDINO/weights
cd GroundingDINO/weights
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ../..
```

**Segment Anything:**
```bash
mkdir -p segment-anything/checkpoints
cd segment-anything/checkpoints
curl -L -o sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ../..
```

**Alternative (Linux with wget):**
```bash
mkdir -p segment-anything/checkpoints
cd segment-anything/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ../..
```

---

## Quick Start

Test the installation with a single image:

```bash
# Place your test image as sample_page.jpg in the repo root
python test_page_segmentation.py
```

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

## Troubleshooting

**CUDA compilation errors with GroundingDINO:**
- Ensure `CUDA_HOME` environment variable is set correctly
- Check that your CUDA version matches your PyTorch installation
- See [GroundingDINO README](GroundingDINO/README.md) for detailed CUDA setup instructions

**"No module named 'torch'" error when installing GroundingDINO:**
- Ensure you've installed `requirements.txt` first (step 2)
- Use `pip install -e . --no-build-isolation` instead of just `pip install -e .`
- This is required because GroundingDINO's setup.py needs torch available during build

**Import errors:**
- Ensure submodules are initialized: `git submodule update --init --recursive`
- Ensure submodules are installed: `pip install -e .` (with `--no-build-isolation` for GroundingDINO)

**Model not found errors:**
- Verify model weights are downloaded to the correct paths
- Check file sizes: GroundingDINO (~500MB), SAM (~2.4GB)

---

## Notes

- All processing is local; no cloud calls.
- Algorithm thresholds and post-processing match the original script.
- Batch runner writes only JSON; overlays are opt-in via the visualization tool.
