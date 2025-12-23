# Physical Page Border Detection

This repository implements a **local, privacy-preserving pipeline** for detecting
full **physical page boundaries** from scanned book / pecha images.

The system is designed to work at scale (tens of thousands of images) without
manual intervention and without sending data to external services.

---

## Overview

The pipeline detects physical page borders in **two stages**:

1. **Coarse localization** using GroundingDINO  
2. **Fine segmentation and cleanup** using SAM + geometry and color constraints  

The final result is a clean binary mask covering the entire physical page,
excluding the scanner bed and background.

---

## Pipeline Summary

1. **GroundingDINO**
   - Prompt: `"page"`
   - Produces a coarse bounding box around page content
   - Box is intentionally kept tight to exclude background

2. **Segment Anything (SAM)**
   - Segments the dominant surface inside the DINO box
   - Largest connected component is retained
   - Minor erosion removes boundary bleed

3. **Geometric Constraint**
   - A convex hull is computed over the SAM mask
   - Enforces a single, solid page surface
   - Removes decorative borders and internal holes

4. **Color + Brightness Gating**
   - Paper color is estimated from interior pixels
   - Dual logic based on text density:
     - Text-heavy pages use robust statistics (median + MAD)
     - Light-text pages use mean/std
   - Background pixels are rejected

---

## Project Structure

```
page-border-detection/
├── test_page_segmentation.py
├── requirements.txt
├── README.md
├── GroundingDINO/          (git submodule)
└── segment-anything/      (git submodule)
```

`GroundingDINO` and `segment-anything` are included as **git submodules**, not
pip-installed packages.

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## External Repositories (Git Submodules)

This project uses the following repositories as **git submodules**:

- GroundingDINO  
  https://github.com/IDEA-Research/GroundingDINO
- Segment Anything  
  https://github.com/facebookresearch/segment-anything

When cloning this repository, initialize submodules:

```bash
git clone --recurse-submodules https://github.com/YOUR_USERNAME/page-border-detection.git
```

If you already cloned without submodules, run:

```bash
git submodule update --init --recursive
```

---

## Model Weights (Manual Download)

### GroundingDINO

Download:
- `groundingdino_swint_ogc.pth`

Place it at:

```
GroundingDINO/weights/groundingdino_swint_ogc.pth
```

### Segment Anything

Download:
- `sam_vit_h_4b8939.pth`

Place it at:

```
segment-anything/checkpoints/sam_vit_h_4b8939.pth
```

---

## Running the Script

```bash
python test_page_segmentation.py
```

Outputs:
- `mask_overlay.png` — visualization
- `mask_binary.png` — final page mask

---

## Notes

- All processing is local
- No cloud services are used
- Designed for automation and batch processing
- Works on text-heavy and light-text pages
