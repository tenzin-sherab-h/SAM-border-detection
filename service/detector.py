from pathlib import Path
from typing import Dict, List, Any

import cv2
import numpy as np
import torch
from groundingdino.util.inference import load_model, predict, load_image
from segment_anything import sam_model_registry, SamPredictor

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_WEIGHTS = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

SAM_CHECKPOINT = "segment-anything/checkpoints/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"

TEXT_PROMPT = "page"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
BOX_PADDING_RATIO = 0.02

# -----------------------------
# LOAD MODELS ONCE
# -----------------------------
print("Loading GroundingDINO...")
_dino_model = load_model(DINO_CONFIG, DINO_WEIGHTS)

print("Loading SAM...")
_sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
_sam.to(device=DEVICE)
_sam_predictor = SamPredictor(_sam)


def _box_to_pixel(boxes: torch.Tensor, width: int, height: int) -> np.ndarray:
    boxes_np = boxes.cpu().numpy()
    pixel_boxes: List[List[float]] = []
    for cx, cy, bw, bh in boxes_np:
        x0 = (cx - bw / 2) * width
        y0 = (cy - bh / 2) * height
        x1 = (cx + bw / 2) * width
        y1 = (cy + bh / 2) * height
        pixel_boxes.append([x0, y0, x1, y1])
    return np.array(pixel_boxes)


def _hull_to_polygon(hull: np.ndarray) -> List[List[float]]:
    pts = hull.reshape(-1, 2).astype(float)
    return pts.tolist()


def detect_page_boundary(image_path: Path) -> Dict[str, Any]:
    """
    Runs page boundary detection on a single image.
    Returns a JSON-serializable dict:
      {
        "image": "<filename>",
        "boundary": {
          "type": "polygon",
          "points": [[x1, y1], [x2, y2], ...]
        },
        "confidence": <float>,
        "source": "auto"
      }
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_source, image = load_image(str(image_path))
    h, w, _ = image_source.shape

    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=_dino_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=DEVICE,
        )

        if len(boxes) == 0:
            raise RuntimeError("No page detected")

        pixel_boxes = _box_to_pixel(boxes, w, h)

        areas = (pixel_boxes[:, 2] - pixel_boxes[:, 0]) * (
            pixel_boxes[:, 3] - pixel_boxes[:, 1]
        )
        box = pixel_boxes[np.argmax(areas)]

        pad_x = (box[2] - box[0]) * BOX_PADDING_RATIO
        pad_y = (box[3] - box[1]) * BOX_PADDING_RATIO

        box = np.array(
            [
                max(0, box[0] - pad_x),
                max(0, box[1] - pad_y),
                min(w, box[2] + pad_x),
                min(h, box[3] + pad_y),
            ]
        )

        _sam_predictor.set_image(image_source)
        masks, scores, _ = _sam_predictor.predict(box=box, multimask_output=True)

    mask = masks[np.argmax(scores)]

    mask_uint8 = (mask * 255).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    page_mask = (labels == largest_label).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    page_mask = cv2.erode(page_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(page_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_cnt = max(contours, key=cv2.contourArea)

    hull = cv2.convexHull(page_cnt)
    page_mask = np.zeros_like(page_mask)
    cv2.fillConvexPoly(page_mask, hull, 255)

    page_mask_clean = cv2.morphologyEx(
        page_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
    )

    gray = cv2.cvtColor(image_source, cv2.COLOR_RGB2GRAY)
    page_gray = gray[page_mask_clean.astype(bool)]
    text_ratio = np.mean(page_gray < 110) if len(page_gray) else 0.0

    if text_ratio > 0.08:
        inner_core = cv2.erode(page_mask_clean, np.ones((25, 25), np.uint8), iterations=1)
        if not np.any(inner_core):
            inner_core = cv2.erode(page_mask_clean, np.ones((15, 15), np.uint8), iterations=1)
        if not np.any(inner_core):
            inner_core = page_mask_clean.copy()

        core_pixels = image_source[inner_core.astype(bool)]
        if len(core_pixels) == 0:
            raise RuntimeError("No paper pixels available after core extraction")

        paper_mean = np.median(core_pixels, axis=0)
        paper_mad = np.median(np.abs(core_pixels - paper_mean), axis=0) + 1e-5
        paper_std = 1.4826 * paper_mad

        dist = np.linalg.norm((image_source - paper_mean) / paper_std, axis=2)

        color_mask = (dist < 2.6).astype(np.uint8) * 255
        core_gray = gray[inner_core.astype(bool)]
        gray_thresh = max(0, int(core_gray.mean() - 12)) if len(core_gray) else 0
        bright_mask = (gray >= gray_thresh).astype(np.uint8) * 255

        page_mask = cv2.bitwise_and(page_mask_clean, color_mask)
        page_mask = cv2.bitwise_and(page_mask, bright_mask)
    else:
        paper_pixels = image_source[page_mask_clean.astype(bool)]
        if len(paper_pixels) == 0:
            raise RuntimeError("No paper pixels available after hull refinement")

        paper_mean = paper_pixels.mean(axis=0)
        paper_std = paper_pixels.std(axis=0) + 1e-5

        dist = np.linalg.norm((image_source - paper_mean) / paper_std, axis=2)
        color_mask = (dist < 2.5).astype(np.uint8) * 255

        page_mask = cv2.bitwise_and(page_mask_clean, color_mask)
        page_mask = cv2.morphologyEx(
            page_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=1
        )

    refined_mask = page_mask.astype(bool)

    # Recompute hull on the final refined mask to match original overlay behavior
    final_contours, _ = cv2.findContours(
        (refined_mask.astype(np.uint8) * 255),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    final_cnt = max(final_contours, key=cv2.contourArea)
    final_hull = cv2.convexHull(final_cnt)

    hull_polygon = _hull_to_polygon(final_hull)
    confidence = float(np.max(scores))

    return {
        "image": image_path.name,
        "boundary": {"type": "polygon", "points": hull_polygon},
        "confidence": confidence,
        "source": "auto",
    }


