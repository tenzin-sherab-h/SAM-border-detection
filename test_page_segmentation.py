import cv2
import torch
import numpy as np

from groundingdino.util.inference import load_model, predict, load_image
from segment_anything import sam_model_registry, SamPredictor

# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH = "sample_page.jpg"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_WEIGHTS = "GroundingDINO/weights/groundingdino_swint_ogc.pth"

SAM_CHECKPOINT = "segment-anything/checkpoints/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"

TEXT_PROMPT = "page"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25
BOX_PADDING_RATIO = 0.02

# -----------------------------
# LOAD MODELS
# -----------------------------
print("Loading GroundingDINO...")
dino_model = load_model(DINO_CONFIG, DINO_WEIGHTS)

print("Loading SAM...")
sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# -----------------------------
# LOAD IMAGE
# -----------------------------
image_source, image = load_image(IMAGE_PATH)
h, w, _ = image_source.shape

# -----------------------------
# DINO PREDICTION
# -----------------------------
boxes, logits, phrases = predict(
    model=dino_model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_THRESHOLD,
    text_threshold=TEXT_THRESHOLD,
    device=DEVICE
)

if len(boxes) == 0:
    raise RuntimeError("No page detected")

# -----------------------------
# CONVERT DINO BOX (cx,cy,w,h) → (x0,y0,x1,y1)
# -----------------------------
boxes_np = boxes.cpu().numpy()

pixel_boxes = []
for cx, cy, bw, bh in boxes_np:
    x0 = (cx - bw / 2) * w
    y0 = (cy - bh / 2) * h
    x1 = (cx + bw / 2) * w
    y1 = (cy + bh / 2) * h
    pixel_boxes.append([x0, y0, x1, y1])

pixel_boxes = np.array(pixel_boxes)

# Pick largest box
areas = (pixel_boxes[:, 2] - pixel_boxes[:, 0]) * (
    pixel_boxes[:, 3] - pixel_boxes[:, 1]
)
box = pixel_boxes[np.argmax(areas)]

# -----------------------------
# ADD PADDING
# -----------------------------
pad_x = (box[2] - box[0]) * BOX_PADDING_RATIO
pad_y = (box[3] - box[1]) * BOX_PADDING_RATIO

box = np.array([
    max(0, box[0] - pad_x),
    max(0, box[1] - pad_y),
    min(w, box[2] + pad_x),
    min(h, box[3] + pad_y),
])

# -----------------------------
# SAM SEGMENTATION
# -----------------------------
sam_predictor.set_image(image_source)

masks, scores, _ = sam_predictor.predict(
    box=box,
    multimask_output=True
)

mask = masks[np.argmax(scores)]

# -----------------------------
# POST-PROCESS: STRICT SHRINK-ONLY PAGE MASK
# -----------------------------
mask_uint8 = (mask * 255).astype(np.uint8)

# Keep only the largest connected component
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8)
largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
page_mask = (labels == largest_label).astype(np.uint8) * 255

# Inward erosion to remove background bleed (NO expansion possible)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
page_mask = cv2.erode(page_mask, kernel, iterations=2)

# Tighten geometry with convex hull to remove stray blobs
contours, _ = cv2.findContours(
    page_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
page_cnt = max(contours, key=cv2.contourArea)

hull = cv2.convexHull(page_cnt)
page_mask = np.zeros_like(page_mask)
cv2.fillConvexPoly(page_mask, hull, 255)

# Color gating: dual-path for light vs text-heavy pages
page_mask_clean = cv2.morphologyEx(
    page_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2
)

# Text density estimate on the cleaned mask
gray = cv2.cvtColor(image_source, cv2.COLOR_RGB2GRAY)
page_gray = gray[page_mask_clean.astype(bool)]
text_ratio = np.mean(page_gray < 110) if len(page_gray) else 0.0

if text_ratio > 0.08:
    # --- Text-heavy path (robust core sampling + MAD + brightness gate) ---
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
    # --- Light-text path (original simple mean/std gating) ---
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

# -----------------------------
# VISUALIZE RESULT
# -----------------------------
overlay = image_source.copy()
overlay[refined_mask] = (
    overlay[refined_mask] * 0.5 + np.array([255, 0, 0]) * 0.5
)

cv2.imwrite(
    "mask_overlay.png",
    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
)
cv2.imwrite(
    "mask_binary.png",
    page_mask
)

print("Saved: mask_overlay.png and mask_binary.png")