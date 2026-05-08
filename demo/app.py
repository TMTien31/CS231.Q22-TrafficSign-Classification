"""
Traffic Sign Classifier — FastAPI backend
"""

from __future__ import annotations

import base64
import math
import sys
import uuid
from pathlib import Path
from typing import Any, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, File, Request, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from svm.config import CLASSES, IMG_SIZE, MODEL_CANDIDATES
from svm.svm_features_extraction.hog import extract_hog, resize_and_gray


# ---------------------------------------------------------------------------
# NumPy compatibility shim (for models saved with older NumPy)
# ---------------------------------------------------------------------------

def _ensure_numpy_compat() -> None:
    import numpy as np

    if hasattr(np, "_core"):
        return

    import numpy.core
    import numpy.core.multiarray
    import numpy.core.numeric

    sys.modules["numpy._core"] = numpy.core
    sys.modules["numpy._core.multiarray"] = numpy.core.multiarray
    sys.modules["numpy._core.numeric"] = numpy.core.numeric


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def _load_model():
    _ensure_numpy_compat()
    for path in MODEL_CANDIDATES:
        if Path(path).exists():
            print(f"[model] Loaded from: {path}")
            return joblib.load(path), path
    print("[model] WARNING: No model file found. Place a model in svm/models/ or svm/svm_models/.")
    return None, None


# ---------------------------------------------------------------------------
# HSV color thresholds
# Red wraps around 0/180 in HSV, so two ranges are required.
# ---------------------------------------------------------------------------
RED_LOWER_1 = np.array([0,   70,  50])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([160, 70,  50])
RED_UPPER_2 = np.array([180, 255, 255])

BLUE_LOWER   = np.array([85,  60,  60])   # widened slightly for darker blues
BLUE_UPPER   = np.array([135, 255, 255])

YELLOW_LOWER = np.array([15,  80,  80])
YELLOW_UPPER = np.array([35,  255, 255])

ORANGE_LOWER = np.array([5,   100, 80])
ORANGE_UPPER = np.array([20,  255, 255])  # widened upper hue for orange-red

# White mask for signs with white backgrounds (Hiệu lệnh, etc.)
# High brightness, low saturation
WHITE_LOWER = np.array([0,   0,   180])
WHITE_UPPER = np.array([180, 40,  255])

# ---------------------------------------------------------------------------
# Contour / ROI filter parameters
# ---------------------------------------------------------------------------
MIN_AREA_RATIO = 0.003   # contour must cover at least 0.1 % of image area
MAX_AREA_RATIO = 0.60    # contour must not cover more than 60 % of image area
MIN_ASPECT     = 0.4     # minimum width / height ratio
MAX_ASPECT     = 2.5     # maximum width / height ratio
PADDING_RATIO  = 0.10    # padding added around each cropped ROI
MAX_ROIS       = 5       # maximum number of ROIs forwarded to the classifier
NMS_THRESHOLD  = 0.3     # IoU threshold for non-maximum suppression


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _encode_bgr_to_base64(image_bgr: np.ndarray) -> str:
    """JPEG-encode a BGR image and return a data-URI string."""
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def _decode_image_bytes(raw_bytes: bytes) -> Optional[np.ndarray]:
    if not raw_bytes:
        return None
    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Stage 2 - Color masking
# ---------------------------------------------------------------------------

def create_color_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Build a binary mask that highlights pixels whose color falls within the
    HSV ranges associated with traffic signs (red, blue, yellow, orange).
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1),
        cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2),
    )
    mask_blue   = cv2.inRange(hsv, BLUE_LOWER,   BLUE_UPPER)
    mask_yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    mask_orange = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)
    mask_white  = cv2.inRange(hsv, WHITE_LOWER,  WHITE_UPPER)

    combined = cv2.bitwise_or(mask_red,  mask_blue)
    combined = cv2.bitwise_or(combined,  mask_yellow)
    combined = cv2.bitwise_or(combined,  mask_orange)
    combined = cv2.bitwise_or(combined,  mask_white)
    return combined


# ---------------------------------------------------------------------------
# Stage 3 - Mask cleaning (morphological operations)
# ---------------------------------------------------------------------------

def clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Remove noise and fill gaps in the binary mask using:
      Gaussian blur -> threshold -> morphological open -> close -> dilate
    """
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    _, blurred = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    kernel_open   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    opened  = cv2.morphologyEx(blurred, cv2.MORPH_OPEN,  kernel_open)
    closed  = cv2.morphologyEx(opened,  cv2.MORPH_CLOSE, kernel_close)
    dilated = cv2.dilate(closed, kernel_dilate, iterations=2)
    return dilated


# ---------------------------------------------------------------------------
# Stage 4a - Contour detection
# ---------------------------------------------------------------------------

def find_candidate_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Return external contours found in the cleaned mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


# ---------------------------------------------------------------------------
# Stage 4b - Shape-based contour filtering
# ---------------------------------------------------------------------------

def filter_contours_by_shape(
    contours: list[np.ndarray],
    image_shape: tuple[int, ...],
) -> list[tuple[int, int, int, int]]:
    """
    Keep contours that pass area, aspect ratio, and geometric shape tests.
    Accepted shapes: circle, triangle, rectangle, and convex polygons.
    Returns bounding boxes sorted by area (largest first).
    """
    img_h, img_w = image_shape[:2]
    img_area = img_h * img_w
    min_area = img_area * MIN_AREA_RATIO
    max_area = img_area * MAX_AREA_RATIO

    valid: list[tuple[int, int, int, int, float]] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h <= 0:
            continue

        aspect_ratio = float(w) / h
        if not (MIN_ASPECT <= aspect_ratio <= MAX_ASPECT):
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue

        circularity = (4 * math.pi * area) / (perimeter ** 2)
        epsilon     = 0.04 * perimeter
        approx      = cv2.approxPolyDP(cnt, epsilon, True)
        n_vertices  = len(approx)

        is_circle  = circularity > 0.6
        is_triangle = n_vertices == 3
        is_rect    = n_vertices == 4
        is_polygon = n_vertices >= 5 and circularity > 0.4

        if is_circle or is_triangle or is_rect or is_polygon:
            valid.append((x, y, w, h, area))

    valid.sort(key=lambda b: b[4], reverse=True)
    return [(x, y, w, h) for x, y, w, h, _ in valid]


# ---------------------------------------------------------------------------
# Non-maximum suppression
# ---------------------------------------------------------------------------

def _iou(
    box1: tuple[int, int, int, int],
    box2: tuple[int, int, int, int],
) -> float:
    """Intersection-over-Union for two (x, y, w, h) bounding boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ix = max(x1, x2)
    iy = max(y1, y2)
    iw = min(x1 + w1, x2 + w2) - ix
    ih = min(y1 + h1, y2 + h2) - iy
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def apply_nms(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """
    Greedy NMS: suppress boxes that overlap more than NMS_THRESHOLD with
    an already-kept box. Returns at most MAX_ROIS boxes.
    """
    suppressed = [False] * len(boxes)
    kept: list[tuple[int, int, int, int]] = []

    for i, box_i in enumerate(boxes):
        if suppressed[i]:
            continue
        kept.append(box_i)
        for j in range(i + 1, len(boxes)):
            if not suppressed[j] and _iou(box_i, boxes[j]) > NMS_THRESHOLD:
                suppressed[j] = True

    return kept[:MAX_ROIS]


# ---------------------------------------------------------------------------
# Stage 5 - ROI cropping
# ---------------------------------------------------------------------------

def _center_crop(image_bgr: np.ndarray, ratio: float = 0.65) -> np.ndarray:
    """
    Crop the center region of the image (default 65% of each dimension).
    Useful as a smarter fallback when color masking fails — most demo images
    have the sign near the center.
    """
    h, w = image_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    half_w = int(w * ratio / 2)
    half_h = int(h * ratio / 2)
    x0 = max(0, cx - half_w)
    y0 = max(0, cy - half_h)
    x1 = min(w, cx + half_w)
    y1 = min(h, cy + half_h)
    return image_bgr[y0:y1, x0:x1]


def crop_rois(
    image_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """
    Crop each bounding box from the image with a small padding margin.

    Improvements over the original:
    - Boxes are sorted by proximity to image center (nearest first) so that
      when there are multiple candidates the most likely sign is ranked first.
    - Square-ify: expand the shorter side to match the longer side before
      padding, giving the SVM+HOG pipeline a more consistent aspect ratio.
    Returns (roi_bgr, padded_bbox) pairs; empty crops are skipped.
    """
    img_h, img_w = image_bgr.shape[:2]
    cx_img, cy_img = img_w / 2, img_h / 2

    # Sort boxes by distance from image center (ascending)
    def _dist_to_center(box: tuple[int, int, int, int]) -> float:
        x, y, w, h = box
        return ((x + w / 2 - cx_img) ** 2 + (y + h / 2 - cy_img) ** 2) ** 0.5

    boxes_sorted = sorted(boxes, key=_dist_to_center)

    rois = []
    for x, y, w, h in boxes_sorted:
        # Square-ify: make the crop region square around the box center
        side = max(w, h)
        cx = x + w // 2
        cy = y + h // 2
        half = side // 2 + 1

        pad = int(side * PADDING_RATIO)
        x0 = max(0, cx - half - pad)
        y0 = max(0, cy - half - pad)
        x1 = min(img_w, cx + half + pad)
        y1 = min(img_h, cy + half + pad)

        roi = image_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        rois.append((roi, (x0, y0, x1 - x0, y1 - y0)))

    return rois


# ---------------------------------------------------------------------------
# Orchestrator: stages 2-5
# ---------------------------------------------------------------------------

def detect_sign_rois(
    image_bgr: np.ndarray,
) -> tuple[
    list[tuple[np.ndarray, tuple[int, int, int, int]]],
    str,
    str,
    bool,
]:
    """
    Run color masking -> clean -> contour filter -> NMS -> crop ROIs.

    Returns:
        rois_with_boxes  : list of (roi_bgr, bbox) pairs
        raw_mask_b64     : base64 data-URI of the raw HSV mask
        clean_mask_b64   : base64 data-URI of the cleaned mask
        fallback         : True when no ROI was found (full image used instead)
    """
    raw_mask  = create_color_mask(image_bgr)
    clean     = clean_mask(raw_mask)

    raw_mask_b64   = _encode_bgr_to_base64(cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR))
    clean_mask_b64 = _encode_bgr_to_base64(cv2.cvtColor(clean,    cv2.COLOR_GRAY2BGR))

    contours = find_candidate_contours(clean)
    boxes    = filter_contours_by_shape(contours, image_bgr.shape)
    boxes    = apply_nms(boxes)

    fallback = False
    if not boxes:
        fallback = True
        # Smarter fallback: use the center 65% of the image instead of the
        # full frame, since demo images typically have the sign centered.
        h, w = image_bgr.shape[:2]
        margin_x = int(w * 0.175)
        margin_y = int(h * 0.175)
        boxes = [(margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)]

    rois_with_boxes = crop_rois(image_bgr, boxes)
    return rois_with_boxes, raw_mask_b64, clean_mask_b64, fallback


# ---------------------------------------------------------------------------
# Stage 9 - SVM prediction
# ---------------------------------------------------------------------------

def predict_label(feature: np.ndarray, model: Any) -> Tuple[str, Optional[float]]:
    """
    Run the SVM on a single HOG feature vector.

    Returns:
        label : predicted class name
        score : confidence as a percentage (predict_proba) or decision score,
                or None if unavailable
    """
    feat = feature.reshape(1, -1)
    raw_pred = model.predict(feat)[0]

    try:
        label = CLASSES[int(raw_pred)]
    except (ValueError, IndexError, TypeError):
        label = str(raw_pred)

    score: Optional[float] = None

    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(feat)[0]
            score = round(float(np.max(proba)) * 100, 2)
        except Exception:
            pass
    elif hasattr(model, "decision_function"):
        try:
            df = model.decision_function(feat)[0]
            score = round(float(np.max(df) if isinstance(df, np.ndarray) else df), 3)
        except Exception:
            pass

    return label, score


# ---------------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------------

_BOX_COLORS = [
    (0,   200, 0),
    (0,   120, 255),
    (255, 60,  0),
    (200, 0,   200),
    (0,   200, 200),
]


def draw_predictions(
    image_bgr: np.ndarray,
    results: list[dict[str, Any]],
) -> np.ndarray:
    """Overlay bounding boxes and predicted labels onto the image."""
    out = image_bgr.copy()

    for i, result in enumerate(results):
        x, y, w, h = result["bbox"]
        color = _BOX_COLORS[i % len(_BOX_COLORS)]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        label_text = result["label"]
        if result["score"] is not None:
            label_text += f" {result['score']}%"

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y - 8, th + 4)
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 4, ty + 2), color, cv2.FILLED)
        cv2.putText(
            out, label_text, (x + 2, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    return out


# ---------------------------------------------------------------------------
# Top-level inference entry point
# ---------------------------------------------------------------------------

def run_inference(image_bgr: np.ndarray, model: Any, use_masking: bool = True) -> dict[str, Any]:
    if use_masking:
        rois_with_boxes, raw_mask_b64, clean_mask_b64, fallback = detect_sign_rois(image_bgr)
    else:
        h, w = image_bgr.shape[:2]
        # Use full image
        rois_with_boxes = [(image_bgr, (0, 0, w, h))]
        raw_mask_b64 = ""
        clean_mask_b64 = ""
        fallback = False

    roi_results: list[dict[str, Any]] = []

    for roi_crop, bbox in rois_with_boxes:
        img_resized, img_gray = resize_and_gray(roi_crop, IMG_SIZE)
        feature, hog_vis = extract_hog(img_gray, visualize=True)
        label, score = predict_label(feature, model)

        roi_results.append(
            {
                "bbox":         bbox,
                "label":        label,
                "score":        score,
                "crop_b64":     _encode_bgr_to_base64(roi_crop),
                "resized_b64":  _encode_bgr_to_base64(img_resized),
                "grayscale_b64": _encode_bgr_to_base64(
                    cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                ),
                "hog_b64":      _encode_bgr_to_base64(hog_vis),
            }
        )

    boxed_img = draw_predictions(image_bgr, roi_results)

    return {
        "original_b64":   _encode_bgr_to_base64(image_bgr),
        "raw_mask_b64":   raw_mask_b64,
        "clean_mask_b64": clean_mask_b64,
        "boxed_b64":      _encode_bgr_to_base64(boxed_img),
        "roi_results":    roi_results,
        "fallback":       fallback,
    }


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    root_dir = Path(__file__).resolve().parents[1]
    template_dir = Path(__file__).resolve().parent
    static_dir = root_dir / "static"
    upload_dir = static_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    model, _model_path = _load_model()

    app = FastAPI(title="Traffic Sign Classifier", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5500",
            "http://127.0.0.1:5500",
            "http://localhost:5000",
            "http://127.0.0.1:5000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    templates = Jinja2Templates(directory=str(template_dir))

    # ------------------------------------------------------------------ GET /
    @app.get("/", response_class=HTMLResponse)
    async def index_get(request: Request) -> HTMLResponse:
        return templates.TemplateResponse("index.html", {"request": request})

    # -------------------------------------------------------------- POST /api/predict
    @app.post("/api/predict")
    async def predict_api(
        file: Optional[UploadFile] = File(None),
        use_masking: str = Form("true")
    ) -> JSONResponse:
        if file is None or not file.filename:
            return JSONResponse({"error": "No file selected."}, status_code=400)

        raw_bytes = await file.read()
        if not raw_bytes:
            return JSONResponse({"error": "Empty upload."}, status_code=400)

        suffix = Path(file.filename).suffix
        unique_name = f"{uuid.uuid4().hex}{suffix}"
        save_path = upload_dir / unique_name
        save_path.write_bytes(raw_bytes)

        if model is None:
            return JSONResponse(
                {
                    "error": "Model is not loaded. Please check svm/models/ or svm/svm_models/."
                },
                status_code=500,
            )

        img = _decode_image_bytes(raw_bytes)
        if img is None:
            return JSONResponse(
                {
                    "error": "Could not decode image file. Please upload a valid image."
                },
                status_code=400,
            )

        use_masking_bool = use_masking.lower() == "true"
        result = run_inference(img, model, use_masking_bool)
        return JSONResponse(result)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

app = create_app()

if __name__ == "__main__":
    import uvicorn

    print("Server running at: http://localhost:5000")
    uvicorn.run("demo.app:app", host="0.0.0.0", port=5000, reload=False)