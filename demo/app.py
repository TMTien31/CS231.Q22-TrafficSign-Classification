"""
Traffic Sign Classifier — FastAPI backend
"""

from __future__ import annotations

import base64
import math
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import joblib
import numpy as np
try:
    import tensorflow as tf
    from tensorflow import keras
except Exception:
    tf = None
    keras = None
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from svm.config import CLASSES, IMG_SIZE, MODEL_CANDIDATES
from svm.svm_features_extraction.hog import extract_hog, resize_and_gray

CNN_MODEL_CANDIDATES = [
    ROOT_DIR / "cnn" / "models" / "cnn_model_final.keras",
]
CNN_IMG_SIZE = 64
CNN_MAX_FEATURES = 12
CNN_FEATURE_COLS = 4


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

def _load_svm_model():
    _ensure_numpy_compat()
    for path in MODEL_CANDIDATES:
        if Path(path).exists():
            print(f"[svm] Loaded from: {path}")
            return joblib.load(path), path
    print("[svm] WARNING: No model file found. Place a model in svm/models/ or svm/svm_models/.")
    return None, None


def _load_cnn_model():
    if keras is None:
        return None, None, "TensorFlow is not available."
    for path in CNN_MODEL_CANDIDATES:
        if Path(path).exists():
            try:
                model = keras.models.load_model(path)
            except Exception as exc:
                return None, path, f"Failed to load CNN model: {exc}"
            print(f"[cnn] Loaded from: {path}")
            return model, path, None
    return None, None, "CNN model not found. Place cnn_model_final.keras in cnn/models/."


# ---------------------------------------------------------------------------
# HSV color thresholds
# Red wraps around 0/180 in HSV, so two ranges are required.
# ---------------------------------------------------------------------------
RED_LOWER_1 = np.array([0,   70,  50])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([160, 70,  50])
RED_UPPER_2 = np.array([180, 255, 255])

BLUE_LOWER   = np.array([90,  60,  60])
BLUE_UPPER   = np.array([130, 255, 255])

YELLOW_LOWER = np.array([15,  80,  80])
YELLOW_UPPER = np.array([35,  255, 255])

ORANGE_LOWER = np.array([5,   100, 80])
ORANGE_UPPER = np.array([15,  255, 255])

# ---------------------------------------------------------------------------
# Contour / ROI filter parameters
# ---------------------------------------------------------------------------
MIN_AREA_RATIO = 0.001   # contour must cover at least 0.1 % of image area
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


def _decode_image_bytes(raw_bytes: bytes) -> np.ndarray | None:
    if not raw_bytes:
        return None
    buf = np.frombuffer(raw_bytes, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _select_conv_layer_names(model: Any, max_layers: int = 3) -> list[str]:
    if tf is None:
        return []
    conv_types = (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)
    candidates: list[str] = []
    for layer in model.submodules:
        if isinstance(layer, conv_types):
            candidates.append(layer.name)

    candidates = _unique_preserve_order(candidates)
    preferred = [name for name in ("conv1", "conv2", "conv3") if name in candidates]
    if preferred:
        return preferred
    return candidates[-max_layers:]


def _feature_map_grid(feature_map: np.ndarray) -> np.ndarray:
    num_channels = feature_map.shape[-1]
    max_channels = min(num_channels, CNN_MAX_FEATURES)
    cols = CNN_FEATURE_COLS
    rows = int(math.ceil(max_channels / cols))

    height, width = feature_map.shape[:2]
    grid = np.zeros((rows * height, cols * width), dtype=np.uint8)

    for i in range(max_channels):
        fmap = feature_map[:, :, i]
        fmap_norm = cv2.normalize(fmap, None, 0, 255, cv2.NORM_MINMAX)
        fmap_norm = fmap_norm.astype(np.uint8)
        row = i // cols
        col = i % cols
        y0 = row * height
        x0 = col * width
        grid[y0:y0 + height, x0:x0 + width] = fmap_norm

    return cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR)


def _make_gradcam_heatmap(
    img_array: np.ndarray,
    model: Any,
    last_conv_layer_name: str,
    pred_index: int | None = None,
) -> tuple[np.ndarray, int]:
    if tf is None or keras is None:
        raise RuntimeError("TensorFlow is not available.")

    grad_model = keras.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), pred_index


def _overlay_heatmap(image_bgr: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heatmap_resized = cv2.resize(heatmap, (image_bgr.shape[1], image_bgr.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(image_bgr, 0.6, heatmap_color, 0.4, 0)


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

    combined = cv2.bitwise_or(mask_red, mask_blue)
    combined = cv2.bitwise_or(combined, mask_yellow)
    combined = cv2.bitwise_or(combined, mask_orange)
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

def crop_rois(
    image_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """
    Crop each bounding box from the image with a small padding margin.
    Returns (roi_bgr, padded_bbox) pairs; empty crops are skipped.
    """
    img_h, img_w = image_bgr.shape[:2]
    rois = []

    for x, y, w, h in boxes:
        pad_x = int(w * PADDING_RATIO)
        pad_y = int(h * PADDING_RATIO)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(img_w, x + w + pad_x)
        y1 = min(img_h, y + h + pad_y)

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
        h, w = image_bgr.shape[:2]
        boxes = [(0, 0, w, h)]

    rois_with_boxes = crop_rois(image_bgr, boxes)
    return rois_with_boxes, raw_mask_b64, clean_mask_b64, fallback


# ---------------------------------------------------------------------------
# Stage 9 - SVM prediction
# ---------------------------------------------------------------------------

def predict_label(feature: np.ndarray, model: Any) -> tuple[str, float | None]:
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

    score: float | None = None

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

def run_svm_inference(
    image_bgr: np.ndarray,
    rois_with_boxes: list[tuple[np.ndarray, tuple[int, int, int, int]]],
    model: Any | None,
    raw_mask_b64: str,
    clean_mask_b64: str,
    fallback: bool,
) -> dict[str, Any]:
    if model is None:
        return {
            "error": "SVM model is not loaded.",
            "original_b64": _encode_bgr_to_base64(image_bgr),
            "raw_mask_b64": raw_mask_b64,
            "clean_mask_b64": clean_mask_b64,
            "boxed_b64": "",
            "roi_results": [],
            "fallback": fallback,
        }

    roi_results: list[dict[str, Any]] = []

    for roi_crop, bbox in rois_with_boxes:
        img_resized, img_gray = resize_and_gray(roi_crop, IMG_SIZE)
        feature, hog_vis = extract_hog(img_gray, visualize=True)
        label, score = predict_label(feature, model)

        roi_results.append(
            {
                "bbox": bbox,
                "label": label,
                "score": score,
                "crop_b64": _encode_bgr_to_base64(roi_crop),
                "resized_b64": _encode_bgr_to_base64(img_resized),
                "grayscale_b64": _encode_bgr_to_base64(
                    cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
                ),
                "hog_b64": _encode_bgr_to_base64(hog_vis),
            }
        )

    boxed_img = draw_predictions(image_bgr, roi_results)

    return {
        "original_b64": _encode_bgr_to_base64(image_bgr),
        "raw_mask_b64": raw_mask_b64,
        "clean_mask_b64": clean_mask_b64,
        "boxed_b64": _encode_bgr_to_base64(boxed_img),
        "roi_results": roi_results,
        "fallback": fallback,
    }


def run_cnn_inference(
    rois_with_boxes: list[tuple[np.ndarray, tuple[int, int, int, int]]],
    model: Any | None,
    feature_extractor: Any | None,
    feature_layer_names: list[str],
    last_conv_layer_name: str | None,
    error_message: str | None,
) -> dict[str, Any]:
    if model is None:
        return {
            "error": error_message or "CNN model is not loaded.",
            "roi_results": [],
            "feature_layers": feature_layer_names,
        }

    roi_results: list[dict[str, Any]] = []

    for roi_crop, _bbox in rois_with_boxes:
        resized_bgr = cv2.resize(roi_crop, (CNN_IMG_SIZE, CNN_IMG_SIZE))
        resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        normalized = resized_rgb.astype("float32") / 255.0
        normalized_bgr = cv2.cvtColor(
            (normalized * 255.0).astype("uint8"),
            cv2.COLOR_RGB2BGR,
        )
        batch = np.expand_dims(resized_rgb.astype("float32"), axis=0)

        probs = model.predict(batch, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        label = CLASSES[pred_idx]
        score = round(float(probs[pred_idx]) * 100, 2)

        feature_maps: list[dict[str, str]] = []
        if feature_extractor is not None and feature_layer_names:
            try:
                feature_outputs = feature_extractor.predict(batch, verbose=0)
                if not isinstance(feature_outputs, list):
                    feature_outputs = [feature_outputs]
                for fmap, layer_name in zip(feature_outputs, feature_layer_names):
                    grid_bgr = _feature_map_grid(fmap[0])
                    feature_maps.append(
                        {
                            "name": layer_name,
                            "grid_b64": _encode_bgr_to_base64(grid_bgr),
                        }
                    )
            except Exception:
                feature_maps = []

        gradcam_b64 = ""
        if last_conv_layer_name:
            try:
                heatmap, _ = _make_gradcam_heatmap(
                    batch, model, last_conv_layer_name, pred_index=pred_idx
                )
                overlay_bgr = _overlay_heatmap(roi_crop, heatmap)
                gradcam_b64 = _encode_bgr_to_base64(overlay_bgr)
            except Exception:
                gradcam_b64 = ""

        roi_results.append(
            {
                "label": label,
                "score": score,
                "crop_b64": _encode_bgr_to_base64(roi_crop),
                "resized_b64": _encode_bgr_to_base64(resized_bgr),
                "normalized_b64": _encode_bgr_to_base64(normalized_bgr),
                "feature_maps": feature_maps,
                "gradcam_b64": gradcam_b64,
            }
        )

    return {
        "roi_results": roi_results,
        "feature_layers": feature_layer_names,
    }


def run_inference(
    image_bgr: np.ndarray,
    svm_model: Any | None,
    cnn_model: Any | None,
    cnn_feature_extractor: Any | None,
    cnn_feature_layers: list[str],
    cnn_last_conv: str | None,
    cnn_error: str | None,
) -> dict[str, Any]:
    rois_with_boxes, raw_mask_b64, clean_mask_b64, fallback = detect_sign_rois(image_bgr)

    cnn_result = run_cnn_inference(
        rois_with_boxes,
        cnn_model,
        cnn_feature_extractor,
        cnn_feature_layers,
        cnn_last_conv,
        cnn_error,
    )
    svm_result = run_svm_inference(
        image_bgr,
        rois_with_boxes,
        svm_model,
        raw_mask_b64,
        clean_mask_b64,
        fallback,
    )

    return {
        "svm": svm_result,
        "cnn": cnn_result,
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

    svm_model, _svm_path = _load_svm_model()
    cnn_model, _cnn_path, cnn_error = _load_cnn_model()

    cnn_feature_layers: list[str] = []
    cnn_feature_extractor = None
    cnn_last_conv = None
    if cnn_model is not None:
        cnn_feature_layers = _select_conv_layer_names(cnn_model)
        if cnn_feature_layers:
            try:
                cnn_feature_extractor = keras.Model(
                    inputs=cnn_model.inputs,
                    outputs=[cnn_model.get_layer(name).output for name in cnn_feature_layers],
                )
                cnn_last_conv = cnn_feature_layers[-1]
            except Exception as exc:
                cnn_error = f"Failed to prepare CNN feature extractor: {exc}"
                cnn_feature_layers = []
                cnn_feature_extractor = None
                cnn_last_conv = None

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
    async def predict_api(file: UploadFile | None = File(None)) -> JSONResponse:
        if file is None or not file.filename:
            return JSONResponse({"error": "No file selected."}, status_code=400)

        raw_bytes = await file.read()
        if not raw_bytes:
            return JSONResponse({"error": "Empty upload."}, status_code=400)

        suffix = Path(file.filename).suffix
        unique_name = f"{uuid.uuid4().hex}{suffix}"
        save_path = upload_dir / unique_name
        save_path.write_bytes(raw_bytes)

        if svm_model is None and cnn_model is None:
            return JSONResponse(
                {
                    "error": "No model is loaded. Please check svm/models/ and cnn/models/."
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

        result = run_inference(
            img,
            svm_model,
            cnn_model,
            cnn_feature_extractor,
            cnn_feature_layers,
            cnn_last_conv,
            cnn_error,
        )
        return JSONResponse(result)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("demo.app:app", host="0.0.0.0", port=5000, reload=False)