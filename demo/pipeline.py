import base64
import math
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from svm.config import CLASSES, IMG_SIZE
from svm.features.hog import extract_hog, resize_and_gray

# HSV thresholds
RED_LOWER_1 = np.array([0, 70, 50])
RED_UPPER_1 = np.array([10, 255, 255])
RED_LOWER_2 = np.array([160, 70, 50])
RED_UPPER_2 = np.array([180, 255, 255])

BLUE_LOWER = np.array([90, 60, 60])
BLUE_UPPER = np.array([130, 255, 255])

YELLOW_LOWER = np.array([15, 80, 80])
YELLOW_UPPER = np.array([35, 255, 255])

ORANGE_LOWER = np.array([5, 100, 80])
ORANGE_UPPER = np.array([15, 255, 255])

# Contour filters
MIN_AREA_RATIO = 0.001
MAX_AREA_RATIO = 0.60
MIN_ASPECT = 0.4
MAX_ASPECT = 2.5
PADDING_RATIO = 0.10
MAX_ROIS = 5
NMS_THRESHOLD = 0.3


def encode_image_to_base64(image_bgr: np.ndarray) -> str:
    ret, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ret:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def create_color_mask(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    mask_red1 = cv2.inRange(hsv, RED_LOWER_1, RED_UPPER_1)
    mask_red2 = cv2.inRange(hsv, RED_LOWER_2, RED_UPPER_2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_blue = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    mask_yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    mask_orange = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)

    combined = cv2.bitwise_or(mask_red, mask_blue)
    combined = cv2.bitwise_or(combined, mask_yellow)
    combined = cv2.bitwise_or(combined, mask_orange)

    return combined


def clean_mask(mask: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    _, blurred = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel_open)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)
    dilated = cv2.dilate(closed, kernel_dilate, iterations=2)

    return dilated


def find_candidate_contours(mask: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_contours_by_shape(contours: List[np.ndarray], image_shape) -> List[Tuple[int, int, int, int]]:
    img_h, img_w = image_shape[:2]
    img_area = img_h * img_w
    min_area = img_area * MIN_AREA_RATIO
    max_area = img_area * MAX_AREA_RATIO

    valid_boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h <= 0:
            continue
        aspect_ratio = float(w) / h
        if aspect_ratio < MIN_ASPECT or aspect_ratio > MAX_ASPECT:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = (4 * math.pi * area) / (perimeter**2)

        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        n_vertices = len(approx)

        is_circle = circularity > 0.6
        is_triangle = n_vertices == 3
        is_rect = n_vertices == 4
        is_polygon = n_vertices >= 5 and circularity > 0.4

        if is_circle or is_triangle or is_rect or is_polygon:
            valid_boxes.append((x, y, w, h, area))

    valid_boxes.sort(key=lambda b: b[4], reverse=True)
    return [(x, y, w, h) for x, y, w, h, _ in valid_boxes]


def iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
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


def merge_or_nms_boxes(boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
    keep = []
    suppressed = [False] * len(boxes)

    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(boxes[i])
        for j in range(i + 1, len(boxes)):
            if not suppressed[j] and iou(boxes[i], boxes[j]) > NMS_THRESHOLD:
                suppressed[j] = True

    return keep[:MAX_ROIS]


def crop_rois(
    image_bgr: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    img_h, img_w = image_bgr.shape[:2]
    rois = []

    for (x, y, w, h) in boxes:
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


def detect_traffic_sign_rois(
    image_bgr: np.ndarray,
) -> Tuple[List[Tuple[np.ndarray, Tuple[int, int, int, int]]], str, str, bool]:
    raw_mask = create_color_mask(image_bgr)
    clean = clean_mask(raw_mask)

    raw_mask_bgr = cv2.cvtColor(raw_mask, cv2.COLOR_GRAY2BGR)
    clean_mask_bgr = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)

    contours = find_candidate_contours(clean)
    boxes = filter_contours_by_shape(contours, image_bgr.shape)
    boxes = merge_or_nms_boxes(boxes)

    fallback = False
    if not boxes:
        fallback = True
        h, w = image_bgr.shape[:2]
        boxes = [(0, 0, w, h)]

    rois_with_boxes = crop_rois(image_bgr, boxes)

    return (
        rois_with_boxes,
        encode_image_to_base64(raw_mask_bgr),
        encode_image_to_base64(clean_mask_bgr),
        fallback,
    )


def predict_label(feature: np.ndarray, model) -> Tuple[str, float]:
    feat = feature.reshape(1, -1)
    pred = model.predict(feat)[0]

    if isinstance(pred, (int, np.integer)):
        label = CLASSES[int(pred)]
    else:
        try:
            label = CLASSES[int(pred)]
        except (ValueError, IndexError):
            label = str(pred)

    score = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(feat)[0]
            score = round(float(np.max(proba)) * 100, 2)
        except Exception:
            pass
    elif hasattr(model, "decision_function"):
        try:
            df = model.decision_function(feat)[0]
            if isinstance(df, np.ndarray):
                score = round(float(np.max(df)), 3)
            else:
                score = round(float(df), 3)
        except Exception:
            pass

    return label, score


def draw_predictions(image_bgr: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
    out = image_bgr.copy()
    colors = [
        (0, 200, 0),
        (0, 120, 255),
        (255, 60, 0),
        (200, 0, 200),
        (0, 200, 200),
    ]

    for i, result in enumerate(results):
        x, y, w, h = result["bbox"]
        color = colors[i % len(colors)]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

        label_text = result["label"]
        if result["score"] is not None:
            label_text += f" {result['score']}%"

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y - 8, th + 4)
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 4, ty + 2), color, -1)
        cv2.putText(
            out,
            label_text,
            (x + 2, ty - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return out


def run_inference(image_bgr: np.ndarray, model) -> Dict[str, Any]:
    rois_with_boxes, raw_mask_b64, clean_mask_b64, fallback = detect_traffic_sign_rois(image_bgr)

    roi_results = []
    for roi_crop, bbox in rois_with_boxes:
        img_resized, img_gray = resize_and_gray(roi_crop, IMG_SIZE)
        feature, hog_bgr = extract_hog(img_gray, visualize=True)
        label, score = predict_label(feature, model)

        roi_results.append(
            {
                "bbox": bbox,
                "label": label,
                "score": score,
                "crop_b64": encode_image_to_base64(roi_crop),
                "resized_b64": encode_image_to_base64(img_resized),
                "grayscale_b64": encode_image_to_base64(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)),
                "hog_b64": encode_image_to_base64(hog_bgr),
            }
        )

    boxed_img = draw_predictions(image_bgr, roi_results)

    return {
        "original_b64": encode_image_to_base64(image_bgr),
        "raw_mask_b64": raw_mask_b64,
        "clean_mask_b64": clean_mask_b64,
        "boxed_b64": encode_image_to_base64(boxed_img),
        "roi_results": roi_results,
        "fallback": fallback,
    }
