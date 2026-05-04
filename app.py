import os
import cv2
import joblib
import numpy as np
import sys
import uuid
import base64
import math

# Workaround cho lỗi numpy version mismatch giữa Kaggle (2.x) và local (1.x)
if not hasattr(np, '_core'):
    import numpy.core
    import numpy.core.multiarray
    import numpy.core.numeric
    sys.modules['numpy._core'] = numpy.core
    sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
    sys.modules['numpy._core.numeric'] = numpy.core.numeric

from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from skimage.feature import hog
from skimage import exposure

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ─── Thông số model (giữ nguyên như lúc train) ───────────────────────────────
IMG_SIZE = (64, 64)
CLASSES = ["Cam", "Chidan", "Hieulenh", "Nguyhiem", "Phu"]

# ─── Load model ──────────────────────────────────────────────────────────────
if os.path.exists(os.path.join("models", "svm_hog_tuned_model.joblib")):
    MODEL_PATH = os.path.join("models", "svm_hog_tuned_model.joblib")
else:
    MODEL_PATH = os.path.join("models", "svm_hog_model.joblib")

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
else:
    model = None
    print(f"Model file not found at {MODEL_PATH}!")

# ─── Ngưỡng HSV (dễ chỉnh) ───────────────────────────────────────────────────
# Đỏ (hai dải vì hue đỏ nằm ở cả đầu và cuối vòng HSV)
RED_LOWER_1  = np.array([0,   70,  50])
RED_UPPER_1  = np.array([10, 255, 255])
RED_LOWER_2  = np.array([160, 70,  50])
RED_UPPER_2  = np.array([180,255, 255])

# Xanh dương (biển hiệu lệnh)
BLUE_LOWER   = np.array([90,  60,  60])
BLUE_UPPER   = np.array([130,255, 255])

# Vàng/cam (biển nguy hiểm, cảnh báo)
YELLOW_LOWER = np.array([15,  80,  80])
YELLOW_UPPER = np.array([35, 255, 255])

ORANGE_LOWER = np.array([5,   100,  80])
ORANGE_UPPER = np.array([15,  255, 255])

# ─── Ngưỡng lọc contour ──────────────────────────────────────────────────────
MIN_AREA_RATIO = 0.001   # tối thiểu 0.1% diện tích ảnh
MAX_AREA_RATIO = 0.60    # tối đa 60% diện tích ảnh
MIN_ASPECT     = 0.4
MAX_ASPECT     = 2.5
PADDING_RATIO  = 0.10    # padding 10% quanh bounding box
MAX_ROIS       = 5       # tối đa bao nhiêu ROI trả về
NMS_THRESHOLD  = 0.3     # IoU threshold cho Non-Max Suppression


# ─── Hàm tiện ích ────────────────────────────────────────────────────────────

def encode_image_to_base64(image_bgr):
    """Chuyển ảnh BGR (numpy array) sang chuỗi base64 để nhúng vào HTML."""
    ret, buf = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ret:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode('utf-8')


def create_color_mask(image):
    """
    Tạo mask tổng hợp từ các màu đặc trưng của biển báo giao thông.
    Trả về: combined_mask (uint8, 0/255)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask_red1   = cv2.inRange(hsv, RED_LOWER_1,   RED_UPPER_1)
    mask_red2   = cv2.inRange(hsv, RED_LOWER_2,   RED_UPPER_2)
    mask_red    = cv2.bitwise_or(mask_red1, mask_red2)

    mask_blue   = cv2.inRange(hsv, BLUE_LOWER,   BLUE_UPPER)
    mask_yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    mask_orange = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER)

    combined = cv2.bitwise_or(mask_red,    mask_blue)
    combined = cv2.bitwise_or(combined,    mask_yellow)
    combined = cv2.bitwise_or(combined,    mask_orange)

    return combined


def clean_mask(mask):
    """
    Làm sạch mask: blur → open (loại nhiễu) → close (lấp lỗ) → dilate (nối vùng).
    Trả về mask đã làm sạch.
    """
    # GaussianBlur nhẹ để mờ nhiễu pixel đơn lẻ
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    _, blurred = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    opened  = cv2.morphologyEx(blurred,  cv2.MORPH_OPEN,  kernel_open)
    closed  = cv2.morphologyEx(opened,   cv2.MORPH_CLOSE, kernel_close)
    dilated = cv2.dilate(closed, kernel_dilate, iterations=2)

    return dilated


def find_candidate_contours(mask):
    """Tìm tất cả external contours từ mask đã làm sạch."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_contours_by_shape(contours, image_shape):
    """
    Lọc contour theo diện tích, tỷ lệ khung hình, và hình dạng (tròn/tam giác/vuông).
    Trả về danh sách (x, y, w, h) đã được lọc.
    """
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
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio < MIN_ASPECT or aspect_ratio > MAX_ASPECT:
            continue

        # Tính circularity
        perimeter = cv2.arcLength(cnt, True)
        circularity = 0.0
        if perimeter > 0:
            circularity = (4 * math.pi * area) / (perimeter ** 2)

        # Xấp xỉ đa giác
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        n_vertices = len(approx)

        # Extent (tỷ lệ lấp đầy bounding box)
        extent = area / (w * h) if (w * h) > 0 else 0

        is_circle    = circularity > 0.6
        is_triangle  = n_vertices == 3
        is_rect      = n_vertices == 4
        is_polygon   = n_vertices >= 5 and circularity > 0.4

        if is_circle or is_triangle or is_rect or is_polygon:
            valid_boxes.append((x, y, w, h, area))

    # Sort theo area giảm dần
    valid_boxes.sort(key=lambda b: b[4], reverse=True)
    return [(x, y, w, h) for x, y, w, h, _ in valid_boxes]


def iou(box1, box2):
    """Tính Intersection over Union của hai bounding box (x,y,w,h)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ix = max(x1, x2)
    iy = max(y1, y2)
    iw = min(x1+w1, x2+w2) - ix
    ih = min(y1+h1, y2+h2) - iy
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = w1*h1 + w2*h2 - inter
    return inter / union if union > 0 else 0.0


def merge_or_nms_boxes(boxes):
    """
    Non-Max Suppression đơn giản: loại các box chồng nhau quá nhiều.
    Giữ lại box lớn hơn (đã được sort theo area giảm dần).
    """
    keep = []
    suppressed = [False] * len(boxes)

    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(boxes[i])
        for j in range(i+1, len(boxes)):
            if not suppressed[j]:
                if iou(boxes[i], boxes[j]) > NMS_THRESHOLD:
                    suppressed[j] = True

    return keep[:MAX_ROIS]


def crop_rois(image, boxes):
    """
    Crop từng ROI từ ảnh gốc với padding.
    Trả về danh sách (roi_crop, padded_box).
    """
    img_h, img_w = image.shape[:2]
    rois = []

    for (x, y, w, h) in boxes:
        pad_x = int(w * PADDING_RATIO)
        pad_y = int(h * PADDING_RATIO)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(img_w, x + w + pad_x)
        y1 = min(img_h, y + h + pad_y)

        roi = image[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        rois.append((roi, (x0, y0, x1-x0, y1-y0)))

    return rois


def preprocess_for_hog(roi):
    """
    Resize về 64x64, chuyển grayscale.
    Trả về: (img_resized_bgr, img_gray)
    """
    img_resized = cv2.resize(roi, IMG_SIZE)
    img_gray    = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    return img_resized, img_gray


def visualize_hog(gray):
    """
    Trích xuất và trực quan hoá HOG.
    Trả về: (feature_vector, hog_image_uint8)
    """
    feature, hog_img = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        visualize=True,
        feature_vector=True
    )
    hog_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))
    hog_uint8    = (hog_rescaled * 255).astype(np.uint8)
    # Chuyển sang BGR để encode nhất quán
    hog_bgr      = cv2.cvtColor(hog_uint8, cv2.COLOR_GRAY2BGR)
    return feature, hog_bgr


def predict_roi(feature, mdl):
    """
    Dự đoán nhãn từ feature vector HOG.
    Trả về: (label_str, score_float_or_None)
    """
    feat = feature.reshape(1, -1)
    pred = mdl.predict(feat)[0]

    if isinstance(pred, (int, np.integer)):
        label = CLASSES[int(pred)]
    else:
        try:
            label = CLASSES[int(pred)]
        except (ValueError, IndexError):
            label = str(pred)

    score = None
    if hasattr(mdl, "predict_proba"):
        try:
            proba = mdl.predict_proba(feat)[0]
            score = round(float(np.max(proba)) * 100, 2)
        except Exception:
            pass
    elif hasattr(mdl, "decision_function"):
        try:
            df = mdl.decision_function(feat)[0]
            # Normalize decision score sang 0-100 (heuristic)
            if isinstance(df, np.ndarray):
                score = round(float(np.max(df)), 3)
            else:
                score = round(float(df), 3)
        except Exception:
            pass

    return label, score


def draw_predictions(image, results):
    """
    Vẽ bounding box và nhãn lên ảnh gốc.
    results: danh sách dict với keys 'bbox', 'label', 'score'
    Trả về ảnh đã vẽ (copy).
    """
    out = image.copy()
    colors = [(0, 200, 0), (0, 120, 255), (255, 60, 0),
              (200, 0, 200), (0, 200, 200)]

    for i, r in enumerate(results):
        x, y, w, h = r['bbox']
        color = colors[i % len(colors)]
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)

        label_text = r['label']
        if r['score'] is not None:
            label_text += f" {r['score']}%"

        # Nền nhãn
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y - 8, th + 4)
        cv2.rectangle(out, (x, ty - th - 4), (x + tw + 4, ty + 2), color, -1)
        cv2.putText(out, label_text, (x + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return out


def detect_traffic_sign_rois(image):
    """
    Pipeline đầy đủ: color mask → clean → contour → filter → NMS → crop.
    Trả về:
      - rois_with_boxes: list of (roi_crop, bbox_tuple)
      - raw_mask_b64: base64 ảnh mask thô
      - clean_mask_b64: base64 ảnh mask đã làm sạch
      - fallback: True nếu không tìm được ROI và dùng toàn bộ ảnh
    """
    raw_mask   = create_color_mask(image)
    clean      = clean_mask(raw_mask)

    # Chuyển mask sang BGR để encode màu
    raw_mask_bgr   = cv2.cvtColor(raw_mask,  cv2.COLOR_GRAY2BGR)
    clean_mask_bgr = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)

    contours = find_candidate_contours(clean)
    boxes    = filter_contours_by_shape(contours, image.shape)
    boxes    = merge_or_nms_boxes(boxes)

    fallback = False
    if not boxes:
        # Fallback: dùng toàn bộ ảnh
        fallback = True
        h, w = image.shape[:2]
        boxes = [(0, 0, w, h)]

    rois_with_boxes = crop_rois(image, boxes)

    return (rois_with_boxes,
            encode_image_to_base64(raw_mask_bgr),
            encode_image_to_base64(clean_mask_bgr),
            fallback)


# ─── Flask routes ─────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename       = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath       = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)

            if model is None:
                return render_template('index.html',
                    error="Model chưa được load. Vui lòng kiểm tra đường dẫn model.")

            img = cv2.imread(filepath)
            if img is None:
                return render_template('index.html',
                    error="Không đọc được file ảnh. Vui lòng upload file ảnh hợp lệ.")

            # ── Detect ROIs ──────────────────────────────────────────────────
            rois_with_boxes, raw_mask_b64, clean_mask_b64, fallback = \
                detect_traffic_sign_rois(img)

            # ── Predict từng ROI ─────────────────────────────────────────────
            roi_results = []
            for roi_crop, bbox in rois_with_boxes:
                img_resized, img_gray = preprocess_for_hog(roi_crop)
                feature, hog_bgr      = visualize_hog(img_gray)
                label, score          = predict_roi(feature, model)

                roi_results.append({
                    'bbox':          bbox,
                    'label':         label,
                    'score':         score,
                    'crop_b64':      encode_image_to_base64(roi_crop),
                    'resized_b64':   encode_image_to_base64(img_resized),
                    'grayscale_b64': encode_image_to_base64(cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)),
                    'hog_b64':       encode_image_to_base64(hog_bgr),
                })

            # ── Vẽ bounding boxes lên ảnh gốc ───────────────────────────────
            boxed_img    = draw_predictions(img, roi_results)
            original_b64 = encode_image_to_base64(img)
            boxed_b64    = encode_image_to_base64(boxed_img)

            return render_template('index.html',
                original_b64   = original_b64,
                raw_mask_b64   = raw_mask_b64,
                clean_mask_b64 = clean_mask_b64,
                boxed_b64      = boxed_b64,
                roi_results    = roi_results,
                fallback       = fallback,
            )

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)