import os
import cv2
import joblib
import numpy as np
import sys
import uuid

# Workaround cho lỗi "ModuleNotFoundError: No module named 'numpy._core'" do khác biệt phiên bản numpy giữa Kaggle (2.x) và local (1.x)
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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['CROP_FOLDER'] = os.path.join('static', 'crops')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Maksimum 16MB

# Tạo thư mục nếu chưa có
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CROP_FOLDER'], exist_ok=True)

# Thông số model (Sao chép từ notebook)
IMG_SIZE = (64, 64)
CLASSES = ["Cam", "Chidan", "Hieulenh", "Nguyhiem", "Phu"]

# Ưu tiên load tuned model, nếu không có thì thử model thường
if os.path.exists(os.path.join("models", "svm_hog_tuned_model.joblib")):
    MODEL_PATH = os.path.join("models", "svm_hog_tuned_model.joblib")
else:
    MODEL_PATH = os.path.join("models", "svm_hog_model.joblib")

# Load model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")
else:
    model = None
    print(f"Model file not found at {MODEL_PATH}!")

def detect_traffic_sign(img):
    """
    Sử dụng color masking để tìm vùng candidate màu đỏ
    Trả về vùng crop nếu tìm thấy, hoặc None
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Mask đỏ (đỏ có 2 dải trong HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Morphology để giảm nhiễu
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Tìm contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w
    
    best_candidate = None
    max_red_area = 0
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = float(w) / h
        
        # Tiêu chí lọc
        if area > 400 and area < img_area * 0.8 and 0.6 <= aspect_ratio <= 1.4:
            # Tính diện tích pixel đỏ trong bounding box
            roi_mask = mask[y:y+h, x:x+w]
            red_pixels = cv2.countNonZero(roi_mask)
            
            if red_pixels > max_red_area:
                max_red_area = red_pixels
                best_candidate = (x, y, w, h)
                
    if best_candidate is not None:
        x, y, w, h = best_candidate
        
        # Thêm padding 20-30%
        pad_x = int(w * 0.25)
        pad_y = int(h * 0.25)
        
        new_x = max(0, x - pad_x)
        new_y = max(0, y - pad_y)
        new_w = min(img_w - new_x, w + 2 * pad_x)
        new_h = min(img_h - new_y, h + 2 * pad_y)
        
        return img[new_y:new_y+new_h, new_x:new_x+new_w]
        
    return None

def preprocess_and_extract_hog(img):
    """
    Tiền xử lý và trích xuất đặc trưng HOG giống như trong notebook huấn luyện.
    """
    # 1. Resize ảnh
    img_resized = cv2.resize(img, IMG_SIZE)
    # 2. Chuyển sang ảnh xám
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 3. Trích xuất đặc trưng HOG
    feature = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        visualize=False,
        feature_vector=True
    )
    return feature.reshape(1, -1)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Kiểm tra xem có file được upload không
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4().hex}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            if model is None:
                return render_template('index.html', error="Model chưa được load. Vui lòng kiểm tra lại đường dẫn model.")
            
            # Đọc ảnh
            img = cv2.imread(filepath)
            
            if img is None:
                return render_template('index.html', error="Không đọc được file ảnh. Vui lòng upload file ảnh hợp lệ.")
            
            # Localization
            crop_img = detect_traffic_sign(img)
            localization_failed = False
            crop_filepath = None
            
            if crop_img is not None:
                crop_filename = f"crop_{unique_filename}"
                crop_filepath = os.path.join(app.config['CROP_FOLDER'], crop_filename)
                cv2.imwrite(crop_filepath, crop_img)
                inference_img = crop_img
            else:
                localization_failed = True
                inference_img = img
            
            # Extract HOG
            features = preprocess_and_extract_hog(inference_img)
            
            # Model Predict
            pred = model.predict(features)[0]
            
            # Xử lý nếu pred là index hoặc string label
            if isinstance(pred, (int, np.integer)):
                pred_class = CLASSES[pred]
            else:
                try:
                    pred_idx = int(pred)
                    pred_class = CLASSES[pred_idx]
                except ValueError:
                    pred_class = pred
            
            # Confidence (nếu model có probability=True)
            confidence = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(features)[0]
                    confidence = round(np.max(proba) * 100, 2)
                except Exception:
                    pass
            
            return render_template('index.html', 
                                   image_path='/' + filepath.replace('\\', '/'), 
                                   crop_path=('/' + crop_filepath.replace('\\', '/')) if crop_filepath else None,
                                   localization_failed=localization_failed,
                                   pred_class=pred_class, 
                                   confidence=confidence)
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
