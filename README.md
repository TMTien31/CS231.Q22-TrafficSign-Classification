# Demo Phân loại Biển báo Giao thông (HOG + SVM & HOG + Random Forest)

Ứng dụng web đơn giản bằng FastAPI dùng để demo kết quả đồ án Nhập môn Thị giác máy tính. 

**Điểm mới:** Demo này không train lại model. Demo chỉ load các mô hình đã được huấn luyện (SVM hoặc Random Forest kết hợp đặc trưng HOG) và thêm bước crop/localization (tự động phát hiện và khoanh vùng biển báo) trước khi tiến hành phân loại.

## Cấu trúc thư mục

```
.
├── cache/                     # Chứa các file .npz lưu trữ đặc trưng HOG đã trích xuất
├── data/                      # Dataset chia theo train/test và các lớp (Cam, Chidan, Hieulenh, Nguyhiem, Phu)
├── demo/                      # Demo FastAPI + HTML
│   ├── app.py                 # Entry point chính cho backend (FastAPI), tích hợp logic pipeline
│   └── index.html             # Giao diện web frontend
├── models/                    # Tọa độ lưu các model đã được huấn luyện (.joblib)
├── sign_classify_train/       # Chứa các thư mục và Notebook để huấn luyện từng loại model
│   ├── HOG_RandomForest/      # Notebook train mô hình HOG + Random Forest (các cấu hình 6x3, 8x2)
│   └── HOG_SVM/               # Notebook train mô hình HOG + SVM (các cấu hình 6x3, 8x2)
├── notebook_demo/             # Chức các notebook thử nghiệm dự đoán và trực quan hóa phân lớp
├── static/                    # Thư mục lưu tĩnh web
│   ├── crops/                 # Ảnh đã được hệ thống crop
│   └── uploads/               # Ảnh người dùng upload
├── svm/                       # Các module trích xuất đặc trưng
│   └── svm_features_extraction/
├── requirements.txt           # Các dependencies cần cài đặt
└── README.md                  # File hướng dẫn
```

## Hướng dẫn cài đặt và chạy ứng dụng

1. **Chuẩn bị model:**
   Các mô hình (`.joblib`) như `HOG_SVM_8x2.joblib`, `HOG_RandomForest_6x3.joblib`,... đã được huấn luyện và lưu sẵn trong thư mục `models/`. Do đó, ứng dụng sẽ có thể chạy được ngay.

2. **Cài đặt các thư viện cần thiết:**
   Mở terminal (PowerShell hoặc Command Prompt) tại thư mục chứa source code và chạy lệnh:
   ```bash
   pip install -r requirements.txt
   ```

3. **(Tùy chọn) Huấn luyện lại model:**
   Nếu muốn huấn luyện lại hoặc thử nghiệm, chuyển vào thư mục `sign_classify_train/`. Tại đây, mở file notebook mong muốn trong `HOG_RandomForest/` hoặc `HOG_SVM/` và lần lượt chạy các cell code.  Dữ liệu nằm ở thư mục `data/`.

4. **Chạy ứng dụng FastAPI:**
   Sau khi cài đặt xong thư viện, khởi động máy chủ bằng lệnh (phải cd vào thư mục gốc `demo/`):
   ```bash
   python demo/app.py
   ```
   Hoặc bạn có thể dùng `uvicorn` trực tiếp:
   ```bash
   uvicorn demo.app:app --reload --port 5000
   ```

5. **Sử dụng Website:**
   - Mở trình duyệt web và truy cập vào địa chỉ: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
   - Upload một bức ảnh chứa biển báo giao thông (ảnh đã crop sát hoặc ảnh toàn cảnh).
   - Backend sẽ tự động phát hiện vùng biển báo màu đỏ, khoanh vùng rồi cắt nó ra (nếu là ảnh toàn cảnh) để dự đoán. Kết quả trả về gồm Nhãn, độ tin cậy và hình ảnh (gốc + cắt).

## Giới hạn
Color-based localization (dựa vào màu sắc HSV) chủ yếu phù hợp với các biển báo có màu đỏ (như biển Cấm, Nguy hiểm). Với các biển báo màu xanh/vàng hoặc những bức ảnh chụp quá mờ/quá xa, hệ thống Computer Vision có thể không phát hiện được vùng biển báo chính xác. Trong trường hợp đó, hệ thống sẽ fallback và thiết lập dự đoán trên toàn bộ ảnh gốc, lúc này kết quả có thể kém chính xác. Tốt nhất là upload ảnh đã được crop sát vào biển báo để mô hình hoạt động hiệu quả tối đa theo tập dataset.

## Cấu trúc luồng chạy (Demo Flow)
- **Lưu ý:** Demo KHÔNG huấn luyện (train) lại mô hình mà chỉ trực tiếp nạp mô hình (`.joblib`) từ thư mục `models/` trong lúc khởi động. 
- **Cách thức hoạt động trên giao diện web:** Hệ thống hỗ trợ xử lý ảnh ở 2 dạng:
  1. **Ảnh cận cảnh (Cropped):** Hoạt động với độ chính xác cao nhất do focus hoàn toàn vào biển báo. Phù hợp nếu ảnh người dùng upload đã cắt riêng mỗi biển.
  2. **Ảnh toàn cảnh chụp thực tế (Full Scene):** Hệ thống sẽ kết hợp thuật toán tự động dò và khoanh vùng (auto-crop) các cấu trúc nghi là biển báo dựa theo màu đặc trưng, tách đối tượng nổi bật, rồi mới chuyển cho model Machine Learning dự đoán. Trường hợp không thể phát hiện cụ thể vùng nào, sẽ lại dùng ảnh nguyên gốc để ráng dự đoán, nhưng báo lỗi đi kèm.