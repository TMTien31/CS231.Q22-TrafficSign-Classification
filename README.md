# Demo Phân loại Biển báo Giao thông (SVM + HOG)

Ứng dụng web đơn giản bằng Flask dùng để demo kết quả đồ án Nhập môn Thị giác máy tính. 

**Điểm mới:** Demo này không train lại model. Demo chỉ load SVM-HOG model đã train và thêm bước crop/localization (tự động phát hiện và cắt vùng biển báo) trước khi phân loại.

## Cấu trúc thư mục

```
.
├── data/                      # Dataset (train/test theo class)
├── demo/                      # Demo Flask + HTML
│   ├── app.py                 # Entry point cho demo
│   ├── index.html             # Giao dien web frontend
│   └── pipeline.py            # Pipeline infer + crop
├── svm/                       # Tat ca tai san lien quan toi SVM
│   ├── SVMHOG.ipynb            # Notebook huan luyen SVM + HOG
│   ├── models/                # Model da train
│   ├── features/              # Feature extraction (HOG)
│   ├── svm_classification_report.txt
│   └── svm_confusion_matrix.png
├── cnn/                       # (De trong) se phat trien sau
├── requirements.txt           # Các dependencies cần cài đặt
├── static/
│   └── uploads/               # Thu muc luu anh upload
└── README.md                  # File huong dan
```

## Hướng dẫn cài đặt và chạy ứng dụng

1. **Chuẩn bị model:**
   Đảm bảo bạn đã train model và lưu file model (ưu tiên tên `svm_hog_tuned_model.joblib` hoặc `svm_hog_model.joblib`). Hãy đặt file này vào thư mục `svm/models/` hoặc `svm/svm_models/`.

2. **Cài đặt các thư viện cần thiết:**
   Mở terminal (PowerShell hoặc Command Prompt) tại thư mục chứa source code và chạy lệnh:
   ```bash
   pip install -r requirements.txt
   ```

3. **(Tuy chon) Huan luyen lai model:**
   Mo notebook [svm/SVMHOG.ipynb](svm/SVMHOG.ipynb) va chay theo thu tu cell. Du lieu lay tu `data/train` va `data/test`.

4. **Chạy ứng dụng Flask:**
   Sau khi cài đặt xong thư viện, khởi động máy chủ bằng lệnh:
   ```bash
   python demo/app.py
   ```

5. **Sử dụng Website:**
   - Mở trình duyệt web và truy cập vào địa chỉ: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
   - Upload một bức ảnh chứa biển báo giao thông (ảnh đã crop sát hoặc ảnh toàn cảnh).
   - Backend sẽ tự động phát hiện vùng biển báo màu đỏ, cắt nó ra và dự đoán. Kết quả trả về gồm Nhãn, độ tin cậy và hình ảnh (gốc + cắt).

## Giới hạn
Color-based localization (dựa vào màu sắc) chủ yếu phù hợp với các biển báo có màu đỏ (như biển Cấm, Nguy hiểm). Với các biển báo màu xanh/vàng hoặc những bức ảnh chụp quá mờ/quá xa, hệ thống có thể không phát hiện được vùng biển báo chính xác. Trong trường hợp đó, hệ thống sẽ fallback dự đoán trên toàn bộ ảnh gốc và hiển thị cảnh báo. Tốt nhất nên upload ảnh được crop sát vào biển báo để đạt độ chính xác cao nhất (như lúc model được huấn luyện).
## Cấu trúc luồng chạy (Demo Flow)
- **Lưu ý:** Demo KHÔNG huấn luyện (train) lại mô hình mà chỉ trực tiếp nạp mô hình (`svm_hog_tuned_model.joblib`) từ quá trình huấn luyện trước đó. 
- **Cách thức hoạt động trên giao diện web:** Hệ thống hỗ trợ xử lý ảnh ở 2 dạng:
  1. **Ảnh cận cảnh (Cropped):** Hoạt động với độ chính xác cao nhất. Phù hợp nếu ảnh người dùng upload chỉ chứa riêng một hình biển.
  2. **Ảnh toàn cảnh chụp thực tế (Full Scene):** Hệ thống sẽ áp dụng thuật toán Computer Vision dùng HSV để tự động dò và khoanh vùng (auto-crop) các đối tượng nghi ngờ là biển báo, tách ra khỏi bối cảnh nhiễu, sau đó dự đoán. Trường hợp không thể phát hiện, sẽ fallback lại dùng ảnh nguyên gốc nhằm báo lỗi.