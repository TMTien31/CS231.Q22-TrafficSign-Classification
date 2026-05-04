```latex
\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage[vietnamese]{babel}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{hyperref}

\title{Báo cáo Đồ án: Phân loại Biển báo Giao thông (SVM + HOG)}
\author{}
\date{}

\begin{document}
\maketitle

\section*{A. Tóm tắt phương pháp SVM + HOG (Dùng khi thuyết trình)}
Hệ thống phân loại biển báo của chúng em sử dụng kỹ thuật trích xuất đặc trưng \textbf{HOG (Histogram of Oriented Gradients)} kết hợp với thuật toán học máy \textbf{SVM (Support Vector Machine)}. Ảnh đầu vào sẽ được chuyển sang ảnh xám và thay đổi về kích thước cố định là $64 \times 64$. Sau đó, thuật toán HOG sẽ phân tích hướng và độ lớn của các đường viền, góc cạnh trên biển báo để tạo thành một vector đặc trưng. Vector này được đưa vào mô hình SVM đã được huấn luyện để phân loại thành 1 trong 5 nhóm biển báo. Để tăng độ chính xác trong thực tế, chúng em tối ưu thêm bước \textbf{auto-crop (tự động khoanh vùng)} bằng không gian màu HSV để tách biển báo khỏi nền trước khi phân loại.

\section*{B. Phương pháp SVM + HOG (Viết cho Báo cáo)}

\subsection*{1. Tiền xử lý ảnh}
Để đảm bảo tính nhất quán của dữ liệu đầu vào và giảm thiểu chi phí tính toán, ảnh gốc được đưa qua hai bước xử lý chính:
\begin{itemize}
    \item \textbf{Chuẩn hóa kích thước (Resize)}: Mọi ảnh đầu vào được đưa về kích thước cố định $64 \times 64$ pixel. Do thuật toán học máy yêu cầu vector đầu vào có độ dài cố định, ảnh cũng cần có kích thước đồng nhất trước khi trích xuất đặc trưng.
    \item \textbf{Chuyển đổi ảnh xám (Grayscale)}: Biển báo giao thông có đặc trưng quan trọng nhất nằm ở hình khối (tròn, tam giác, chữ nhật) và các biểu tượng bên trong. Việc chuyển sang ảnh xám giúp loại bỏ sự phụ thuộc vào màu sắc và các nhiễu về độ rọi sáng, cho phép HOG tập trung phân tích không gian hình học, đồng thời giảm $2/3$ khối lượng tính toán.
\end{itemize}

\subsection*{2. Trích xuất đặc trưng HOG (Histogram of Oriented Gradients)}
HOG là thuật toán trích xuất đặc trưng mạnh mẽ trong thị giác máy tính, chuyên dùng để nhận diện đối tượng dựa trên hình dạng. 
HOG hoạt động bằng cách chia ảnh thành các ô nhỏ ($8 \times 8$ pixel) và tính toán biến thiên cường độ sáng (gradient) của các pixel theo 9 hướng (orientations=9). Các ô được nhóm lại thành các block ($2 \times 2$) để chuẩn độ (normalization) bằng \texttt{L2-Hys}, giúp hạn chế ảnh hưởng của độ tương phản sáng tối khác nhau. Cuối cùng, một vector đặc trưng duy nhất đại diện cho hình dạng của biển báo được tạo ra.

\subsection*{3. Phân loại bằng Support Vector Machine (SVM)}
SVM là thuật toán phân lớp được lựa chọn vì tính hiệu quả cao khi làm việc với dữ liệu đa chiều. Vector HOG trích xuất từ ảnh có số chiều tương đối lớn. SVM sẽ tìm kiếm các "siêu mặt phẳng" (hyperplanes) tối ưu trong không gian đa chiều này nhằm phân chia dữ liệu ra làm 5 lớp biển báo với biên nhận dạng (margin) lớn nhất có thể. Với bộ dữ liệu thử nghiệm (test set), phương pháp này đạt độ chính xác tổng quát lên tới 83.23\%.

\subsection*{4. Quy trình Huấn luyện \& Suy luận}
\begin{itemize}
    \item \textbf{Quy trình Huấn luyện (Training):} Đọc tập dữ liệu dạng ảnh được cắt sẵn (cropped) $\rightarrow$ Resize $64 \times 64$ $\rightarrow$ Grayscale $\rightarrow$ Trích xuất vector HOG $\rightarrow$ Huấn luyện mô hình phân loại SVM $\rightarrow$ Lưu file \texttt{.joblib}.
    \item \textbf{Quy trình Suy luận (Inference trong Demo):} Ảnh chụp truyền vào $\rightarrow$ Chạy thuật toán phát hiện và cắt (localization/crop) vùng chứa biển báo $\rightarrow$ Resize vùng được cắt về $64 \times 64$ $\rightarrow$ Trích xuất đặc trưng HOG $\rightarrow$ SVC dự đoán xác suất, trả về nhãn cuối cùng.
\end{itemize}

\section*{C. Cải tiến demo với Auto-Crop/Localization (Báo cáo)}
\textbf{Vấn đề khác biệt miền dữ liệu (Domain Mismatch):} 
Ban đầu, đối với hình ảnh ngoài đời thực (toàn cảnh), đối tượng biển báo thường rất nhỏ và xung quanh chứa nhiều nhiễu như xe cộ, tòa nhà, cây cối rậm rạp. Nếu thu nhỏ trực tiếp toàn bộ bức ảnh này xuống $64 \times 64$, biển báo sẽ bị mất đi hình dạng, thay vào đó vector HOG sẽ thu nhận đặc trưng của hậu cảnh (background), dẫn đến việc mô hình phân loại sai hoàn toàn.

\textbf{Giải pháp Localizaton/Auto-crop:}
Để khắc phục, một pipeline dò tìm (detect) đơn giản được tích hợp:
\begin{enumerate}
    \item Chuyển ảnh không gian BGR sang hệ màu HSV (do HSV biểu diễn màu sắc gần với cảm nhận của mắt người và xử lý ánh sáng tốt hơn).
    \item Tạo \textit{mask} (mặt nạ) để trích xuất dải màu đỏ (do phần lớn biển hệ thống Cấm và Nguy Hiểm có viền màu đỏ).
    \item Sử dụng các kỹ thuật Morphology (mở/đóng) nhằm khử nhiễu các điểm màu lốm đốm.
    \item Tìm đường viền (find contours) dựa trên mặt nạ. Lọc các bounding box dựa trên diện tích vùng và \textbf{Tỷ lệ khung hình (Aspect Ratio)} sao cho gần bằng 1 (hình vuông hoặc tỉ lệ cân xứng của hình tròn/tam giác).
    \item Sau khi lấy được vùng ứng viên có khả năng là biển báo cao nhất, hệ thống sẽ thực hiện padding thêm một khoảng nhỏ và crop (cắt) riêng biệt vùng đó, loại bỏ hoàn toàn background trước khi đưa tới bước phân loại SVM. Kết quả thực nghiệm cho thấy cải tiến này khắc phục dứt điểm tình trạng dự đoán sai với ảnh toàn cảnh.
\end{enumerate}

\section*{D. Hạn chế và Hướng phát triển (Báo cáo)}
\textbf{Hạn chế:}
\begin{enumerate}
    \item Thuật toán auto-crop với không gian màu HSV chủ yếu tìm kiếm dải màu đỏ, do vậy chỉ hoạt động tối ưu nhất cho Biển Báo Cấm và Biển Cảnh Báo Nguy hiểm. Nếu nhập vào các dạng Biển Chỉ dẫn (màu xanh lá / xanh dương), hệ thống có khả năng bỏ sót vùng crop (fallback về ảnh nguyên bản).
    \item Dễ bị nhầm lẫn và nhận diện sai (False Positive) nếu ảnh có nhiều vật thể màu đỏ (bảng quảng cáo, xe màu đỏ) hoặc bị nhiễu do điều kiện thiếu sáng.
    \item HOG mô phỏng đặc trưng rất tốt nhưng khá nhạy cảm với việc biển báo bị biến dạng mạnh hoặc xoay nhiều góc độ. Pipeline kết hợp SVM + HOG thuần túy là hệ thống phân loại định danh (Classification), không phải thuật toán phân loại và định vị chi tiết nhiều đối tượng (Object Detection) hoàn chỉnh.
\end{enumerate}

\textbf{Hướng phát triển:}
\begin{itemize}
    \item Thay vì dùng OpenCV Mask màu truyền thống, tương lai có thể áp dụng nguyên bản các thuật toán Object Detection (như YOLOv8 hoặc SSD, Faster R-CNN) để khoanh vùng tự động (Bounding Box Detection) mọi loại màu biển báo một cách bền vững.
    \item Thu thập và Data augmentation (tăng cường dữ liệu) thêm nhằm khắc phục thay đổi góc xoay.
\end{itemize}

\section*{E. Câu ngắn trong Slide: "Vì sao phải crop trước khi predict?"}
\textbf{"Để loại bỏ khoảng nhiễu lớn từ hậu cảnh. Nếu trực tiếp resize ảnh toàn cảnh về chuẩn $64 \times 64$, biển báo sẽ quá nhỏ và bị biến dạng, khiến thuật toán HOG trích xuất nhầm đặc trưng của cây cối/nhà cửa thay vì hình dáng biển báo, dẫn đến dự đoán sai."}

\section*{F. Câu trả lời giảng viên: "Tại sao test accuracy tập dữ liệu train cao (83.23\%) nhưng chụp thực tế đưa vào mạng lại sai?"}
"Dạ thưa thầy/cô, đó là hiện tượng \textbf{Mismatch Domain (Lệch miền dữ liệu)}. Trong tập dataset dùng để train và test ảnh đã được khoanh vùng cắt rập cận cảnh, tức là biển báo chiếm từ 80-100\% diện tích các góc ảnh. Thuật toán HOG học được một cách tuyệt đối hình khối đó. Khi chúng em đem hình chụp thực tế toàn cảnh vào mạng mà không crop, không gian chiếm chủ đạo trong ảnh là đường sá, xe cộ, tòa nhà. Thuật toán đưa nguyên cả khu phố về kích thước $64 \times 64$, nên model bị biến dạng và không thấy biển báo đâu cả. Vì vậy chúng em mới phải bổ sung luồng Auto-crop khoanh màu đỏ đằng trước để giải quyết triệt để lỗi này khi chạy demo."

\section*{G. Đề xuất chỉnh sửa README.md (markdown format)}
\begin{verbatim}
## Cấu trúc luồng chạy (Demo Flow)
- **Lưu ý:** Demo KHÔNG huấn luyện (train) lại mô hình mà chỉ trực tiếp nạp mô hình (`svm_hog_tuned_model.joblib`) từ quá trình huấn luyện trước đó. 
- **Cách thức hoạt động trên giao diện web:** Hệ thống hỗ trợ xử lý ảnh ở 2 dạng:
  1. **Ảnh cận cảnh (Cropped):** Hoạt động với độ chính xác cao nhất. Phù hợp nếu ảnh người dùng upload chỉ chứa riêng một hình biển.
  2. **Ảnh toàn cảnh chụp thực tế (Full Scene):** Hệ thống sẽ áp dụng thuật toán Computer Vision dùng HSV để tự động dò và khoanh vùng (auto-crop) các đối tượng nghi ngờ là biển báo, tách ra khỏi bối cảnh nhiễu, sau đó dự đoán. Trường hợp không thể phát hiện, sẽ fallback lại dùng ảnh nguyên gốc nhằm báo lỗi.
\end{verbatim}

\end{document}
```