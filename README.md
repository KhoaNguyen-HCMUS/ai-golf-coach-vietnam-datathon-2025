# How to run project

## Structure folder

This project is organized into two main folders:

- **`envs/`**: Contains environment for machine learning.
- **`apps/`**: Contains frontend, backend and machine learning app.

## Requirements

- Node 20+, pnpm 10+
- Conda 23+

## Install dependencies

```bash
pnpm install
```

## Activate conda environment

Create and activate the conda environment from the yml file:

```bash
conda env create -f envs/datathon2025.yml
conda activate datathon2025
```

## Run project

### Run all

```bash
pnpm dev
```

### Only server

```bash
pnpm be dev
```

### Only web

```bash
pnpm fe dev
```

## Outcome

Web: https://localhost:4000
Server: https://localhost:5000

## ML pipeline scripts (apps/ml/src)

- [adjust_event_frames.py](apps/ml/src/adjust_event_frames.py): đọc GolfDB CSV, lấy FPS từng video và cộng thêm `offset_seconds * fps` vào toàn bộ `events` để đồng bộ mốc pha sau khi cắt/ghép video.
- [evaluate.py](apps/ml/src/evaluate.py): chuẩn hóa skeleton người dùng, so khớp DTW với mẫu tham chiếu theo từng góc quay, áp luật hình học (chicken wing, head bobbing, soft lead leg), rồi xuất báo cáo JSON gồm điểm và feedback.
- [extract_skelectons.py](apps/ml/src/extract_skelectons.py): chạy YOLOv8 Pose trên video thô, chọn người có `confidence` cao nhất mỗi frame và lưu toàn bộ keypoint thành `.npy` để dùng cho các bước tiếp theo.
- [organize_videos.py](apps/ml/src/organize_videos.py): duyệt cấu trúc thư mục TDTU gốc, copy từng `.mov` thành `id.mp4` chuẩn hóa và đồng thời xuất CSV metadata (`place/band/view`).
- [prepare_videos.py](apps/ml/src/prepare_videos.py): chuẩn hóa toàn bộ video GolfDB và TDTU về chiều cao 720p, áp CLAHE tùy mức sáng để giảm nhiễu ánh sáng; đầu ra dùng chung cho mọi bước trích xuất/đánh giá (không còn đưa sang MediaPipe).
- [rulebased_detector.py](apps/ml/src/rulebased_detector.py): dùng tín hiệu cổ tay từ skeleton `.npy` để tự suy luận các pha Address/Top/Impact/Finish, đồng thời hỗ trợ xuất video debug với marker sự kiện.
- [score_mapper_trainer.py](apps/ml/src/score_mapper_trainer.py): gom điểm mô hình vs band thật, fit hàm tuyến tính + warp hai phía để hiệu chỉnh thang điểm 0-10 và báo cáo chính xác theo band.

### Gợi ý thứ tự chạy

```bash
# 1. Chuẩn hóa video nguồn (GolfDB + TDTU)
python apps/ml/src/prepare_videos.py

# 2. Tổ chức và gán metadata cho TDTU (điền lại DATA_FOLDER/OUTPUT_* trước)
python apps/ml/src/organize_videos.py

# 3. Trích skeleton từ video đã chuẩn hóa
python apps/ml/src/extract_skelectons.py

# 4. Nhận diện pha Address/Top/Impact/Finish và (tùy chọn) tạo video debug
python apps/ml/src/rulebased_detector.py

# 5. Điều chỉnh event frame trong GolfDB CSV (cập nhật đường dẫn trước)
python apps/ml/src/adjust_event_frames.py

# 6. Đánh giá một skeleton người dùng với mẫu tham chiếu
python apps/ml/src/evaluate.py

# 7. Huấn luyện bộ chuyển đổi điểm -> band thực tế
python apps/ml/src/score_mapper_trainer.py
```

> Lưu ý: trước khi chạy từng script hãy cập nhật đường dẫn cấu hình trong file tương ứng (thư mục video, skeleton, CSV, JSON đầu ra, model YOLO, v.v.) và đảm bảo môi trường `datathon2025` đã được kích hoạt đầy đủ dependency (OpenCV, Ultralytics, NumPy, Pandas, SciPy,...).

### Notebooks liên quan

- [apps/ml/notebooks/eda_dataset_comparison.ipynb](apps/ml/notebooks/eda_dataset_comparison.ipynb) thực hiện EDA so sánh dữ liệu TDTU và GolfDB: thống kê phân bố góc quay, môi trường, kỹ năng, phân tích variance skeleton để chứng minh GolfDB là bộ tham chiếu chuẩn.

## IoT edge node (Device/)

- [Device/src/main.cpp](Device/src/main.cpp#L1-L47) là entrypoint chạy trên ESP32: sau khi nhận lệnh `START` qua MQTT thì đọc IMU liên tục, phát hiện va chạm và gửi kết quả về topic phản hồi.
- [Device/src/sensors/IMUSensor.*](Device/src/sensors/IMUSensor.cpp#L1-L35) gói MPU6050, quy đổi raw value sang gia tốc (g) và vận tốc góc (deg/s) rồi trả về độ lớn vector.
- [Device/src/detection/ImpactDetector.*](Device/src/detection/ImpactDetector.cpp#L1-L18) so sánh gia tốc/vận tốc góc với ngưỡng (`ACC_THRESHOLD`, `GYRO_THRESHOLD`) đồng thời áp dụng thời gian hồi để tránh spam.
- [Device/src/communication/MqttClient.*](Device/src/communication/MqttClient.cpp#L1-L69) quản lý Wi-Fi, MQTT và giao thức điều khiển: subscribe lệnh từ topic `MQTT_TOPIC_CMD`, publish JSON kết quả lên `MQTT_TOPIC_RESULT` với thời điểm va chạm, độ trễ và biên độ.
- Cấu hình phần cứng/mạng nằm trong [Device/src/config.h](Device/src/config.h) (SSID, broker, topic, ngưỡng cảm biến); cập nhật file này trước khi flash.

### Cách build & flash nhanh bằng PlatformIO

```bash
cd Device
pio run              # build firmware
pio run -t upload    # flash ESP32 (ESP32 DOIT Devkit V1 theo platformio.ini)
pio device monitor   # xem log nối tiếp 115200 bps
```

> Yêu cầu đã cài PlatformIO CLI (`pip install platformio` hoặc cài VS Code PlatformIO). Trước khi cắm board nhớ cập nhật `config.h` với Wi-Fi/MQTT thật, sau đó reset để node kết nối và sẵn sàng nhận lệnh từ backend/web.
