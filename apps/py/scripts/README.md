# Golf Swing Prediction - Hướng Dẫn Sử Dụng

Script `predict.py` dùng để dự đoán điểm golf swing từ video và cung cấp phân tích chi tiết.

## Yêu Cầu

- Python 3.8+
- Các thư viện: numpy, opencv-python, torch, ultralytics, scikit-learn, joblib, pandas
- CUDA (tùy chọn, để tăng tốc độ xử lý)

## Cài Đặt

```bash
pip install numpy opencv-python torch ultralytics scikit-learn joblib pandas
```

## Cấu Trúc Thư Mục

```
scripts/
├── predict.py
├── extract_features_biomech_augmented.py
├── feature_metadata.py
├── models/
│   ├── yolov8m-pose.pt                      # YOLO pose detection model
│   ├── stage2_model_*.pkl                   # Trained classifier
│   ├── stage2_scaler_*.pkl                  # Feature scaler
│   ├── stage2_metadata_*.json               # Model metadata
│   └── feature_statistics.json              # Feature statistics
└── README.md
```

## Cách Sử Dụng

### 1. Dự đoán từ video

```bash
cd scripts/
python predict.py <đường_dẫn_video> --output <file_json>
```

**Ví dụ:**

```bash
python predict.py ../data/raw/1.mp4 --output result.json
```

### 2. Tùy chọn nâng cao

```bash
python predict.py <video> --output <json> --model_dir <thư_mục_model>
```

**Tham số:**
- `video`: Đường dẫn đến file video golf swing (required)
- `--output`: Đường dẫn file JSON output (optional)
- `--model_dir`: Thư mục chứa models (optional, mặc định: `scripts/models/`)

## Output

### Console Output

```
======================================================================
  GOLF SWING PREDICTION
======================================================================

Loading model: stage2_model_20260117_112718.pkl
Extracting skeleton...
Running prediction...
JSON saved: result.json

======================================================================
  Score: band_0_2
  Confidence: 98.8%
======================================================================

Strengths:
  + Bio Finish Angle: Good (84.0 degrees), close to pro avg (84.9)
  + Bio Shoulder Loc: Good (0.2 ), close to pro avg (2.2)
  + Kin Peak Velocity: Good (43.8 m/s (norm)), close to pro avg (73.7)

Areas for Improvement:
  - Bio Shoulder Hanging Back: above pro level (19.1 vs 6.1 ratio)
  - Bio Left Arm Angle Top: below pro level (72.5 vs 143.3 degrees)
  - Bio Hip Hanging Back: above pro level (11.7 vs 3.8 ratio)
```

### JSON Output

File JSON chứa:
- **score**: Dự đoán score band (band_0_2, band_2_4, etc.)
- **confidence**: Độ tin cậy của prediction
- **probabilities**: Xác suất cho từng band
- **insights**: Điểm mạnh và điểm yếu
- **features**: Chi tiết tất cả features với đánh giá (Good/Average/Poor/Excellent)

## Cách Hoạt Động

1. **Trích xuất skeleton**: Sử dụng YOLOv8 pose model để detect keypoints
2. **Tính toán features**: Tự động tính các features biomechanical từ skeleton
3. **Dự đoán**: Sử dụng trained classifier để dự đoán score band
4. **Phân tích**: So sánh với pro statistics để đưa ra insights

## Lưu Ý

- Video nên rõ ràng, ghi ở góc backside hoặc side để kết quả tốt nhất
- Lần chạy đầu tiên có thể mất thời gian load models
- Model tự động chọn file model mới nhất trong thư mục `models/`

## Troubleshooting

### Lỗi "No models found"

Đảm bảo thư mục `scripts/models/` có đầy đủ các file model:
- `yolov8m-pose.pt`
- `stage2_model_*.pkl`
- `stage2_scaler_*.pkl`
- `stage2_metadata_*.json`
- `feature_statistics.json`

### Lỗi CUDA/GPU

Nếu gặp lỗi CUDA, script sẽ tự động chuyển sang CPU mode.

### Lỗi import

Đảm bảo chạy từ thư mục `scripts/`:
```bash
cd scripts/
python predict.py ../data/raw/1.mp4
```
