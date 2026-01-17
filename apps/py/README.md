# Golf Swing Prediction API

FastAPI service để expose Python ML prediction service cho Golf Swing Analysis.

## 🚀 Cài đặt

### 1. Activate Environment

Go to the py directory

```bash
conda activate datathon2025
conda install -c conda-forge fastapi uvicorn
```

or use venv

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

### Chạy API Server

```bash
python main.py
```

Hoặc sử dụng uvicorn trực tiếp:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API sẽ chạy tại: **http://localhost:8000**

### API Documentation

Sau khi chạy server, truy cập:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🔌 API Endpoints

### 1. Health Check

```bash
GET /health
```

Kiểm tra trạng thái API và model.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_dir": "models"
}
```

### 2. Predict từ File Path

```bash
POST /predict
Content-Type: application/json

{
  "skeleton_path": "data/TDTU_skeletons_npy/2.npy"
}
```

**Response:**

```json
{
  "success": true,
  "prediction": 2,
  "band_name": "band 4-6",
  "probabilities": [0.1, 0.2, 0.5, 0.15, 0.05],
  "message": "Prediction thành công"
}
```

### 3. Predict từ Uploaded File

```bash
POST /predict/upload
Content-Type: multipart/form-data

file: <skeleton.npy file>
```

Upload file `.npy` và nhận prediction.

### 4. Batch Predict

```bash
POST /predict/batch
Content-Type: application/json

{
  "skeleton_paths": [
    "data/TDTU_skeletons_npy/2.npy",
    "data/TDTU_skeletons_npy/3.npy",
    "data/TDTU_skeletons_npy/8.npy"
  ]
}
```

**Response:**

```json
{
  "success": true,
  "results": [
    {
      "skeleton_path": "data/TDTU_skeletons_npy/2.npy",
      "prediction": 2,
      "band_name": "band 4-6",
      "probabilities": [0.1, 0.2, 0.5, 0.15, 0.05],
      "success": true
    },
    ...
  ],
  "message": "Đã xử lý 3 files"
}
```

## ⚙️ Configuration

### Environment Variables

- `MODEL_DIR`: Đường dẫn đến thư mục chứa models (mặc định: `../../models`)
- `PORT`: Port để chạy API (mặc định: `8000`)
- `HOST`: Host để bind (mặc định: `0.0.0.0`)

**Ví dụ:**

```bash
export MODEL_DIR=../../models
export PORT=8000
export HOST=0.0.0.0
python main.py
```

## 📝 Handicap Bands

Model predict 5 handicap bands:

- **0**: band 0-2
- **1**: band 2-4
- **2**: band 4-6
- **3**: band 6-8
- **4**: band 8-10

## 🧪 Test API

### Sử dụng curl

```bash
# Health check
curl http://localhost:8000/health

# Predict từ path
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"skeleton_path": "data/TDTU_skeletons_npy/2.npy"}'

# Upload file
curl -X POST http://localhost:8000/predict/upload \
  -F "file=@data/TDTU_skeletons_npy/2.npy"
```

### Sử dụng Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Predict
response = requests.post(
    "http://localhost:8000/predict",
    json={"skeleton_path": "data/TDTU_skeletons_npy/2.npy"}
)
print(response.json())

# Upload file
with open("data/TDTU_skeletons_npy/2.npy", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict/upload",
        files={"file": f}
    )
print(response.json())
```

## 🔧 Troubleshooting

### Lỗi: "Không tìm thấy GolfSwingPredictor"

Đảm bảo bạn đang chạy từ root directory của project và các module trong `src/` và `scripts/` có thể import được.

### Lỗi: "Model directory không tồn tại"

Kiểm tra:

1. Thư mục `models/` có tồn tại không
2. Set environment variable `MODEL_DIR` đúng đường dẫn
3. Có file model `stage2_model_*.pkl` trong thư mục

### Lỗi: "Skeleton file không tồn tại"

Đảm bảo đường dẫn skeleton file là đúng (relative hoặc absolute path).

## 📚 Tích hợp với Frontend/Backend

API này có thể được gọi từ:

- Node.js backend (apps/server)
- React/Next.js frontend (apps/web)
- Mobile apps
- IoT devices

Ví dụ tích hợp với Node.js:

```javascript
const axios = require('axios');

async function predictGolfSwing(skeletonPath) {
  const response = await axios.post('http://localhost:8000/predict', {
    skeleton_path: skeletonPath,
  });
  return response.data;
}
```

## 🚀 Production Deployment

Để deploy production, nên:

1. Sử dụng Gunicorn với Uvicorn workers
2. Set up reverse proxy (Nginx)
3. Enable HTTPS
4. Set CORS origins cụ thể thay vì `["*"]`
5. Add authentication/authorization
6. Monitor và logging

```bash
# Production với Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```
