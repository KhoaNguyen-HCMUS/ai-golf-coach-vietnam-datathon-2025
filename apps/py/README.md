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

Kiểm tra trạng thái API.

**Response:**

```json
{
  "status": "healthy",
  "message": "API is running"
}
```

### 2. Predict từ Video Upload ⭐ **NEW**

```bash
POST /predict
Content-Type: multipart/form-data

file: <video.mp4>
```

Upload video golf swing và nhận prediction với insights chi tiết.

**Response:**

```json
{
  "score": "band_0_2",
  "band_index": 0,
  "confidence": 0.988,
  "probabilities": {
    "band_0_2": 0.988,
    "band_2_4": 0.012,
    "band_4_6": 0.0,
    "band_6_8": 0.0,
    "band_8_10": 0.0
  },
  "insights": {
    "strengths": [
      "Bio Finish Angle: Good (84.0 degrees), close to pro avg (84.9)",
      "Bio Shoulder Loc: Good (0.2), close to pro avg (2.2)"
    ],
    "weaknesses": [
      "Bio Shoulder Hanging Back: above pro level (19.1 vs 6.1 ratio)",
      "Bio Left Arm Angle Top: below pro level (72.5 vs 143.3 degrees)"
    ]
  },
  "features": [
    {
      "name": "Bio Finish Angle",
      "key": "bio_finish_angle",
      "value": 84.0,
      "unit": "degrees",
      "importance": 0.085,
      "evaluation": "Good",
      "description": "Within pro range (84.9±5.2)"
    }
    // ... more features
  ]
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

# Predict từ video
curl -X POST http://localhost:8000/predict \
  -F "file=@../data/raw/1.mp4"
```

### Sử dụng Python requests

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Predict từ video
with open("../data/raw/1.mp4", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
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
const FormData = require('form-data');
const fs = require('fs');

async function predictGolfSwing(videoPath) {
  const formData = new FormData();
  formData.append('file', fs.createReadStream(videoPath));
  
  const response = await axios.post('http://localhost:8000/predict', formData, {
    headers: formData.getHeaders()
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
