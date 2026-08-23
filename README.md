# YOLOv5 Object Detection – Train → Serve (FastAPI)

Pipeline tái sử dụng: **data → train (Colab) → export weights → REST API**.

Đổi dataset / số lớp / weights chỉ cần sửa file config YAML, **không phải viết lại code**.

---

## Cấu trúc thư mục

```
yolov5_api/
├── app.py                 # FastAPI server
├── pipeline.py            # Load model, predict image/video, draw boxes
├── configs/
│   ├── default.yaml       # Cấu hình mặc định
│   └── visdrone.yaml      # Ví dụ profile dataset khác
├── models/                # Đặt best.pt / last.pt vào đây
├── outputs/               # Video/ảnh đã annotate (tự tạo)
├── requirements.txt
└── README.md
```

---

## 1. Huấn luyện (Google Colab – tóm tắt)

```python
# Mount Drive, clone YOLOv5, cài deps
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/ultralytics/yolov5
%cd yolov5 && pip install -q -r requirements.txt

# data.yaml mẫu (COCO subset / VisDrone / custom)
# path, train, val, nc, names: [...]

# Train (transfer learning)
!python train.py --img 640 --batch 16 --epochs 50 \
  --data /content/data.yaml --weights yolov5s.pt --project /content/drive/MyDrive/runs

# best.pt nằm tại runs/train/exp/weights/best.pt → copy vào models/
```

Dataset format YOLO: mỗi ảnh có file `.txt`  
`class_id x_center y_center width height` (chuẩn hóa 0–1) + `data.yaml`.

---

## 2. Cài đặt & chạy API

```bash
cd yolov5_api
pip install -r requirements.txt

# Đặt weights đã train
cp /path/to/best.pt models/best.pt

# (Tuỳ chọn) sửa configs/default.yaml: device, conf_thres, names, ...

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: **http://localhost:8000/docs**

Colab public: dùng `pyngrok` expose port 8000.

---

## 3. API endpoints

| Method | Path | Mô tả |
|--------|------|--------|
| GET | `/` | Thông tin service |
| GET | `/health` | Health check + CUDA |
| GET | `/models` | Liệt kê weights & profiles |
| POST | `/models/upload` | Upload file `.pt` |
| POST | `/predict/image` | Nhận ảnh → JSON boxes (hoặc PNG annotate) |
| POST | `/predict/video` | Nhận video → JSON từng frame (+ video output) |
| GET | `/outputs/{filename}` | Tải file output |

### Ví dụ gọi API

```bash
# JSON kết quả
curl -X POST "http://localhost:8000/predict/image?conf=0.3" \
  -F "file=@test.jpg"

# Ảnh đã vẽ box
curl -X POST "http://localhost:8000/predict/image?return_image=true" \
  -F "file=@test.jpg" -o result.png

# Dùng profile VisDrone
curl -X POST "http://localhost:8000/predict/image?model=visdrone" \
  -F "file=@drone.jpg"

# Video
curl -X POST "http://localhost:8000/predict/video?save_video=true" \
  -F "file=@clip.mp4"
```

### Response ảnh (JSON)

```json
{
  "num_detections": 3,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.87,
      "bbox": [120.5, 80.0, 300.2, 450.1]
    }
  ],
  "inference_ms": 42.5,
  "image_width": 1280,
  "image_height": 720,
  "model_weights": "models/best.pt"
}
```

---

## 4. Tính mở rộng (scalable)

| Việc cần làm | Cách làm |
|--------------|----------|
| Đổi dataset / số lớp | Train lại → copy `best.pt` vào `models/` |
| Đổi tên lớp hiển thị | Sửa `names:` trong `configs/*.yaml` |
| Nhiều model song song | Thêm `configs/coco.yaml`, `configs/visdrone.yaml` → gọi `?model=visdrone` |
| Đổi ngưỡng conf/IoU | Query param `?conf=0.4&iou=0.5` hoặc sửa yaml |
| Đổi device | `model.device: cuda` trong yaml hoặc `device: auto` |

`pipeline.Detector.from_config(...)` đọc yaml → không hard-code path / số lớp.

---

## 5. Dùng pipeline trực tiếp (không qua API)

```python
from pipeline import Detector
import cv2

det = Detector.from_config("configs/default.yaml")
img = cv2.imread("test.jpg")
results = det.predict_image(img, conf_thres=0.3)
annotated = det.draw(img, results)
cv2.imwrite("out.jpg", annotated)
print(results)
```

---

## Ghi chú

- Weights phải train bằng **classic** `ultralytics/yolov5` (repo GitHub), không phải package `ultralytics` (YOLOv8).
- Lần chạy đầu tiên sẽ tự `git clone` yolov5 nếu chưa có.
- Export ONNX (tuỳ chọn): `python yolov5/export.py --weights models/best.pt --include onnx`