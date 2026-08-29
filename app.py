"""
YOLOv5 & PyTorch Object Detection - FastAPI REST service.
Sử dụng mô hình hợp nhất: MobileNetV2 (128x128) cho cả PC và Mobile.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import time
import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from PIL import Image

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import (Detector, apply_env_overrides, default_config,
                      deep_merge, load_config)

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
CONFIGS_DIR = APP_DIR / "configs"
OUTPUTS_DIR = APP_DIR / "outputs"
DEFAULT_CONFIG = CONFIGS_DIR / "default.yaml"

for d in (MODELS_DIR, CONFIGS_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Shogi & Object Detection AI Dashboard",
    description="REST API hỗ trợ YOLOv5 và PyTorch MobileNetV2.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================================
# 1. KHỞI TẠO MÔ HÌNH B (MOBILENETV2 DÙNG CHUNG)
# ===========================================================================
class MobileNetClassifier(nn.Module):
    def __init__(self, num_figures=15, num_directions=2):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc_figure = nn.Linear(1280, num_figures)
        self.fc_direction = nn.Linear(1280, num_directions)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc_figure(x), self.fc_direction(x)

class PadToSquare:
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h: return img
        size = max(w, h)
        new_img = Image.new(img.mode, (size, size), color=self.fill)
        left = (size - w) // 2
        top = (size - h) // 2
        new_img.paste(img, (left, top))
        return new_img

FIGURE_CLASSES = [
    "BISHOP", "BISHOP_PROM", "EMPTY", "GOLD", "KING",
    "KNIGHT", "KNIGHT_PROM", "LANCE", "LANCE_PROM",
    "PAWN", "PAWN_PROM", "ROOK", "ROOK_PROM", "SILVER", "SILVER_PROM"
]
DIRECTION_CLASSES = ["UP", "DOWN"]

FIGURE_VIETNAMESE = {
    "BISHOP": "Tượng", "BISHOP_PROM": "Tượng phong cấp", "EMPTY": "Ô trống",
    "GOLD": "Tướng Vàng", "KING": "Vua", "KNIGHT": "Mã",
    "KNIGHT_PROM": "Mã phong cấp", "LANCE": "Hương xa", "LANCE_PROM": "Hương xa phong cấp",
    "PAWN": "Tốt", "PAWN_PROM": "Tốt phong cấp", "ROOK": "Xe",
    "ROOK_PROM": "Xe phong cấp", "SILVER": "Tướng Bạc", "SILVER_PROM": "Tướng Bạc phong cấp"
}
DIR_VIETNAMESE = {"UP": "Quân của mình (hướng lên)", "DOWN": "Quân đối thủ (hướng xuống)", "NONE": "Không xác định"}

# Transform 128x128 dùng chung cho cả 2 loại ảnh
transform_unified = T.Compose([
    PadToSquare(fill=0), 
    T.Resize((128, 128)), 
    T.ToTensor(), 
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device_b = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_b = None

# Nạp Mạng MobileNetV2 (Unified)
unified_path = MODELS_DIR / "piece_detection.pt"
if not unified_path.is_file():
    unified_path = APP_DIR / "piece_detection.pt"

if unified_path.is_file():
    try:
        model_b = MobileNetClassifier(num_figures=len(FIGURE_CLASSES), num_directions=len(DIRECTION_CLASSES)).to(device_b)
        model_b.load_state_dict(torch.load(unified_path, map_location=device_b, weights_only=True))
        model_b.eval()
        print(f"[app] Nạp thành công Mô hình Hợp nhất (MobileNetV2): {unified_path.name}")
    except Exception as e: print(f"[app] Lỗi nạp Mô hình: {e}")
else:
    print("[app] CẢNH BÁO: Không tìm thấy file piece_detection.pt trong thư mục models!")


# ===========================================================================
# 2. KHỞI TẠO MÔ HÌNH A (YOLOV5 DETECTOR)
# ===========================================================================
_detectors: Dict[str, Detector] = {}
_cache_lock = threading.Lock()
MAX_CACHED_MODELS = int(os.getenv("MAX_CACHED_MODELS", "4"))

def _resolve_config(model_name: Optional[str] = None) -> dict:
    cfg = default_config()
    cfg_path = Path(os.getenv("API_CONFIG", str(DEFAULT_CONFIG)))
    if cfg_path.is_file(): deep_merge(cfg, load_config(cfg_path))
    apply_env_overrides(cfg)

    if model_name:
        profile = CONFIGS_DIR / f"{model_name}.yaml"
        if profile.is_file(): deep_merge(cfg, load_config(profile))
        else:
            candidate = MODELS_DIR / model_name
            if candidate.is_file(): cfg.setdefault("model", {})["weights"] = str(candidate)
            elif Path(model_name).is_file(): cfg.setdefault("model", {})["weights"] = str(Path(model_name).resolve())
            else: raise HTTPException(404, f"Model '{model_name}' không tồn tại.")
    
    m = cfg.setdefault("model", {})
    weights = m.get("weights", "")
    if weights and not os.path.isabs(weights):
        alt = MODELS_DIR / Path(weights).name
        if alt.is_file(): m["weights"] = str(alt)
    if m.get("device") in (None, "", "auto"): m["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

def get_detector(model_name: Optional[str] = None) -> Detector:
    cfg = _resolve_config(model_name)
    key = json.dumps(cfg, sort_keys=True, default=str)
    det = _detectors.get(key)
    if det is not None: return det

    weights = cfg["model"].get("weights", "")
    if not weights or not os.path.isfile(weights): raise HTTPException(404, "Weights YOLOv5 không tìm thấy.")
    det = Detector.from_cfg(cfg)
    with _cache_lock:
        _detectors.setdefault(key, det)
        while len(_detectors) > MAX_CACHED_MODELS: _detectors.pop(next(iter(_detectors)))
        det = _detectors[key]
    return det

# ===========================================================================
# 3. RESPONSE SCHEMAS & ENDPOINTS
# ===========================================================================
class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2]")

class ImagePredictResponse(BaseModel):
    num_detections: int
    detections: List[DetectionItem]
    inference_ms: float
    image_width: int
    image_height: int
    annotated_image_b64: Optional[str] = Field(None)
    model_weights: str
    piece_info: Optional[Dict[str, Any]] = None

@app.get("/")
def root():
    index_file = APP_DIR / "index.html"
    if index_file.is_file(): return FileResponse(index_file)
    return {"error": "Không tìm thấy file index.html trong thư mục!"}

# --- ENDPOINT PHÂN LOẠI QUÂN CỜ ---
@app.post("/predict/piece")
async def predict_piece_endpoint(file: UploadFile = File(...)):
    if model_b is None:
        raise HTTPException(500, "Lỗi: Chưa nạp được mô hình. Hãy chắc chắn file piece_detection.pt đang nằm trong thư mục models/.")

    raw = await file.read()
    if not raw: raise HTTPException(400, "File rỗng")

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        tensor = transform_unified(img).unsqueeze(0).to(device_b)

        with torch.no_grad():
            pred_fig, pred_dir = model_b(tensor)
            fig_idx = pred_fig.argmax(1).item()
            dir_idx = pred_dir.argmax(1).item()
            prob_fig = torch.softmax(pred_fig, dim=1)[0, fig_idx].item()

        figure_raw = FIGURE_CLASSES[fig_idx] if fig_idx < len(FIGURE_CLASSES) else f"CLASS_{fig_idx}"
        direction_raw = DIRECTION_CLASSES[dir_idx] if figure_raw != "EMPTY" and dir_idx < len(DIRECTION_CLASSES) else "NONE"

        return {
            "success": True,
            "figure_code": figure_raw,
            "figure_name": FIGURE_VIETNAMESE.get(figure_raw, figure_raw),
            "direction_code": direction_raw,
            "direction_name": DIR_VIETNAMESE.get(direction_raw, direction_raw),
            "confidence": round(prob_fig, 4)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# --- ENDPOINT YOLOv5 ẢNH ---
@app.post("/predict/image", response_model=ImagePredictResponse)
async def predict_image_endpoint(
    file: UploadFile = File(...), model: Optional[str] = Query(None),
    conf: Optional[float] = Query(None), iou: Optional[float] = Query(None),
    include_image: bool = Query(False)
):
    raw = await file.read()
    if not raw: raise HTTPException(400, "File rỗng")
    det = await run_in_threadpool(get_detector, model)

    kw = {}
    if conf is not None: kw["conf_thres"] = conf
    if iou is not None: kw["iou_thres"] = iou

    def _infer():
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: return None
        t0 = time.perf_counter()
        results = det.predict_image(img, **kw)
        ms = (time.perf_counter() - t0) * 1000
        annotated_b64 = None
        if include_image:
            annotated = det.draw(img, results)
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok: annotated_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return img, results, ms, annotated_b64

    out = await run_in_threadpool(_infer)
    if out is None: raise HTTPException(400, "Không thể đọc file ảnh")
    img, results, elapsed_ms, annotated_b64 = out

    h, w = img.shape[:2]
    return ImagePredictResponse(
        num_detections=len(results),
        detections=[DetectionItem(**r) for r in results],
        inference_ms=round(elapsed_ms, 2),
        image_width=w, image_height=h,
        annotated_image_b64=annotated_b64,
        model_weights=det.weights,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)