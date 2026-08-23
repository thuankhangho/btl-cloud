"""
YOLOv5 & PyTorch Object Detection – FastAPI REST service.

Endpoints
---------
  GET  /                – Trả về file giao diện index.html
  GET  /health          – health check (+ CUDA availability)
  GET  /models          – list available weights & config profiles
  GET  /models/info     – model metadata (class names → proves dataset swap)
  POST /models/upload   – upload a trained .pt checkpoint into models/
  POST /predict/image   – detect on an uploaded image → JSON (+ optional
                          base64 annotated image or direct PNG)
  POST /predict/video   – frame-by-frame detection → JSON summary (+ optional
                          annotated video in outputs/)
  POST /predict/piece   - Phân loại 1 ô cờ Shogi bằng PyTorch Model
  GET  /outputs/{file}  – download a generated annotated video
  POST /cache/clear     – drop cached models (after swapping weight files)
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
from PIL import Image

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pipeline import (Detector, apply_env_overrides, default_config,
                      deep_merge, load_config)

# ---------------------------------------------------------------------------
# Paths & app setup
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
CONFIGS_DIR = APP_DIR / "configs"
OUTPUTS_DIR = APP_DIR / "outputs"
DEFAULT_CONFIG = CONFIGS_DIR / "default.yaml"

for d in (MODELS_DIR, CONFIGS_DIR, OUTPUTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Shogi AI Dashboard (YOLOv5 + PyTorch)",
    description=(
        "REST API for Object Detection & Image Classification."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cho phép mọi trang web gọi API này
    allow_credentials=True,
    allow_methods=["*"], # Cho phép mọi lệnh POST, GET...
    allow_headers=["*"],
)


# ===========================================================================
# 1. KHỞI TẠO MODEL B (PYTORCH - ĐỌC QUÂN CỜ)
# ===========================================================================
class MixedClassifier(nn.Module):
    def __init__(self, num_figures, num_directions=2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc_figure = nn.Linear(128, num_figures)
        self.fc_direction = nn.Linear(128, num_directions)

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.fc_figure(feat), self.fc_direction(feat)

device_b = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_b = None
try:
    # 8 loại quân cờ, 3 hướng (Trống, Mình, Địch)
    model_b = MixedClassifier(num_figures=8, num_directions=3).to(device_b)
    model_b.load_state_dict(torch.load(MODELS_DIR / "piece_detection.pt", map_location=device_b, weights_only=True))
    model_b.eval()
    print("Đã nạp thành công model piece_detection.pt")
except Exception as e:
    print("Cảnh báo model (Chưa nạp được hoặc sai tham số):", e)

transform_b = T.Compose([
    T.Resize((100, 100)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

FIGURE_MAP = {0: 'Trống', 1: 'Tướng', 2: 'Xe', 3: 'Vàng', 4: 'Bạc', 5: 'Mã', 6: 'Hương xa', 7: 'Tốt'}
DIR_MAP = {0: 'Trống', 1: 'Của mình', 2: 'Của địch'}


# ---------------------------------------------------------------------------
# 2. KHỞI TẠO MODEL A (YOLOV5) & CACHE
# ---------------------------------------------------------------------------
_detectors: Dict[str, Detector] = {}
_cache_lock = threading.Lock()
MAX_CACHED_MODELS = int(os.getenv("MAX_CACHED_MODELS", "4"))

def _resolve_config(model_name: Optional[str] = None) -> dict:
    cfg = default_config()

    cfg_path = Path(os.getenv("API_CONFIG", str(DEFAULT_CONFIG)))
    if cfg_path.is_file():
        deep_merge(cfg, load_config(cfg_path))

    apply_env_overrides(cfg)

    if model_name:
        profile = CONFIGS_DIR / f"{model_name}.yaml"
        if profile.is_file():
            deep_merge(cfg, load_config(profile))
        else:
            candidate = MODELS_DIR / model_name
            if candidate.is_file():
                cfg.setdefault("model", {})["weights"] = str(candidate)
            elif Path(model_name).is_file():
                cfg.setdefault("model", {})["weights"] = str(Path(model_name).resolve())
            else:
                raise HTTPException(
                    404,
                    f"Model '{model_name}' not found. Expected a yaml profile in "
                    f"configs/ or a weights file in models/. See GET /models.",
                )

    m = cfg.setdefault("model", {})

    weights = m.get("weights", "")
    if weights and not os.path.isabs(weights):
        alt = MODELS_DIR / Path(weights).name
        if alt.is_file():
            m["weights"] = str(alt)
        elif (APP_DIR / weights).is_file():
            m["weights"] = str(APP_DIR / weights)

    if m.get("device") in (None, "", "auto"):
        m["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg

def get_detector(model_name: Optional[str] = None) -> Detector:
    cfg = _resolve_config(model_name)
    key = json.dumps(cfg, sort_keys=True, default=str)

    det = _detectors.get(key)
    if det is not None:
        return det

    weights = cfg["model"].get("weights", "")
    if not weights or not os.path.isfile(weights):
        raise HTTPException(
            404,
            f"Weights not found: '{weights}'. Train on Colab, put best.pt into "
            "models/, or upload via POST /models/upload.",
        )
    try:
        det = Detector.from_cfg(cfg)
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}") from e

    with _cache_lock:
        _detectors.setdefault(key, det)
        while len(_detectors) > MAX_CACHED_MODELS: 
            _detectors.pop(next(iter(_detectors)))
        det = _detectors[key]
    return det


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class DetectionItem(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2] in pixels")

class ImagePredictResponse(BaseModel):
    num_detections: int
    detections: List[DetectionItem]
    inference_ms: float
    image_width: int
    image_height: int
    annotated_image_b64: Optional[str] = Field(None)
    model_weights: str

class VideoPredictResponse(BaseModel):
    num_frames: int
    total_frames_in_video: Optional[int] = None
    fps: float
    frame_step: int = 1
    elapsed_sec: float
    total_detections: int
    class_counts: Dict[str, int] = Field(default_factory=dict)
    detections_per_frame: List[List[DetectionItem]]
    output_video: Optional[str] = None
    output_video_url: Optional[str] = None
    model_weights: str

def _parse_classes(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or not raw.strip():
        return None
    try:
        return [int(x) for x in raw.replace(" ", "").split(",") if x != ""]
    except ValueError:
        raise HTTPException(400, f"Invalid classes filter: {raw!r}")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    """Trả về trực tiếp file giao diện index.html khi người dùng vào thư mục gốc"""
    index_file = APP_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    return {"error": "Không tìm thấy file index.html trong thư mục! Hãy tải file index.html lên Google Drive."}

@app.get("/health")
def health():
    return {"status": "ok", "cuda": torch.cuda.is_available(),
            "cached_models": len(_detectors)}

@app.get("/models")
def list_models():
    return {
        "weights": sorted(p.name for p in MODELS_DIR.glob("*.pt")),
        "profiles": sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml")),
        "usage": "Add ?model=<profile-or-filename> to /predict/* to pick one at runtime",
    }

@app.get("/models/info")
async def model_info(model: Optional[str] = Query(None)):
    det = await run_in_threadpool(get_detector, model)
    info = det.info()
    info["names"] = {str(k): v for k, v in info.get("names", {}).items()}
    return info

@app.post("/models/upload")
async def upload_weights(file: UploadFile = File(...)):
    if not file.filename or not file.filename.endswith(".pt"):
        raise HTTPException(400, "Only .pt files are accepted")
    dest = MODELS_DIR / Path(file.filename).name
    size = 0
    with dest.open("wb") as f:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk: break
            f.write(chunk)
            size += len(chunk)
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Empty file")
    with _cache_lock: _detectors.clear()
    return {"saved": str(dest), "size_bytes": size}

# --- ENDPOINT MODEL B (PYTORCH: ĐỌC QUÂN CỜ) ---
@app.post("/predict/piece")
async def predict_piece_endpoint(file: UploadFile = File(...)):
    if model_b is None:
        raise HTTPException(500, "Model B chưa được nạp.")
    
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert('RGB')
    tensor = transform_b(img).unsqueeze(0).to(device_b)

    with torch.no_grad():
        out_fig, out_dir = model_b(tensor)
        fig_idx = out_fig.argmax(1).item()
        dir_idx = out_dir.argmax(1).item()

    return {
        "success": True,
        "figure": FIGURE_MAP.get(fig_idx, "Không rõ"),
        "direction": DIR_MAP.get(dir_idx, "Không rõ")
    }

# --- ENDPOINT MODEL A (YOLOV5: ẢNH VÀ VIDEO) ---
@app.post("/predict/image", response_model=ImagePredictResponse)
async def predict_image_endpoint(
    file: UploadFile = File(...),
    model: Optional[str] = Query(None),
    conf: Optional[float] = Query(None, ge=0.0, le=1.0),
    iou: Optional[float] = Query(None, ge=0.0, le=1.0),
    classes: Optional[str] = Query(None),
    return_image: bool = Query(False),
    include_image: bool = Query(False),
):
    raw = await file.read()
    if not raw: raise HTTPException(400, "Empty file")

    det = await run_in_threadpool(get_detector, model)

    kw: Dict[str, Any] = {}
    if conf is not None: kw["conf_thres"] = conf
    if iou is not None: kw["iou_thres"] = iou
    cls_ids = _parse_classes(classes)
    if cls_ids is not None: kw["classes"] = cls_ids

    def _infer():
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: return None
        t0 = time.perf_counter()
        results = det.predict_image(img, **kw)
        ms = (time.perf_counter() - t0) * 1000
        annotated_b64, png_bytes = None, None
        if return_image or include_image:
            annotated = det.draw(img, results)
            if include_image:
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok: annotated_b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            if return_image:
                ok, buf = cv2.imencode(".png", annotated)
                if ok: png_bytes = buf.tobytes()
        return img, results, ms, annotated_b64, png_bytes

    out = await run_in_threadpool(_infer)
    if out is None: raise HTTPException(400, "Cannot decode image")
    img, results, elapsed_ms, annotated_b64, png_bytes = out

    if return_image:
        if png_bytes is None: raise HTTPException(500, "Failed to encode annotated image")
        return Response(content=png_bytes, media_type="image/png",
                        headers={"Content-Disposition": 'inline; filename="detections.png"'})

    h, w = img.shape[:2]
    return ImagePredictResponse(
        num_detections=len(results),
        detections=[DetectionItem(**r) for r in results],
        inference_ms=round(elapsed_ms, 2),
        image_width=w,
        image_height=h,
        annotated_image_b64=annotated_b64,
        model_weights=det.weights,
    )

@app.post("/predict/video", response_model=VideoPredictResponse)
async def predict_video_endpoint(
    file: UploadFile = File(...),
    model: Optional[str] = Query(None),
    conf: Optional[float] = Query(None, ge=0.0, le=1.0),
    iou: Optional[float] = Query(None, ge=0.0, le=1.0),
    classes: Optional[str] = Query(None),
    save_video: bool = Query(True),
    frame_step: int = Query(1, ge=1),
    max_frames: Optional[int] = Query(None, ge=1),
    include_detections: bool = Query(True),
):
    suffix = Path(file.filename or "video.mp4").suffix.lower() or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk: break
            tmp.write(chunk)
        tmp_path = tmp.name

    det = await run_in_threadpool(get_detector, model)

    kw: Dict[str, Any] = {}
    if conf is not None: kw["conf_thres"] = conf
    if iou is not None: kw["iou_thres"] = iou
    cls_ids = _parse_classes(classes)
    if cls_ids is not None: kw["classes"] = cls_ids

    out_path = None
    if save_video:
        out_ext = suffix if suffix in {".mp4", ".avi", ".mov"} else ".mp4"
        out_path = str(OUTPUTS_DIR / f"pred_{int(time.time() * 1000)}{out_ext}")

    try:
        summary = await run_in_threadpool(
            det.predict_video, tmp_path, out_path,
            frame_step=frame_step, max_frames=max_frames,
            include_detections=include_detections, **kw,
        )
    finally:
        try: os.unlink(tmp_path)
        except OSError: pass

    frames = [[DetectionItem(**d) for d in fr] for fr in summary["detections_per_frame"]]
    out_name = Path(summary["output_path"]).name if summary.get("output_path") else None

    return VideoPredictResponse(
        num_frames=summary["num_frames"],
        total_frames_in_video=summary.get("total_frames_in_video"),
        fps=summary["fps"],
        frame_step=summary.get("frame_step", 1),
        elapsed_sec=summary["elapsed_sec"],
        total_detections=sum(summary["class_counts"].values()),
        class_counts=summary["class_counts"],
        detections_per_frame=frames,
        output_video=out_name,
        output_video_url=f"/outputs/{out_name}" if out_name else None,
        model_weights=det.weights,
    )

@app.get("/outputs/{filename}")
def download_output(filename: str):
    if Path(filename).name != filename: raise HTTPException(400, "Invalid filename")
    path = OUTPUTS_DIR / filename
    if not path.is_file(): raise HTTPException(404, "File not found")
    ext = path.suffix.lower()
    media = ("video/mp4" if ext == ".mp4" else "video/quicktime" if ext == ".mov" else "video/x-msvideo" if ext == ".avi" else "application/octet-stream")
    return FileResponse(path, media_type=media, filename=filename)

@app.post("/cache/clear")
def clear_model_cache():
    with _cache_lock:
        n = len(_detectors)
        _detectors.clear()
    return {"cleared_models": n}

# ---------------------------------------------------------------------------
# Local entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)