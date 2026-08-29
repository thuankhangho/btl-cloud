"""
YOLOv5 Object Detection Pipeline - config-driven & dataset-agnostic.
"""

from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import yaml


def load_config(path: Union[str, Path]) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def default_config() -> dict:
    return {
        "model": {
            "weights": "models/best.pt",
            "device": "auto",
            "img_size": 640,
            "conf_thres": 0.25,
            "iou_thres": 0.45,
            "max_det": 300,
            "classes": None,
            "half": False,
        },
        "names": None,
        "data_yaml": None,
        "yolov5_repo": None,
    }


def apply_env_overrides(cfg: dict) -> dict:
    m = cfg.setdefault("model", {})
    env = os.getenv
    if env("YOLOV5_WEIGHTS"):
        m["weights"] = env("YOLOV5_WEIGHTS")
    if env("YOLOV5_DEVICE"):
        m["device"] = env("YOLOV5_DEVICE")
    if env("YOLOV5_IMG_SIZE"):
        m["img_size"] = int(env("YOLOV5_IMG_SIZE"))
    if env("YOLOV5_CONF"):
        m["conf_thres"] = float(env("YOLOV5_CONF"))
    if env("YOLOV5_IOU"):
        m["iou_thres"] = float(env("YOLOV5_IOU"))
    if env("YOLOV5_DATA_YAML"):
        cfg["data_yaml"] = env("YOLOV5_DATA_YAML")
    return cfg


def names_from_data_yaml(path: Union[str, Path]) -> Dict[int, str]:
    data = load_config(path)
    names = data.get("names")
    if isinstance(names, (list, tuple)):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    raise ValueError(f"No 'names' entry in data yaml: {path}")


def ensure_yolov5_repo(root: Optional[str] = None) -> str:
    if root is None:
        root = os.getenv("YOLOV5_REPO") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "yolov5"
        )
    if not os.path.isdir(root):
        print(f"[pipeline] Cloning ultralytics/yolov5 -> {root}")
        code = os.system(
            f'git clone --depth 1 https://github.com/ultralytics/yolov5.git "{root}"'
        )
        if code != 0 or not os.path.isdir(root):
            raise RuntimeError(
                "Cannot clone yolov5. Run manually:\n"
                f'  git clone https://github.com/ultralytics/yolov5.git "{root}"\n'
                f"  pip install -r {root}/requirements.txt"
            )
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _resolve_torch_device(device: Union[str, torch.device] = "auto") -> torch.device:
    s = str(device)
    if s in ("", "auto"):
        s = "cuda" if torch.cuda.is_available() else "cpu"
    if s == "cuda":
        s = "cuda:0"
    elif s.isdigit():
        s = f"cuda:{s}"
    return torch.device(s)


def load_model(
    weights_path: str,
    device: Union[str, torch.device] = "auto",
    yolov5_repo: Optional[str] = None,
    half: bool = False,
    dnn: bool = False,
    data: Optional[str] = None,
    warmup: bool = True,
    img_size: int = 640,
):
    weights_path = os.path.abspath(weights_path)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    ensure_yolov5_repo(yolov5_repo)
    from models.common import DetectMultiBackend

    dev = _resolve_torch_device(device)
    use_half = bool(half) and dev.type == "cuda"

    model = DetectMultiBackend(weights_path, device=dev, dnn=dnn, data=data, fp16=use_half)
    try:
        model.eval()
    except Exception:
        pass
    if warmup:
        try:
            model.warmup(imgsz=(1, 3, img_size, img_size))
        except Exception:
            pass
    return model


def get_class_names(model, override: Optional[Dict[int, str]] = None) -> Dict[int, str]:
    if override:
        return {int(k): str(v) for k, v in override.items()}
    names = getattr(model, "names", None)
    if names is None and hasattr(model, "model"):
        names = getattr(model.model, "names", None)
    if names is None:
        return {}
    if isinstance(names, (list, tuple)):
        return {i: str(n) for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def letterbox(
    im: np.ndarray,
    new_shape: Union[int, Tuple[int, int]] = 640,
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    stride: int = 32,
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = dw % stride, dh % stride
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)


def scale_coords(
    img1_shape, coords: np.ndarray, img0_shape, ratio_pad=None
) -> np.ndarray:
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )
    else:
        gain, pad = ratio_pad[0], ratio_pad[1]

    coords = coords.copy()
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])
    coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])
    return coords


def predict_image(
    model,
    image_bgr: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    img_size: int = 640,
    max_det: int = 300,
    classes: Optional[List[int]] = None,
    names: Optional[Dict[int, str]] = None,
) -> List[Dict[str, Any]]:
    try:
        from utils.general import non_max_suppression
    except ImportError as e:
        raise ImportError("Không tìm thấy yolov5/utils. Đảm bảo đã clone repo yolov5.") from e

    h0, w0 = image_bgr.shape[:2]
    device = getattr(model, "device", None)
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    fp16 = bool(getattr(model, "fp16", False))

    stride = getattr(model, "stride", None)
    try:
        stride = int(max(stride)) if stride is not None else 32
    except TypeError:
        stride = int(stride) if stride is not None else 32

    img, ratio, (dw, dh) = letterbox(image_bgr, new_shape=img_size, auto=True, stride=stride)
    img_in = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    img_t = torch.from_numpy(img_in).to(device)
    img_t = img_t.half() if fp16 else img_t.float()
    img_t /= 255.0
    if img_t.ndimension() == 3:
        img_t = img_t.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_t)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        det = non_max_suppression(
            pred, conf_thres=conf_thres, iou_thres=iou_thres,
            classes=classes, agnostic=False, max_det=max_det,
        )

    class_names = get_class_names(model, override=names)
    results: List[Dict[str, Any]] = []
    if not det or det[0] is None or len(det[0]) == 0:
        return results

    d = det[0].float().cpu().numpy()
    boxes = scale_coords(img_t.shape[2:], d[:, :4], (h0, w0), ratio_pad=(ratio, (dw, dh)))

    for i in range(len(d)):
        x1, y1, x2, y2 = boxes[i].tolist()
        results.append({
            "class_id": int(d[i, 5]),
            "class_name": class_names.get(int(d[i, 5]), str(int(d[i, 5]))),
            "confidence": float(d[i, 4]),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })
    return results


_PALETTE = [
    (0, 0, 255), (0, 165, 255), (0, 255, 0), (255, 0, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 255, 0),
    (0, 128, 255), (128, 255, 0), (255, 128, 0), (0, 0, 128),
]


def draw_detections(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
    thickness: int = 2,
) -> np.ndarray:
    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        color = _PALETTE[int(det["class_id"]) % len(_PALETTE)]
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        ty = y1 - 4
        if ty - th - 4 < 0:
            ty = y1 + th + 4

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        cv2.rectangle(canvas, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
        cv2.putText(canvas, label, (x1 + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def predict_video(
    model,
    video_path: str,
    output_path: Optional[str] = None,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    img_size: int = 640,
    max_det: int = 300,
    classes: Optional[List[int]] = None,
    names: Optional[Dict[int, str]] = None,
    save_frames: bool = True,
    frame_step: int = 1,
    max_frames: Optional[int] = None,
    include_detections: bool = True,
) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if output_path and save_frames and width > 0 and height > 0:
        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        ext = os.path.splitext(output_path)[1].lower()
        candidates = ("mp4v", "XVID") if ext in (".mp4", ".mov") else ("XVID", "mp4v")
        writer_fps = fps / frame_step if frame_step > 1 else fps
        for cc in candidates:
            w = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*cc),
                                writer_fps, (width, height))
            if w.isOpened():
                writer = w
                break
            w.release()

    all_dets: List[List[Dict[str, Any]]] = []
    class_counts: Dict[str, int] = {}
    t0 = time.time()
    read_idx = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if read_idx % frame_step != 0:
            read_idx += 1
            continue
        read_idx += 1

        dets = predict_image(
            model, frame,
            conf_thres=conf_thres, iou_thres=iou_thres,
            img_size=img_size, max_det=max_det,
            classes=classes, names=names,
        )
        processed += 1
        for d in dets:
            class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1
        if include_detections:
            all_dets.append(dets)
        if writer is not None:
            writer.write(draw_detections(frame, dets))
        if processed % 200 == 0:
            print(f"[pipeline] video: {processed} frames, {time.time() - t0:.1f}s")
        if max_frames is not None and processed >= max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()

    return {
        "num_frames": processed,
        "total_frames_in_video": total if total > 0 else None,
        "fps": float(fps),
        "frame_step": frame_step,
        "detections_per_frame": all_dets,
        "class_counts": class_counts,
        "output_path": output_path if writer is not None else None,
        "elapsed_sec": round(time.time() - t0, 2),
    }


class Detector:
    def __init__(self, model, cfg: dict, names: Optional[Dict[int, str]] = None):
        self.model = model
        self.cfg = cfg
        m = cfg.get("model", {})
        self.img_size = int(m.get("img_size", 640))
        self.conf_thres = float(m.get("conf_thres", 0.25))
        self.iou_thres = float(m.get("iou_thres", 0.45))
        self.max_det = int(m.get("max_det", 300))
        self.classes = m.get("classes")
        self.names = names if names is not None else get_class_names(model)

    @classmethod
    def from_config(cls, config_path: Union[str, Path, None] = None, **overrides):
        cfg = default_config()
        if config_path and Path(config_path).is_file():
            deep_merge(cfg, load_config(config_path))
        elif config_path:
            print(f"[pipeline] NOTE: config '{config_path}' not found -> using defaults")
        apply_env_overrides(cfg)

        m = cfg.setdefault("model", {})
        for k, v in overrides.items():
            if v is None:
                continue
            if k in ("weights", "device", "img_size", "conf_thres", "iou_thres",
                     "max_det", "classes", "half"):
                m[k] = v
            else:
                cfg[k] = v
        return cls.from_cfg(cfg)

    @classmethod
    def from_cfg(cls, cfg: dict) -> "Detector":
        cfg = copy.deepcopy(cfg)
        m = cfg.setdefault("model", {})

        data_yaml = cfg.get("data_yaml")
        if data_yaml:
            if not os.path.isabs(data_yaml):
                alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_yaml)
                if os.path.isfile(alt):
                    data_yaml = alt
                    cfg["data_yaml"] = alt
            if not os.path.isfile(data_yaml):
                data_yaml = None

        model = load_model(
            m.get("weights") or "models/best.pt",
            device=m.get("device", "auto"),
            yolov5_repo=cfg.get("yolov5_repo"),
            half=bool(m.get("half", False)),
            data=data_yaml,
            img_size=int(m.get("img_size", 640)),
        )

        names_override = cfg.get("names") or (names_from_data_yaml(data_yaml) if data_yaml else None)
        return cls(model, cfg, names=get_class_names(model, override=names_override))

    @property
    def weights(self) -> str:
        return self.cfg.get("model", {}).get("weights", "")

    def info(self) -> dict:
        backend = "pytorch"
        for attr, name in (("onnx", "onnx"), ("engine", "tensorrt"), ("xml", "openvino"),
                           ("saved_model", "tf-saved_model"), ("pb", "tf-graphdef"),
                           ("tflite", "tflite"), ("edgetpu", "edgetpu"),
                           ("coreml", "coreml"), ("triton", "triton")):
            if getattr(self.model, attr, False):
                backend = name
                break
        device = getattr(self.model, "device", None)
        return {
            "weights": self.weights,
            "backend": backend,
            "device": str(device) if device is not None else "unknown",
            "img_size": self.img_size,
            "conf_thres": self.conf_thres,
            "iou_thres": self.iou_thres,
            "max_det": self.max_det,
            "num_classes": len(self.names),
            "names": self.names,
        }

    def predict_image(self, image_bgr: np.ndarray, **kw) -> List[Dict[str, Any]]:
        return predict_image(
            self.model, image_bgr,
            conf_thres=kw.get("conf_thres", self.conf_thres),
            iou_thres=kw.get("iou_thres", self.iou_thres),
            img_size=kw.get("img_size", self.img_size),
            max_det=kw.get("max_det", self.max_det),
            classes=kw.get("classes", self.classes),
            names=kw.get("names", self.names),
        )

    def predict_video(self, video_path: str, output_path: Optional[str] = None, **kw):
        return predict_video(
            self.model, video_path, output_path=output_path,
            conf_thres=kw.get("conf_thres", self.conf_thres),
            iou_thres=kw.get("iou_thres", self.iou_thres),
            img_size=kw.get("img_size", self.img_size),
            max_det=kw.get("max_det", self.max_det),
            classes=kw.get("classes", self.classes),
            names=kw.get("names", self.names),
            save_frames=kw.get("save_frames", True),
            frame_step=kw.get("frame_step", 1),
            max_frames=kw.get("max_frames", None),
            include_detections=kw.get("include_detections", True),
        )

    def draw(self, image_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        return draw_detections(image_bgr, detections)