"""
Shogi board recognition pipeline.
Model A: YOLOv5-seg board corner detection
Model B: CNN piece type + direction classification
"""

from __future__ import annotations

import os
import sys
import glob
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIGURE_CLASSES = [
    "BISHOP", "BISHOP_PROM", "EMPTY", "GOLD", "KING",
    "KNIGHT", "KNIGHT_PROM", "LANCE", "LANCE_PROM",
    "PAWN", "PAWN_PROM", "ROOK", "ROOK_PROM", "SILVER", "SILVER_PROM",
]
DIRECTION_CLASSES = ["UP", "DOWN"]

# Short labels for drawing
FIGURE_SHORT = {
    "BISHOP": "BI", "BISHOP_PROM": "B+", "EMPTY": ".",
    "GOLD": "GO", "KING": "KI", "KNIGHT": "KN", "KNIGHT_PROM": "N+",
    "LANCE": "LA", "LANCE_PROM": "L+", "PAWN": "PA", "PAWN_PROM": "P+",
    "ROOK": "RO", "ROOK_PROM": "R+", "SILVER": "SI", "SILVER_PROM": "S+",
}

# Colors BGR for each piece family
PIECE_COLORS = {
    "KING": (0, 0, 220),
    "GOLD": (0, 165, 255),
    "SILVER": (200, 100, 0),
    "KNIGHT": (180, 0, 180),
    "LANCE": (0, 180, 180),
    "BISHOP": (0, 200, 0),
    "ROOK": (220, 50, 50),
    "PAWN": (100, 100, 100),
    "EMPTY": (180, 180, 180),
}


def _color_for(figure: str) -> Tuple[int, int, int]:
    base = figure.replace("_PROM", "")
    return PIECE_COLORS.get(base, (0, 255, 255))


# ---------------------------------------------------------------------------
# Model B architecture (must match training)
# ---------------------------------------------------------------------------
class MixedClassifier(nn.Module):
    def __init__(self, num_figures: int, num_directions: int = 2):
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


class PadToSquare:
    def __init__(self, fill: int = 0):
        self.fill = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        size = max(w, h)
        new_img = Image.new(img.mode, (size, size), color=self.fill)
        new_img.paste(img, ((size - w) // 2, (size - h) // 2))
        return new_img


TRANSFORM = transforms.Compose([
    PadToSquare(fill=0),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]],
        dtype=np.float32,
    )


def expand_corners(corners: np.ndarray, scale_x: float = 1.06, scale_y: float = 1.12) -> np.ndarray:
    center = np.mean(corners, axis=0)
    expanded = []
    for pt in corners:
        dx = (pt[0] - center[0]) * scale_x
        dy = (pt[1] - center[1]) * scale_y
        expanded.append([center[0] + dx, center[1] + dy])
    return np.array(expanded, dtype=np.float32)


def warp_board(image: np.ndarray, corners: np.ndarray, out_size: int = 900) -> Tuple[np.ndarray, np.ndarray]:
    """Return warped image and the perspective matrix H (src->dst)."""
    dst = np.array(
        [[0, 0], [out_size, 0], [out_size, out_size], [0, out_size]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, H, (out_size, out_size))
    return warped, H


def split_into_cells(
    warped: np.ndarray,
    margins: Tuple[int, int, int, int] = (35, 35, 35, 35),
    pad_sides: int = 6,
    n: int = 9,
) -> Dict[Tuple[int, int], Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """
    Returns dict (row, col) -> (cell_image, (x1, y1, x2, y2)) in warped coords.
    """
    mt, mb, ml, mr = margins
    h, w = warped.shape[:2]
    grid_w = w - ml - mr
    grid_h = h - mt - mb
    cell_w = grid_w / n
    cell_h = grid_h / n

    cells = {}
    for row in range(n):
        for col in range(n):
            x1 = ml + col * cell_w
            y1 = mt + row * cell_h
            x2 = ml + (col + 1) * cell_w
            y2 = mt + (row + 1) * cell_h

            if row <= 3:
                pad_top, pad_bottom = 8, 6
            else:
                pad_top, pad_bottom = 22, 6

            crop_x1 = max(0, int(x1 - pad_sides))
            crop_x2 = min(w, int(x2 + pad_sides))
            crop_y1 = max(0, int(y1 - pad_top))
            crop_y2 = min(h, int(y2 + pad_bottom))

            # Bounding box without pad (tight cell) for drawing
            box = (int(x1), int(y1), int(x2), int(y2))
            cells[(row, col)] = (warped[crop_y1:crop_y2, crop_x1:crop_x2], box)
    return cells


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _ensure_yolov5_repo(root: Optional[str] = None) -> str:
    """Clone classic YOLOv5 repo if missing (needed for YOLOv5-seg weights)."""
    if root is None:
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov5")
    if not os.path.isdir(root):
        print(f"Cloning ultralytics/yolov5 into {root} ...")
        code = os.system(f'git clone --depth 1 https://github.com/ultralytics/yolov5.git "{root}"')
        if code != 0 or not os.path.isdir(root):
            raise RuntimeError(
                "Không clone được yolov5. Hãy chạy thủ công:\n"
                f'  git clone https://github.com/ultralytics/yolov5.git "{root}"'
            )
    return root


def load_model_a(weights_path: str, device: str = "cpu"):
    """
    Load classic YOLOv5-seg weights (trained with github.com/ultralytics/yolov5).
    NOT compatible with the ultralytics (YOLOv8) package.
    """
    weights_path = os.path.abspath(weights_path)
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Không tìm thấy: {weights_path}")

    yolov5_root = _ensure_yolov5_repo()
    if yolov5_root not in sys.path:
        sys.path.insert(0, yolov5_root)

    # DetectMultiBackend handles yolov5 detection + segmentation checkpoints
    from models.common import DetectMultiBackend

    model = DetectMultiBackend(weights_path, device=torch.device(device), dnn=False, data=None, fp16=False)
    model.eval()
    model.warped_device = device  # stash for predict
    return model


def load_model_b(weights_path: str, device: str = "cpu") -> MixedClassifier:
    model = MixedClassifier(num_figures=len(FIGURE_CLASSES)).to(device)
    # weights_only=True may fail on older torch; try both
    try:
        state = torch.load(weights_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def detect_board_corners_yolo(model_a, image_bgr: np.ndarray, conf: float = 0.4) -> np.ndarray:
    """
    Run classic YOLOv5-seg Model A.
    Return 4 ordered corners (TL, TR, BR, BL) in pixel coords.
    """
    import torch.nn.functional as F
    from utils.general import non_max_suppression, scale_boxes
    from utils.segment.general import process_mask

    h0, w0 = image_bgr.shape[:2]
    device = next(model_a.model.parameters()).device if hasattr(model_a, "model") else torch.device("cpu")

    # Letterbox-style resize to model stride (same as yolov5)
    stride = int(model_a.stride) if hasattr(model_a, "stride") else 32
    img_size = 960
    r = min(img_size / h0, img_size / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = img_size - new_unpad[0], img_size - new_unpad[1]
    dw /= 2
    dh /= 2
    img = cv2.resize(image_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    img_in = img[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img_in = np.ascontiguousarray(img_in)
    img_t = torch.from_numpy(img_in).to(device).float() / 255.0
    if img_t.ndimension() == 3:
        img_t = img_t.unsqueeze(0)

    with torch.no_grad():
        pred = model_a(img_t)
        # DetectMultiBackend seg output: (predictions, proto)
        if isinstance(pred, (list, tuple)) and len(pred) == 2:
            det_out, proto = pred
        else:
            det_out, proto = pred, None

        det = non_max_suppression(
            det_out, conf_thres=conf, iou_thres=0.45, classes=None, agnostic=False, max_det=10, nm=32
        )

    def _fallback():
        inset = 0.05
        return order_corners(np.array([
            [w0 * inset, h0 * inset],
            [w0 * (1 - inset), h0 * inset],
            [w0 * (1 - inset), h0 * (1 - inset)],
            [w0 * inset, h0 * (1 - inset)],
        ], dtype=np.float32))

    if det is None or len(det) == 0 or det[0] is None or len(det[0]) == 0:
        return _fallback()

    d = det[0]  # (n, 6+nm)
    # Pick highest confidence detection
    best = d[d[:, 4].argmax()]
    # xyxy in letterboxed space
    xyxy = best[:4].cpu().numpy()

    # Try mask polygon if proto available
    poly = None
    if proto is not None and best.shape[0] > 6:
        try:
            masks = process_mask(proto, best[6:].unsqueeze(0), best[:4].unsqueeze(0), img_t.shape[2:], upsample=True)
            mask = masks[0].cpu().numpy()
            mask_u8 = (mask > 0.5).astype(np.uint8) * 255
            # Map mask from letterbox size back to original
            mh, mw = mask_u8.shape
            # Remove letterbox padding then scale
            top_i, left_i = top, left
            mask_crop = mask_u8[top_i:mh - bottom if bottom > 0 else mh, left_i:mw - right if right > 0 else mw]
            if mask_crop.size > 0:
                mask_orig = cv2.resize(mask_crop, (w0, h0), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(mask_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(cnt.astype(np.float32))
                    return order_corners(cv2.boxPoints(rect))
        except Exception:
            poly = None

    # Fallback: scale xyxy box to original image, use as rectangle corners
    # Undo letterbox
    xyxy[0] -= left
    xyxy[2] -= left
    xyxy[1] -= top
    xyxy[3] -= top
    xyxy[:4] /= r
    x1, y1, x2, y2 = xyxy
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w0 - 1, x2), min(h0 - 1, y2)
    return order_corners(np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2],
    ], dtype=np.float32))


def predict_cell(model_b: MixedClassifier, cell_bgr: np.ndarray, device: str) -> Tuple[str, Optional[str], float]:
    if cell_bgr is None or cell_bgr.size == 0:
        return "EMPTY", None, 0.0
    img_rgb = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2RGB)
    tensor = TRANSFORM(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_fig, pred_dir = model_b(tensor)
        fig_probs = torch.softmax(pred_fig, dim=1)
        fig_idx = fig_probs.argmax(1).item()
        conf = fig_probs[0, fig_idx].item()
        dir_idx = pred_dir.argmax(1).item()
    figure = FIGURE_CLASSES[fig_idx]
    direction = DIRECTION_CLASSES[dir_idx] if figure != "EMPTY" else None
    return figure, direction, conf


# ---------------------------------------------------------------------------
# Full analysis + drawing
# ---------------------------------------------------------------------------
def analyze_board(
    image_bgr: np.ndarray,
    model_a,
    model_b: MixedClassifier,
    device: str = "cpu",
    margins: Tuple[int, int, int, int] = (35, 35, 35, 35),
    out_size: int = 900,
) -> dict:
    corners = detect_board_corners_yolo(model_a, image_bgr)
    corners = expand_corners(corners, scale_x=1.06, scale_y=1.12)
    warped, H = warp_board(image_bgr, corners, out_size=out_size)
    cells = split_into_cells(warped, margins=margins)

    board_state = {}
    for (row, col), (cell_img, box) in cells.items():
        figure, direction, conf = predict_cell(model_b, cell_img, device)
        board_state[(row, col)] = {
            "figure": figure,
            "direction": direction,
            "conf": conf,
            "box": box,  # on warped image
        }

    return {
        "board_state": board_state,
        "warped": warped,
        "corners": corners,
        "H": H,
        "out_size": out_size,
    }


def draw_boxes_on_warped(warped: np.ndarray, board_state: dict, show_empty: bool = False) -> np.ndarray:
    canvas = warped.copy()
    for (row, col), info in board_state.items():
        figure = info["figure"]
        if figure == "EMPTY" and not show_empty:
            continue
        x1, y1, x2, y2 = info["box"]
        color = _color_for(figure)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = FIGURE_SHORT.get(figure, figure[:2])
        if info["direction"] == "UP":
            label += "^"
        elif info["direction"] == "DOWN":
            label += "v"

        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            canvas, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return canvas


def draw_boxes_on_original(
    image_bgr: np.ndarray,
    board_state: dict,
    H: np.ndarray,
    out_size: int,
    show_empty: bool = False,
) -> np.ndarray:
    """Project cell boxes back to original image via inverse homography."""
    canvas = image_bgr.copy()
    H_inv = np.linalg.inv(H)

    for (row, col), info in board_state.items():
        figure = info["figure"]
        if figure == "EMPTY" and not show_empty:
            continue
        x1, y1, x2, y2 = info["box"]
        # 4 corners of cell in warped space
        pts = np.array([
            [x1, y1], [x2, y1], [x2, y2], [x1, y2],
        ], dtype=np.float32).reshape(-1, 1, 2)
        pts_orig = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2).astype(np.int32)

        color = _color_for(figure)
        cv2.polylines(canvas, [pts_orig], isClosed=True, color=color, thickness=2)

        # Label near top-left of projected box
        lx, ly = int(pts_orig[0, 0]), int(pts_orig[0, 1])
        label = FIGURE_SHORT.get(figure, figure[:2])
        if info["direction"] == "UP":
            label += "^"
        elif info["direction"] == "DOWN":
            label += "v"
        cv2.putText(
            canvas, label, (lx, max(ly - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
        )
    return canvas


def board_state_to_text(board_state: dict, n: int = 9) -> str:
    lines = []
    for row in range(n):
        cells = []
        for col in range(n):
            info = board_state[(row, col)]
            fig = info["figure"]
            if fig == "EMPTY":
                cells.append(" .  ")
            else:
                mark = "^" if info["direction"] == "UP" else "v"
                cells.append(f"{FIGURE_SHORT.get(fig, fig[:2])}{mark}")
        lines.append(" ".join(f"{c:>5}" for c in cells))
    return "\n".join(lines)
