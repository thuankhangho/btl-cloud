"""
Shogi Board Recognition – Streamlit Web App

Usage:
  1. Place model files in ./models/
       - best.pt
       - model_b_figure_direction.pt
  2. pip install -r requirements.txt
  3. streamlit run app.py
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageGrab

from pipeline import (
    load_model_a,
    load_model_b,
    analyze_board,
    draw_boxes_on_warped,
    draw_boxes_on_original,
    board_state_to_text,
    predict_cell,
    FIGURE_SHORT,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
DEFAULT_A = MODELS_DIR / "best.pt"
DEFAULT_B = MODELS_DIR / "model_b_figure_direction.pt"

st.set_page_config(
    page_title="Shogi Board Recognition",
    page_icon="♟️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------
@st.cache_resource
def get_model_a(path_a: str, device: str):
    return load_model_a(path_a, device=device)


@st.cache_resource
def get_model_b(path_b: str, device: str):
    return load_model_b(path_b, device=device)


def find_model_files():
    """Search common locations for model weights."""
    candidates_a = [
        DEFAULT_A,
        APP_DIR / "best.pt",
        Path("/content/uploaded_models/best.pt"),
        Path("/content/best.pt"),
        Path("/content/runs/board_corner_seg_uploaded/weights/best.pt"),
    ]
    candidates_b = [
        DEFAULT_B,
        APP_DIR / "model_b_figure_direction.pt",
        Path("/content/uploaded_models/model_b_figure_direction.pt"),
        Path("/content/model_b_figure_direction.pt"),
    ]

    if Path("/content/runs").exists():
        for p in Path("/content/runs").glob("board_corner_seg*/weights/best.pt"):
            candidates_a.append(p)

    path_a = next((str(p) for p in candidates_a if p.exists()), None)
    path_b = next((str(p) for p in candidates_b if p.exists()), None)
    return path_a, path_b


def encode_png(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    return buf.tobytes() if ok else b""


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Cấu hình")

st.sidebar.markdown("### Model files")
st.sidebar.caption("Upload hoặc đặt file vào thư mục `models/`")

up_a = st.sidebar.file_uploader("Model A – best.pt", type=["pt"], key="up_a")
up_b = st.sidebar.file_uploader("Model B – model_b_figure_direction.pt", type=["pt"], key="up_b")

if up_a is not None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / "best.pt"
    dest.write_bytes(up_a.read())
    st.sidebar.success(f"Đã lưu {dest.name}")
    st.cache_resource.clear()

if up_b is not None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / "model_b_figure_direction.pt"
    dest.write_bytes(up_b.read())
    st.sidebar.success(f"Đã lưu {dest.name}")
    st.cache_resource.clear()

path_a, path_b = find_model_files()

if path_a:
    st.sidebar.info(f"Model A: `{path_a}`")
else:
    st.sidebar.warning("Chưa có best.pt (cần cho chế độ cả bàn)")

if path_b:
    st.sidebar.info(f"Model B: `{path_b}`")
else:
    st.sidebar.error("Chưa có model_b_figure_direction.pt")

device_options = ["cpu"]
if torch.cuda.is_available():
    device_options.insert(0, "cuda")
device = st.sidebar.selectbox("Device", device_options, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### Chế độ")
mode = st.sidebar.radio(
    "Chọn chế độ nhận diện",
    ["Cả bàn cờ (9×9)", "Một quân cờ"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Tham số (chế độ bàn cờ)")
margin = st.sidebar.slider("Margins (lề lưới)", 5, 80, 35, 5)
show_empty = st.sidebar.checkbox("Hiện ô trống", value=False)
conf_threshold = st.sidebar.slider("Ngưỡng conf hiển thị", 0.0, 1.0, 0.0, 0.05)

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("♟️ Shogi Board Recognition")
st.markdown(
    "Upload / dán ảnh → AI nhận diện quân cờ Shogi và vẽ **bounding box**."
)

# ---- Input: file upload + clipboard ----
if "image_bgr" not in st.session_state:
    st.session_state.image_bgr = None
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0

col_up, col_paste = st.columns([3, 1])

with col_up:
    uploaded = st.file_uploader(
        "Chọn ảnh",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="Ảnh bàn cờ hoặc ảnh 1 quân cờ.",
        key=f"uploader_{st.session_state.upload_key}",
    )

with col_paste:
    st.write("")
    st.write("")
    paste_clicked = st.button("📋 Dán từ clipboard", use_container_width=True)

if paste_clicked:
    clip = ImageGrab.grabclipboard()
    if clip is None:
        st.warning("Clipboard không có ảnh. Hãy Copy ảnh (Ctrl+C) rồi bấm lại.")
    else:
        st.session_state.image_bgr = cv2.cvtColor(
            np.array(clip.convert("RGB")), cv2.COLOR_RGB2BGR
        )
        st.success("Đã lấy ảnh từ clipboard.")

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    decoded = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if decoded is not None:
        st.session_state.image_bgr = decoded

image_bgr = st.session_state.image_bgr

col_run, col_clear, _ = st.columns([1, 1, 4])
run_btn = col_run.button(
    "🔍 Nhận diện", type="primary", disabled=(image_bgr is None)
)
if col_clear.button("🗑️ Xóa ảnh"):
    st.session_state.image_bgr = None
    st.session_state.upload_key += 1  # reset uploader
    st.rerun()

# ---- Preview + run ----
if image_bgr is None:
    st.info("👆 Upload ảnh hoặc dán từ clipboard để bắt đầu.")
    st.markdown(
        """
### Hướng dẫn
1. Đặt / upload **best.pt** (Model A) và **model_b_figure_direction.pt** (Model B) ở sidebar.
2. Chọn chế độ:
   - **Cả bàn cờ (9×9)** — cần cả 2 model
   - **Một quân cờ** — chỉ cần Model B
3. Upload hoặc **Dán từ clipboard** (Ctrl+C ảnh trước).
4. Bấm **Nhận diện**.
"""
    )
    st.stop()

image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
st.subheader("Ảnh đầu vào")
if mode == "Một quân cờ":
    st.image(image_rgb, width=240)
else:
    st.image(image_rgb, use_container_width=True)

if not run_btn:
    st.caption("Bấm **Nhận diện** để chạy AI.")
    st.stop()

# =====================================================================
# MODE: Một quân cờ
# =====================================================================
if mode == "Một quân cờ":
    if not path_b:
        st.error("Thiếu `model_b_figure_direction.pt`. Upload ở sidebar.")
        st.stop()

    with st.spinner("Đang nhận diện 1 quân..."):
        try:
            model_b = get_model_b(path_b, device)
            figure, direction, conf = predict_cell(model_b, image_bgr, device)
        except Exception as e:
            st.error(f"Lỗi: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.stop()

    st.subheader("Kết quả — 1 quân")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(image_rgb, width=200)
    with c2:
        short = FIGURE_SHORT.get(figure, figure[:2])
        arrow = ""
        if direction == "UP":
            arrow = " ^"
        elif direction == "DOWN":
            arrow = " v"

        st.success(f"**{figure}**  →  `{short}{arrow}`")
        st.metric("Confidence", f"{conf:.1%}")
        if direction:
            st.write(f"Hướng: **{direction}**")
        else:
            st.write("Hướng: _(ô trống / không áp dụng)_")

    st.stop()

# =====================================================================
# MODE: Cả bàn cờ (9×9)
# =====================================================================
if not path_a or not path_b:
    st.error("Chế độ bàn cờ cần cả `best.pt` và `model_b_figure_direction.pt`.")
    st.stop()

with st.spinner("Đang load model & nhận diện bàn cờ..."):
    try:
        model_a = get_model_a(path_a, device)
        model_b = get_model_b(path_b, device)
    except Exception as e:
        st.error(f"Lỗi load model: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

    try:
        result = analyze_board(
            image_bgr,
            model_a,
            model_b,
            device=device,
            margins=(margin, margin, margin, margin),
        )
    except Exception as e:
        st.error(f"Lỗi nhận diện: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

board_state = result["board_state"]
warped = result["warped"]
H = result["H"]

# Filter low-confidence predictions
if conf_threshold > 0:
    for k, v in list(board_state.items()):
        if v["figure"] != "EMPTY" and v["conf"] < conf_threshold:
            board_state[k] = {**v, "figure": "EMPTY", "direction": None}

warped_boxed = draw_boxes_on_warped(warped, board_state, show_empty=show_empty)
original_boxed = draw_boxes_on_original(
    image_bgr, board_state, H, result["out_size"], show_empty=show_empty
)

st.subheader("Kết quả")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Bàn đã nắn phẳng + bounding box**")
    st.image(cv2.cvtColor(warped_boxed, cv2.COLOR_BGR2RGB), use_container_width=True)
with c2:
    st.markdown("**Ảnh gốc + bounding box**")
    st.image(cv2.cvtColor(original_boxed, cv2.COLOR_BGR2RGB), use_container_width=True)

st.subheader("Trạng thái bàn cờ (text)")
text = board_state_to_text(board_state)
st.code(text, language=None)

counts = Counter(v["figure"] for v in board_state.values() if v["figure"] != "EMPTY")
if counts:
    st.markdown("**Số lượng quân phát hiện:**")
    st.write(dict(counts))

st.subheader("Tải kết quả")
d1, d2, d3 = st.columns(3)
d1.download_button(
    "⬇️ Warped + boxes (PNG)",
    data=encode_png(warped_boxed),
    file_name="shogi_warped_boxes.png",
    mime="image/png",
)
d2.download_button(
    "⬇️ Original + boxes (PNG)",
    data=encode_png(original_boxed),
    file_name="shogi_original_boxes.png",
    mime="image/png",
)
d3.download_button(
    "⬇️ Board state (TXT)",
    data=text.encode("utf-8"),
    file_name="board_state.txt",
    mime="text/plain",
)
