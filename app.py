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
import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from pipeline import (
    load_model_a,
    load_model_b,
    analyze_board,
    draw_boxes_on_warped,
    draw_boxes_on_original,
    board_state_to_text,
    FIGURE_CLASSES,
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
def get_models(path_a: str, path_b: str, device: str):
    model_a = load_model_a(path_a, device=device)
    model_b = load_model_b(path_b, device=device)
    return model_a, model_b


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

    # Also search runs folders
    for p in Path("/content/runs").glob("board_corner_seg*/weights/best.pt") if Path("/content/runs").exists() else []:
        candidates_a.append(p)

    path_a = next((str(p) for p in candidates_a if p.exists()), None)
    path_b = next((str(p) for p in candidates_b if p.exists()), None)
    return path_a, path_b


# ---------------------------------------------------------------------------
# Sidebar – model setup
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Cấu hình")

st.sidebar.markdown("### Model files")
st.sidebar.caption("Upload hoặc đặt file vào thư mục `models/`")

up_a = st.sidebar.file_uploader("Model A – best.pt", type=["pt"], key="up_a")
up_b = st.sidebar.file_uploader("Model B – model_b_figure_direction.pt", type=["pt"], key="up_b")

# Save uploaded models
if up_a is not None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / "best.pt"
    dest.write_bytes(up_a.read())
    st.sidebar.success(f"Đã lưu {dest.name}")

if up_b is not None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / "model_b_figure_direction.pt"
    dest.write_bytes(up_b.read())
    st.sidebar.success(f"Đã lưu {dest.name}")

path_a, path_b = find_model_files()

if path_a:
    st.sidebar.info(f"Model A: `{path_a}`")
else:
    st.sidebar.error("Chưa có best.pt")

if path_b:
    st.sidebar.info(f"Model B: `{path_b}`")
else:
    st.sidebar.error("Chưa có model_b_figure_direction.pt")

device = st.sidebar.selectbox("Device", ["cuda", "cpu"], index=0 if __import__("torch").cuda.is_available() else 1)

st.sidebar.markdown("---")
st.sidebar.markdown("### Tham số crop")
margin = st.sidebar.slider("Margins (lề lưới)", 10, 80, 35, 5)
pad_note = st.sidebar.caption("Padding theo hàng đã tối ưu trong code (top/bottom khác nhau).")
show_empty = st.sidebar.checkbox("Hiện ô trống", value=False)
conf_threshold = st.sidebar.slider("Ngưỡng conf hiển thị", 0.0, 1.0, 0.0, 0.05)

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("♟️ Shogi Board Recognition")
st.markdown(
    "Upload ảnh bàn cờ Shogi → AI phát hiện bàn, nhận diện từng quân và vẽ **bounding box**."
)

uploaded = st.file_uploader(
    "Chọn ảnh bàn cờ",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Ảnh chụp bàn cờ từ góc nghiêng hoặc thẳng đều được.",
)

col_run, col_info = st.columns([1, 3])
run_btn = col_run.button("🔍 Nhận diện", type="primary", disabled=uploaded is None)

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Không đọc được ảnh.")
        st.stop()

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    st.subheader("Ảnh gốc")
    st.image(image_rgb, use_container_width=True)

    if run_btn:
        if not path_a or not path_b:
            st.error("Thiếu model. Hãy upload `best.pt` và `model_b_figure_direction.pt` ở sidebar.")
            st.stop()

        with st.spinner("Đang load model & nhận diện..."):
            try:
                model_a, model_b = get_models(path_a, path_b, device)
            except Exception as e:
                st.error(f"Lỗi load model: {e}")
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

        # Filter by confidence if needed
        if conf_threshold > 0:
            for k, v in board_state.items():
                if v["conf"] < conf_threshold and v["figure"] != "EMPTY":
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

        # Summary counts
        from collections import Counter
        counts = Counter(
            v["figure"] for v in board_state.values() if v["figure"] != "EMPTY"
        )
        if counts:
            st.markdown("**Số lượng quân phát hiện:**")
            st.write(dict(counts))

        # Download buttons
        st.subheader("Tải kết quả")
        d1, d2 = st.columns(2)

        def _encode_png(bgr: np.ndarray) -> bytes:
            ok, buf = cv2.imencode(".png", bgr)
            return buf.tobytes() if ok else b""

        d1.download_button(
            "⬇️ Warped + boxes (PNG)",
            data=_encode_png(warped_boxed),
            file_name="shogi_warped_boxes.png",
            mime="image/png",
        )
        d2.download_button(
            "⬇️ Original + boxes (PNG)",
            data=_encode_png(original_boxed),
            file_name="shogi_original_boxes.png",
            mime="image/png",
        )

        # SFEN-like simple export (optional)
        st.download_button(
            "⬇️ Board state (TXT)",
            data=text.encode("utf-8"),
            file_name="board_state.txt",
            mime="text/plain",
        )

else:
    st.info("👆 Upload ảnh bàn cờ để bắt đầu.")
    st.markdown(
        """
### Hướng dẫn nhanh
1. Đặt hoặc upload **best.pt** (Model A) và **model_b_figure_direction.pt** (Model B) ở sidebar trái.
2. Upload ảnh bàn cờ.
3. Bấm **Nhận diện**.

### Pipeline
1. **Model A** (YOLOv5-seg) → tìm 4 góc bàn cờ  
2. Perspective warp → bàn vuông 9×9  
3. Cắt từng ô + **Model B** → loại quân + hướng (UP/DOWN)  
4. Vẽ bounding box màu theo loại quân
"""
    )
