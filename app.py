import base64
import io
import json
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from groq import Groq


st.set_page_config(
    page_title="MSV Scanner Image Anomaly Detector",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 MSV Scanner — Normal / Abnormal / EMPTY")
st.caption(
    "Hybrid inspection prototype: computer-vision density/pattern analysis + Groq vision review."
)


# -----------------------------
# Configuration
# -----------------------------
GRID_ROWS = 12
GRID_COLS = 12

# Higher values make the detector less sensitive.
DENSITY_Z_THRESHOLD = 2.2
PATTERN_Z_THRESHOLD = 2.4
MIN_COMPONENT_AREA_RATIO = 0.0008

VISION_MODEL = "qwen/qwen3.6-27b"


# -----------------------------
# Helper functions
# -----------------------------
def get_groq_client():
    """Read the Groq key from Streamlit secrets or an environment variable."""
    api_key = None

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass

    api_key = api_key or os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def image_to_bgr(pil_image):
    """Convert uploaded PIL image to OpenCV BGR."""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def normalize_gray(gray):
    """Improve contrast while keeping the original structure."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def foreground_mask(gray):
    """
    Estimate whether meaningful scanner content exists.
    This is intentionally conservative and should be calibrated with real MSV images.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu gives a first separation of object/background.
    _, mask1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Scanner images can have either bright or dark objects, so also test
    # deviation from the global background.
    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    deviation = cv2.absdiff(gray, bg)
    _, mask2 = cv2.threshold(deviation, 12, 255, cv2.THRESH_BINARY)

    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def is_empty(gray):
    """Return empty decision and foreground ratio."""
    mask = foreground_mask(gray)
    ratio = float(np.count_nonzero(mask)) / mask.size

    # A very small amount of foreground is considered empty.
    return ratio < 0.015, ratio, mask


def tile_features(gray, rows=GRID_ROWS, cols=GRID_COLS):
    """
    Calculate robust density and local-pattern features for each tile.

    Density:
      median grayscale value and robust spread.

    Pattern:
      average horizontal and vertical gradient strength.
    """
    h, w = gray.shape
    density = np.zeros((rows, cols), dtype=np.float32)
    pattern = np.zeros((rows, cols), dtype=np.float32)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    for r in range(rows):
        y1 = int(r * h / rows)
        y2 = int((r + 1) * h / rows)

        for c in range(cols):
            x1 = int(c * w / cols)
            x2 = int((c + 1) * w / cols)

            tile = gray[y1:y2, x1:x2]
            tile_grad = grad[y1:y2, x1:x2]

            density[r, c] = np.median(tile)
            pattern[r, c] = np.mean(tile_grad)

    return density, pattern


def robust_z_score(matrix):
    """Median/MAD z-score; much less affected by one abnormal tile."""
    med = np.median(matrix)
    mad = np.median(np.abs(matrix - med))

    if mad < 1e-6:
        std = np.std(matrix)
        if std < 1e-6:
            return np.zeros_like(matrix)
        return (matrix - med) / (std + 1e-6)

    return 0.6745 * (matrix - med) / (mad + 1e-6)


def pattern_break_map(gray, rows=GRID_ROWS, cols=GRID_COLS):
    """
    Detect local continuity breaks.

    A tile is suspicious when its gradient energy is substantially different
    from neighboring tiles in the same row/column.
    """
    _, pattern = tile_features(gray, rows, cols)

    neighbor_difference = np.zeros_like(pattern)

    for r in range(rows):
        for c in range(cols):
            neighbors = []

            if r > 0:
                neighbors.append(pattern[r - 1, c])
            if r < rows - 1:
                neighbors.append(pattern[r + 1, c])
            if c > 0:
                neighbors.append(pattern[r, c - 1])
            if c < cols - 1:
                neighbors.append(pattern[r, c + 1])

            neighbor_difference[r, c] = abs(
                pattern[r, c] - float(np.mean(neighbors))
            )

    return neighbor_difference


def build_anomaly_map(gray):
    """
    Combine density anomalies and pattern-continuity anomalies.
    """
    density, pattern = tile_features(gray)
    density_z = np.abs(robust_z_score(density))
    pattern_z = np.abs(robust_z_score(pattern))

    continuity = pattern_break_map(gray)
    continuity_z = np.abs(robust_z_score(continuity))

    density_anomaly = density_z >= DENSITY_Z_THRESHOLD
    pattern_anomaly = (
        (pattern_z >= PATTERN_Z_THRESHOLD)
        | (continuity_z >= PATTERN_Z_THRESHOLD)
    )

    combined = density_anomaly | pattern_anomaly

    # Ignore isolated tiny detections.
    tile_mask = (combined.astype(np.uint8) * 255)
    kernel = np.ones((3, 3), np.uint8)
    tile_mask = cv2.morphologyEx(tile_mask, cv2.MORPH_CLOSE, kernel)

    return {
        "density": density,
        "pattern": pattern,
        "density_z": density_z,
        "pattern_z": pattern_z,
        "continuity_z": continuity_z,
        "density_anomaly": density_anomaly,
        "pattern_anomaly": pattern_anomaly,
        "combined": tile_mask > 0,
    }


def tile_map_to_pixel_mask(tile_map, image_shape):
    """Expand the tile-level anomaly map to image pixels."""
    rows, cols = tile_map.shape
    h, w = image_shape[:2]

    small = (tile_map.astype(np.uint8) * 255)
    full = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Smooth boundaries while retaining the detected region.
    kernel = np.ones((11, 11), np.uint8)
    full = cv2.morphologyEx(full, cv2.MORPH_CLOSE, kernel)

    return full


def remove_tiny_components(mask):
    """Remove extremely small connected components."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )

    cleaned = np.zeros_like(mask, dtype=np.uint8)
    image_area = mask.shape[0] * mask.shape[1]
    min_area = max(25, int(image_area * MIN_COMPONENT_AREA_RATIO))

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def make_overlay(bgr, anomaly_mask):
    """Draw detected abnormal areas in red and add bounding boxes."""
    overlay = bgr.copy()
    red = np.zeros_like(bgr)
    red[:, :, 2] = 255

    alpha = 0.42
    mask_bool = anomaly_mask > 0

    overlay[mask_bool] = cv2.addWeighted(
        bgr[mask_bool], 1 - alpha, red[mask_bool], alpha, 0
    )

    contours, _ = cv2.findContours(
        anomaly_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h >= 25:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 3)
            boxes.append({"x": x, "y": y, "width": w, "height": h})

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), boxes


def ask_groq_about_image(pil_image, local_result):
    """
    Groq is used as a second opinion / explanation layer.
    The quantitative anomaly detector remains local and deterministic.
    """
    client = get_groq_client()
    if client is None:
        return {
            "available": False,
            "text": "Groq key not configured. Local computer-vision result is shown."
        }

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    prompt = f"""
You are assisting with an MSV/X-ray scanner image inspection prototype.

The local computer-vision system calculated:
- Local status: {local_result["status"]}
- Foreground ratio: {local_result["foreground_ratio"]:.4f}
- Density-anomaly tiles: {local_result["density_tiles"]}
- Pattern-anomaly tiles: {local_result["pattern_tiles"]}
- Total anomaly tiles: {local_result["total_anomaly_tiles"]}

Review the uploaded scanner image visually.

Important:
1. Do NOT claim that visual inspection alone proves a real security threat.
2. Give a concise second-opinion assessment.
3. Look for major changes in apparent density/attenuation and broken/repeated-pattern regions.
4. If there is no visible cargo/item, say EMPTY.
5. If suspicious regions are visible, describe their approximate location such as upper-left, center, lower-right.
6. State that the result is a prototype and should be calibrated/validated using labelled MSV scanner images.

Return:
Status: NORMAL / ABNORMAL / EMPTY / UNCERTAIN
Reason: ...
Approximate suspicious regions: ...
"""

    try:
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_completion_tokens=500,
        )

        return {
            "available": True,
            "text": response.choices[0].message.content
        }

    except Exception as exc:
        return {
            "available": False,
            "text": f"Groq vision review failed: {exc}"
        }


# -----------------------------
# UI
# -----------------------------
with st.sidebar:
    st.header("Detection settings")
    st.write("These are prototype thresholds. Calibrate them with real labelled scanner images.")

    density_threshold = st.slider(
        "Density sensitivity",
        min_value=1.0,
        max_value=5.0,
        value=float(DENSITY_Z_THRESHOLD),
        step=0.1,
    )

    pattern_threshold = st.slider(
        "Pattern sensitivity",
        min_value=1.0,
        max_value=5.0,
        value=float(PATTERN_Z_THRESHOLD),
        step=0.1,
    )

    st.info(
        "For a production MSV system, use a labelled dataset of NORMAL, ABNORMAL "
        "and EMPTY images and train/validate a dedicated model."
    )

uploaded = st.file_uploader(
    "Upload an MSV scanner image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded is None:
    st.markdown(
        """
### How this prototype works

1. Upload a scanner image.
2. The app checks whether meaningful content is present.
3. It divides the image into a grid.
4. It measures local grayscale/density and gradient/pattern features.
5. It detects unusual density and continuity changes.
6. It marks suspicious regions in red.
7. Groq's vision model provides a second visual opinion.
        """
    )
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except Exception as exc:
    st.error(f"Could not read image: {exc}")
    st.stop()

bgr = image_to_bgr(image)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
gray = normalize_gray(gray)

empty, foreground_ratio, _ = is_empty(gray)

if empty:
    local_status = "EMPTY"
    anomaly_mask = np.zeros_like(gray, dtype=np.uint8)
    boxes = []
    density_tiles = 0
    pattern_tiles = 0
    total_tiles = 0
else:
    # Use the sidebar thresholds for this run.
    density, pattern = tile_features(gray)
    density_z = np.abs(robust_z_score(density))
    pattern_z = np.abs(robust_z_score(pattern))
    continuity = pattern_break_map(gray)
    continuity_z = np.abs(robust_z_score(continuity))

    density_anomaly = density_z >= density_threshold
    pattern_anomaly = (
        (pattern_z >= pattern_threshold)
        | (continuity_z >= pattern_threshold)
    )

    combined = density_anomaly | pattern_anomaly

    # Clean tile map and expand it to pixels.
    tile_mask = (combined.astype(np.uint8) * 255)
    tile_mask = cv2.morphologyEx(
        tile_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )

    anomaly_mask = tile_map_to_pixel_mask(
        tile_mask > 0, bgr.shape
    )
    anomaly_mask = remove_tiny_components(anomaly_mask)

    density_tiles = int(np.count_nonzero(density_anomaly))
    pattern_tiles = int(np.count_nonzero(pattern_anomaly))
    total_tiles = int(np.count_nonzero(combined))

    local_status = "ABNORMAL" if total_tiles > 0 else "NORMAL"

overlay_rgb, boxes = make_overlay(bgr, anomaly_mask)

local_result = {
    "status": local_status,
    "foreground_ratio": foreground_ratio,
    "density_tiles": density_tiles,
    "pattern_tiles": pattern_tiles,
    "total_anomaly_tiles": total_tiles,
}

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original image")
    st.image(image, use_container_width=True)

with col2:
    st.subheader("AI/CV inspection result")
    st.image(overlay_rgb, use_container_width=True)

if local_status == "EMPTY":
    st.success("🟢 EMPTY — no significant item/structure was detected.")
elif local_status == "ABNORMAL":
    st.error(f"🔴 ABNORMAL — {len(boxes)} suspicious region(s) marked in red.")
else:
    st.success("🟢 NORMAL — no significant density/pattern anomaly was detected.")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Local status", local_status)
m2.metric("Foreground ratio", f"{foreground_ratio:.3f}")
m3.metric("Density anomaly tiles", density_tiles)
m4.metric("Pattern anomaly tiles", pattern_tiles)

if boxes:
    st.subheader("Marked abnormal regions")
    st.json(boxes)

with st.spinner("Getting Groq vision second opinion..."):
    groq_result = ask_groq_about_image(image, local_result)

st.subheader("Groq vision second opinion")
st.write(groq_result["text"])

with st.expander("Technical details"):
    st.write(
        "This prototype uses robust tile-level statistics. "
        "It is not a trained X-ray security classifier."
    )
    st.json(local_result)

st.warning(
    "IMPORTANT: Do not use this prototype as the sole safety/security decision "
    "for live cargo screening. Real MSV/X-ray inspection requires labelled data, "
    "calibration, validation, false-negative testing, and qualified human review."
)
