"""
MSV Scanner - AI Anomaly Detector
----------------------------------
Upload a scan image. The app checks:
  1) DENSITY  - is pixel intensity roughly uniform across the image, or are
                some regions unusually darker/lighter (grid + z-score)?
  2) PATTERN  - is the texture/edge pattern continuous, or does it break
                sharply in some regions (grid + edge-density comparison)?
  3) EMPTY    - is there basically no item in the frame at all (very flat,
                low-edge image)?

Verdict logic:
  EMPTY     -> almost no edges and almost no intensity variation at all
  ABNORMAL  -> any grid cell flagged by density or pattern checks
  NORMAL    -> otherwise

Abnormal cells are drawn as colored boxes on the image:
  red     = density anomaly
  orange  = pattern break
  magenta = both

Optionally, the annotated result + stats can be sent to a Groq vision model
for a short plain-language explanation for the operator.
"""

import base64
import io
import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="MSV Anomaly Scanner", layout="wide")


# ----------------------------------------------------------------------
# Image loading
# ----------------------------------------------------------------------
def load_image(uploaded_file, size=(512, 512)):
    """Load an uploaded file into RGB / BGR / grayscale numpy arrays."""
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize(size)
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return rgb, bgr, gray


def grid_cell_size(shape, grid_size):
    h, w = shape
    return h // grid_size, w // grid_size


# ----------------------------------------------------------------------
# 1) Density analysis
# ----------------------------------------------------------------------
def analyze_density(gray, grid_size=8, z_thresh=2.0):
    """
    Split the image into a grid_size x grid_size grid, compute the mean
    intensity ("density") of each cell, then flag any cell whose mean is
    a statistical outlier (z-score) relative to the whole image.
    """
    h, w = gray.shape
    gh, gw = grid_cell_size((h, w), grid_size)

    means = np.zeros((grid_size, grid_size))
    for r in range(grid_size):
        for c in range(grid_size):
            block = gray[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw]
            means[r, c] = block.mean() if block.size else 0.0

    overall_mean = means.mean()
    overall_std = means.std() + 1e-6
    z_scores = (means - overall_mean) / overall_std
    abnormal_mask = np.abs(z_scores) > z_thresh
    return means, z_scores, abnormal_mask, overall_std


# ----------------------------------------------------------------------
# 2) Pattern / continuity analysis
# ----------------------------------------------------------------------
def analyze_pattern(gray, grid_size=8, diff_thresh=0.15):
    """
    Run edge detection, compute the edge density of each grid cell, and
    flag any cell whose edge density differs sharply from its neighbors
    (a "broken pattern").
    """
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape
    gh, gw = grid_cell_size((h, w), grid_size)

    ratios = np.zeros((grid_size, grid_size))
    for r in range(grid_size):
        for c in range(grid_size):
            block = edges[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw]
            ratios[r, c] = np.count_nonzero(block) / block.size if block.size else 0.0

    abnormal_mask = np.zeros((grid_size, grid_size), dtype=bool)
    for r in range(grid_size):
        for c in range(grid_size):
            neighbors = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        neighbors.append(ratios[nr, nc])
            if neighbors and abs(ratios[r, c] - np.mean(neighbors)) > diff_thresh:
                abnormal_mask[r, c] = True

    return edges, ratios, abnormal_mask


# ----------------------------------------------------------------------
# 3) Empty-frame check
# ----------------------------------------------------------------------
def is_empty(gray, edges, edge_ratio_thresh=0.01, std_thresh=3.0):
    edge_density = np.count_nonzero(edges) / edges.size
    intensity_std = gray.std()
    return edge_density < edge_ratio_thresh and intensity_std < std_thresh


# ----------------------------------------------------------------------
# Draw abnormal cells on the image
# ----------------------------------------------------------------------
def draw_annotations(rgb_image, grid_size, density_mask, pattern_mask):
    out = rgb_image.copy()
    h, w, _ = out.shape
    gh, gw = h // grid_size, w // grid_size
    for r in range(grid_size):
        for c in range(grid_size):
            d, p = density_mask[r, c], pattern_mask[r, c]
            if not (d or p):
                continue
            color = (255, 0, 255) if (d and p) else ((255, 0, 0) if d else (255, 165, 0))
            x1, y1 = c * gw, r * gh
            x2, y2 = (c + 1) * gw, (r + 1) * gh
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return out


# ----------------------------------------------------------------------
# Groq vision explanation (optional)
# ----------------------------------------------------------------------
def get_default_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")


def get_groq_client(api_key):
    from groq import Groq
    return Groq(api_key=api_key)


def find_vision_model(client):
    """Groq's list of vision-capable models can change, so look it up live
    instead of hard-coding a model name that may be retired later."""
    try:
        models = client.models.list()
        candidates = [m.id for m in models.data if "vision" in m.id.lower()]
        return candidates[0] if candidates else None
    except Exception:
        return None


def image_to_b64_jpeg(rgb_image):
    pil_img = Image.fromarray(rgb_image)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def groq_explain(client, model, rgb_image, verdict, stats_summary):
    b64 = image_to_b64_jpeg(rgb_image)
    prompt = (
        f"You are assisting an MSV scanner operator. An automated density/pattern "
        f"check labeled this image '{verdict}'. Stats: {stats_summary}. "
        f"In 3-4 short, plain-language sentences, tell the operator what to look at "
        f"and whether this looks like something worth a manual re-check."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        max_tokens=300,
    )
    return resp.choices[0].message.content


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("🔍 MSV Scanner — AI Anomaly Detector")
st.caption(
    "Upload a scan image. The system checks density uniformity and pattern "
    "continuity to flag abnormal regions, or reports EMPTY if no item is present."
)

with st.sidebar:
    st.header("Settings")
    grid_size = st.slider("Grid size (N x N)", 4, 16, 8)
    z_thresh = st.slider("Density sensitivity (z-score)", 1.0, 4.0, 2.0, 0.1)
    pattern_thresh = st.slider("Pattern-break sensitivity", 0.05, 0.5, 0.15, 0.01)
    empty_edge_thresh = st.slider("Empty: edge-density threshold", 0.001, 0.05, 0.01, 0.001)
    empty_std_thresh = st.slider("Empty: intensity-std threshold", 1.0, 15.0, 3.0, 0.5)
    st.divider()
    api_key = st.text_input("Groq API Key", type="password", value=get_default_api_key())
    use_ai_explain = st.checkbox("Get AI explanation (Groq vision)", value=False)

uploaded_file = st.file_uploader("Upload scan image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file:
    rgb, bgr, gray = load_image(uploaded_file)

    density_means, density_z, density_mask, density_std = analyze_density(gray, grid_size, z_thresh)
    edges, pattern_ratios, pattern_mask = analyze_pattern(gray, grid_size, pattern_thresh)
    empty_flag = is_empty(gray, edges, empty_edge_thresh, empty_std_thresh)

    if empty_flag:
        verdict = "EMPTY"
    elif density_mask.any() or pattern_mask.any():
        verdict = "ABNORMAL"
    else:
        verdict = "NORMAL"

    annotated = draw_annotations(rgb, grid_size, density_mask, pattern_mask) if verdict == "ABNORMAL" else rgb

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(rgb, use_container_width=True)
    with col2:
        st.subheader("Analyzed")
        st.image(annotated, use_container_width=True)

    if verdict == "NORMAL":
        st.success("✅ Verdict: NORMAL — uniform density and continuous pattern.")
    elif verdict == "ABNORMAL":
        n_density = int(density_mask.sum())
        n_pattern = int(pattern_mask.sum())
        st.error(f"⚠️ Verdict: ABNORMAL — {n_density} density-flagged cell(s), {n_pattern} pattern-break cell(s).")
    else:
        st.warning("⬜ Verdict: EMPTY — no item detected in the scan.")

    with st.expander("Details / metrics"):
        st.write(f"Grid: {grid_size} x {grid_size}")
        st.write(f"Density std across blocks: {density_std:.2f}")
        st.write(f"Edge density (overall): {np.count_nonzero(edges) / edges.size:.4f}")
        st.write(f"Intensity std (overall): {gray.std():.2f}")
        st.write("Legend: 🔴 red = density anomaly · 🟠 orange = pattern break · 🟣 magenta = both")

    if use_ai_explain:
        if not api_key:
            st.info("Enter a Groq API key in the sidebar (or set it in Streamlit secrets) to use this.")
        else:
            with st.spinner("Contacting Groq vision model..."):
                try:
                    client = get_groq_client(api_key)
                    model = find_vision_model(client)
                    if not model:
                        st.warning(
                            "No vision-capable model was found on this Groq account. "
                            "Check the Groq console for the current vision model name."
                        )
                    else:
                        stats_summary = (
                            f"density_std={density_std:.2f}, "
                            f"density_abnormal_cells={int(density_mask.sum())}, "
                            f"pattern_abnormal_cells={int(pattern_mask.sum())}"
                        )
                        explanation = groq_explain(client, model, annotated, verdict, stats_summary)
                        st.subheader("AI Explanation")
                        st.write(explanation)
                except Exception as e:
                    st.error(f"Groq API error: {e}")
else:
    st.info("Upload an image to begin analysis.")
