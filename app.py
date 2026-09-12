"""
MSV Scanner - AI Anomaly Detector
----------------------------------
Upload a scan image. The app checks:

  1) DENSITY  - is intensity roughly consistent with each grid cell's local
                neighborhood, or does a cell stand out relative to the cells
                around it (local, neighbor-relative z-score)?
  2) PATTERN  - is the texture/edge pattern continuous, or does it break
                sharply in some regions (grid + edge-density comparison)?
  3) EMPTY    - is there basically no item in the frame at all (very flat,
                low-edge image)?

Design notes (learned from real MSV trailer scans):
  - Trailer wheel/axle clusters always sit in the bottom-left and
    bottom-right corners of the scan and are legitimately much denser than
    the rest of the frame. They are expected hardware, not cargo, so they
    are excluded from scoring via an "axle exclusion" zone rather than
    being allowed to trigger false ABNORMAL calls.
  - Density is scored with a LOCAL, neighbor-relative z-score (cell vs. the
    mean/std of its immediate neighbors) rather than one global threshold
    for the whole image. A single global threshold is unreliable because an
    empty trailer (low overall variation) and a fully loaded trailer
    (naturally high, texture-driven variation) need very different absolute
    thresholds; comparing each cell only to its immediate neighborhood
    keeps sensitivity consistent regardless of how loaded the trailer is.

Verdict logic:
  EMPTY     -> almost no edges and almost no intensity variation at all
  ABNORMAL  -> any (non-excluded) grid cell flagged by density or pattern
  NORMAL    -> otherwise

Abnormal cells are drawn as colored boxes on the image:
  red     = density anomaly
  orange  = pattern break
  magenta = both
  gray outline = axle/wheel zone (excluded from scoring, shown for reference)

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

# Floor added to the local neighbor std so that a perfectly flat/uniform
# neighborhood doesn't become hyper-sensitive to tiny noise. Found by
# testing against real empty + loaded scan photos.
DENSITY_STD_FLOOR = 3.0


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


def compute_cell_means(gray, grid_size):
    h, w = gray.shape
    gh, gw = grid_cell_size((h, w), grid_size)
    means = np.zeros((grid_size, grid_size))
    for r in range(grid_size):
        for c in range(grid_size):
            block = gray[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw]
            means[r, c] = block.mean() if block.size else 0.0
    return means


# ----------------------------------------------------------------------
# Axle / wheel exclusion zone
# ----------------------------------------------------------------------
def make_axle_exclusion_mask(grid_size, corner_size):
    """
    Trailer wheels/axles sit in the bottom-left and bottom-right corners of
    the scan. Mark a corner_size x corner_size block in each bottom corner
    as excluded from anomaly scoring (they're expected hardware, not cargo).
    corner_size=0 disables exclusion entirely.
    """
    mask = np.zeros((grid_size, grid_size), dtype=bool)
    cs = min(corner_size, grid_size)
    if cs > 0:
        mask[grid_size - cs:, :cs] = True          # bottom-left
        mask[grid_size - cs:, grid_size - cs:] = True  # bottom-right
    return mask


# ----------------------------------------------------------------------
# 1) Density analysis (local, neighbor-relative)
# ----------------------------------------------------------------------
def analyze_density(gray, grid_size=8, z_thresh=3.0, exclusion_mask=None):
    """
    For each non-excluded grid cell, compare its mean intensity to the
    mean/std of its immediate (up to 8) neighbors, using a robust z-score.
    Cells inside the exclusion zone are never flagged and are not used as
    neighbor context for other cells (so a real anomaly next to a wheel
    isn't judged against the wheel's density).
    """
    means = compute_cell_means(gray, grid_size)
    if exclusion_mask is None:
        exclusion_mask = np.zeros((grid_size, grid_size), dtype=bool)

    abnormal = np.zeros((grid_size, grid_size), dtype=bool)
    z_map = np.zeros((grid_size, grid_size))

    for r in range(grid_size):
        for c in range(grid_size):
            if exclusion_mask[r, c]:
                continue
            neighbors = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size and not exclusion_mask[nr, nc]:
                        neighbors.append(means[nr, nc])
            if len(neighbors) >= 3:
                n_mean = np.mean(neighbors)
                n_std = np.std(neighbors) + DENSITY_STD_FLOOR
                z = (means[r, c] - n_mean) / n_std
                z_map[r, c] = z
                if abs(z) > z_thresh:
                    abnormal[r, c] = True

    return means, z_map, abnormal


# ----------------------------------------------------------------------
# 2) Pattern / continuity analysis
# ----------------------------------------------------------------------
def analyze_pattern(gray, grid_size=8, diff_thresh=0.15, exclusion_mask=None):
    """
    Run edge detection, compute the edge density of each grid cell, and
    flag any non-excluded cell whose edge density differs sharply from its
    neighbors (a "broken pattern").
    """
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape
    gh, gw = grid_cell_size((h, w), grid_size)
    if exclusion_mask is None:
        exclusion_mask = np.zeros((grid_size, grid_size), dtype=bool)

    ratios = np.zeros((grid_size, grid_size))
    for r in range(grid_size):
        for c in range(grid_size):
            block = edges[r * gh:(r + 1) * gh, c * gw:(c + 1) * gw]
            ratios[r, c] = np.count_nonzero(block) / block.size if block.size else 0.0

    abnormal = np.zeros((grid_size, grid_size), dtype=bool)
    for r in range(grid_size):
        for c in range(grid_size):
            if exclusion_mask[r, c]:
                continue
            neighbors = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size and not exclusion_mask[nr, nc]:
                        neighbors.append(ratios[nr, nc])
            if neighbors and abs(ratios[r, c] - np.mean(neighbors)) > diff_thresh:
                abnormal[r, c] = True

    return edges, ratios, abnormal


# ----------------------------------------------------------------------
# 3) Empty-frame check
# ----------------------------------------------------------------------
def is_empty(gray, edges, edge_ratio_thresh=0.01, std_thresh=3.0):
    edge_density = np.count_nonzero(edges) / edges.size
    intensity_std = gray.std()
    return edge_density < edge_ratio_thresh and intensity_std < std_thresh


# ----------------------------------------------------------------------
# Draw abnormal cells (and the axle exclusion zone) on the image
# ----------------------------------------------------------------------
def draw_annotations(rgb_image, grid_size, density_mask, pattern_mask, exclusion_mask):
    out = rgb_image.copy()
    h, w, _ = out.shape
    gh, gw = h // grid_size, w // grid_size
    for r in range(grid_size):
        for c in range(grid_size):
            x1, y1 = c * gw, r * gh
            x2, y2 = (c + 1) * gw, (r + 1) * gh
            if exclusion_mask[r, c]:
                cv2.rectangle(out, (x1, y1), (x2, y2), (160, 160, 160), 1)
                continue
            d, p = density_mask[r, c], pattern_mask[r, c]
            if not (d or p):
                continue
            color = (255, 0, 255) if (d and p) else ((255, 0, 0) if d else (255, 165, 0))
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
    "Upload a scan image. The system checks local density consistency and "
    "pattern continuity to flag abnormal regions, or reports EMPTY if no "
    "item is present. Wheel/axle corners are excluded from scoring."
)

with st.sidebar:
    st.header("Settings")
    grid_size = st.slider("Grid size (N x N)", 4, 16, 8)
    axle_corner_size = st.slider(
        "Axle/wheel exclusion (grid cells from each bottom corner)", 0, 4, 2,
        help="Wheels/axles always sit in the bottom-left and bottom-right corners "
             "of the scan. Cells inside this zone are shown but never scored.",
    )
    density_z_thresh = st.slider(
        "Density sensitivity (local z-score)", 1.5, 5.0, 3.0, 0.1,
        help="How far a cell's density can differ from its immediate neighbors "
             "before it's flagged. Lower = more sensitive.",
    )
    pattern_thresh = st.slider("Pattern-break sensitivity", 0.05, 0.5, 0.15, 0.01)
    empty_edge_thresh = st.slider("Empty: edge-density threshold", 0.001, 0.05, 0.01, 0.001)
    empty_std_thresh = st.slider("Empty: intensity-std threshold", 1.0, 15.0, 3.0, 0.5)
    st.divider()
    api_key = st.text_input("Groq API Key", type="password", value=get_default_api_key())
    use_ai_explain = st.checkbox("Get AI explanation (Groq vision)", value=False)

uploaded_file = st.file_uploader("Upload scan image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file:
    rgb, bgr, gray = load_image(uploaded_file)
    exclusion_mask = make_axle_exclusion_mask(grid_size, axle_corner_size)

    density_means, density_z, density_mask = analyze_density(
        gray, grid_size, density_z_thresh, exclusion_mask
    )
    edges, pattern_ratios, pattern_mask = analyze_pattern(
        gray, grid_size, pattern_thresh, exclusion_mask
    )
    empty_flag = is_empty(gray, edges, empty_edge_thresh, empty_std_thresh)

    if empty_flag:
        verdict = "EMPTY"
    elif density_mask.any() or pattern_mask.any():
        verdict = "ABNORMAL"
    else:
        verdict = "NORMAL"

    annotated = draw_annotations(rgb, grid_size, density_mask, pattern_mask, exclusion_mask)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(rgb, use_container_width=True)
    with col2:
        st.subheader("Analyzed")
        st.image(annotated, use_container_width=True)

    if verdict == "NORMAL":
        st.success("✅ Verdict: NORMAL — consistent density and continuous pattern.")
    elif verdict == "ABNORMAL":
        n_density = int(density_mask.sum())
        n_pattern = int(pattern_mask.sum())
        st.error(f"⚠️ Verdict: ABNORMAL — {n_density} density-flagged cell(s), {n_pattern} pattern-break cell(s).")
    else:
        st.warning("⬜ Verdict: EMPTY — no item detected in the scan.")

    with st.expander("Details / metrics"):
        st.write(f"Grid: {grid_size} x {grid_size}  ·  Axle exclusion: {axle_corner_size}-cell corners "
                 f"({int(exclusion_mask.sum())} cells excluded)")
        st.write(f"Density z-threshold: {density_z_thresh}  ·  cells flagged: {int(density_mask.sum())}")
        st.write(f"Pattern-break threshold: {pattern_thresh}  ·  cells flagged: {int(pattern_mask.sum())}")
        st.write(f"Edge density (overall): {np.count_nonzero(edges) / edges.size:.4f}")
        st.write(f"Intensity std (overall): {gray.std():.2f}")
        st.write(
            "Legend: 🔴 red = density anomaly · 🟠 orange = pattern break · "
            "🟣 magenta = both · ⬜ gray outline = axle/wheel zone (not scored)"
        )

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
                            f"density_abnormal_cells={int(density_mask.sum())}, "
                            f"pattern_abnormal_cells={int(pattern_mask.sum())}, "
                            f"axle_excluded_cells={int(exclusion_mask.sum())}"
                        )
                        explanation = groq_explain(client, model, annotated, verdict, stats_summary)
                        st.subheader("AI Explanation")
                        st.write(explanation)
                except Exception as e:
                    st.error(f"Groq API error: {e}")
else:
    st.info("Upload an image to begin analysis.")
