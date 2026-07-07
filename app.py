"""
Streamlit app: upload a photo -> classify its color season / subtype.

Run:
    pip install streamlit st-clickable-images plotly
    streamlit run app_draping.py

Required next to this file:
    - season_features.py   (shared color feature extractor)
    - one or more model_*.joblib files (see MODEL_DIR / MODEL_FILES below)
    - palettes/            (palette images, optional)
"""
import io
import base64
from pathlib import Path

import numpy as np
import streamlit as st
import cv2
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────
MODEL_DIR = Path("./models/")

# Display name -> (file, feature_kind). The FIRST entry is the default.
# feature_kind: 'color' uses season_features; 'farl' uses FaRL embeddings.
MODEL_FILES = {
    "Color features (raw)": ("model_color-raw.joblib", "color"),
    "Color features (whitebalance)": ("model_color-vk.joblib",  "color"),
    "FaRL frozen + boosting": ("model_FaRL_frozen_boosting.joblib", "farl"),
    "CLIP + boosting": ("model_CLIP.joblib", "farl"),
    "FaRL frozen + FC": ("model_FaRL_frozen_FC.joblib", "farl"),

}

SEASON_COLORS = {
    "spring": "#cf9077", "summer": "#72928f",
    "autumn": "#c09440", "winter": "#304372",
}

PALETTE_DIR = Path("palettes")


# ──────────────────────────────────────────────────────────────────────────
# Feature backends
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _get_color_extractor():
    """Import the shared color feature extractor once."""
    from season_features import extract_features as _extract
    return _extract


def color_features(img_rgb, variant):
    """Zone color features. variant: 'raw' or 'vk' (von Kries)."""
    extract = _get_color_extractor()
    return extract(img_rgb, use_von_kries=(variant == "vk"))


@st.cache_resource
def _get_farl():
    """Load FaRL (ViT-B/16) once, with its preprocess transform."""
    import torch
    import open_clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16")
    weights = Path("FaRL-Base-Patch16-LAIONFace20M-ep64.pth")
    if weights.exists():
        state = torch.load(weights, map_location="cpu")
        model.load_state_dict(state.get("state_dict", state), strict=False)
    model = model.to(device).eval()
    return torch, model, preprocess, device


def farl_features(img_rgb):
    """FaRL image embedding for one RGB image."""
    torch, model, preprocess, device = _get_farl()
    pil = Image.fromarray(img_rgb)
    x = preprocess(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(x).squeeze().cpu().numpy().astype(np.float32)
    return emb


def extract_for_model(img_rgb, feature_kind, variant="raw"):
    """Route to the correct feature backend. Returns a 1D vector or None."""
    if feature_kind == "color":
        return color_features(img_rgb, variant)
    if feature_kind == "farl":
        return farl_features(img_rgb)
    raise ValueError(f"unknown feature_kind: {feature_kind}")


# ──────────────────────────────────────────────────────────────────────────
# Model loading & inference
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(file_name):
    import joblib
    path = MODEL_DIR / file_name
    if not path.exists():
        return None, f"Model not found: {path}"
    return joblib.load(path), "model loaded"


def classify(bundle, features):
    """Return (season_proba[4], subtype_proba[12]).

    Season probabilities are aggregated from the subtype head so the two
    charts always agree with each other.
    """
    x = np.asarray(features, np.float32).reshape(1, -1)
    sub_proba = bundle["clf_sub"].predict_proba(x)[0]
    season_subs = bundle["SEASON_SUBS"]
    sea_proba = np.array([sub_proba[season_subs[s]].sum() for s in range(4)])
    return sea_proba, sub_proba


# ──────────────────────────────────────────────────────────────────────────
# Small helpers (charts, chin detection, draping, palette parsing)
# ──────────────────────────────────────────────────────────────────────────
def _hbar(labels, values, colors):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(labels) + 0.5))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(0, (max(values) * 1.15) if values else 1)
    ax.set_xlabel("%")
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.01, yi, f"{v:.1f}%", va="center", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _hex_to_rgb(hx):
    hx = hx.lstrip("#")
    return tuple(int(hx[i:i + 2], 16) for i in (0, 2, 4))


def _mesh_landmarks(img_rgb):
    """Run the shared mesh once; return landmarks or None."""
    import mediapipe as mp
    from season_features import _get_shared_mesh
    mesh = _get_shared_mesh()
    res = mesh.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(img_rgb)))
    if not res.face_landmarks:
        return None
    return res.face_landmarks[0]


def find_chin(img_rgb):
    """Return (chin_x, chin_y) — bottom of the chin from landmarks, or None."""
    lm = _mesh_landmarks(img_rgb)
    if lm is None:
        return None
    h, w = img_rgb.shape[:2]
    return int(lm[152].x * w), int(lm[152].y * h)  # 152 = chin tip


def drape_rect(pil_img, chin, color_hex, gap=6):
    """Fill a colored rectangle from (chin + gap) to the bottom, full width."""
    from PIL import ImageDraw
    img = pil_img.convert("RGB").copy()
    w, h = img.size
    cy = (chin[1] + gap) if chin is not None else int(h * 0.75)
    cy = min(cy, h - 1)
    ImageDraw.Draw(img).rectangle([0, cy, w, h], fill=_hex_to_rgb(color_hex))
    return img


def _to_height(pil_img, target_h=280):
    """Scale to a fixed height, keeping aspect ratio."""
    w, h = pil_img.size
    return pil_img.resize((max(1, int(w * target_h / h)), target_h), Image.LANCZOS)


def palette_path(subtype_name):
    season, sub = subtype_name.split("/")
    return PALETTE_DIR / f"{sub}-{season}.webp"


@st.cache_data
def extract_palette_colors(path_str, max_colors=64):
    """Auto-extract hex tile colors from a palette image grid."""
    img = np.array(Image.open(path_str).convert("RGB"))
    h, w = img.shape[:2]
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = mx - mn
    bright = mx
    is_tile = ((sat > 25) | ((bright < 235) & (bright > 30) & (sat <= 25)))

    whiteness = ((bright > 240) & (sat < 12)).mean(axis=1)
    start_y = int(h * 0.35)
    for y in range(int(h * 0.35), int(h * 0.55)):
        if whiteness[y] > 0.6:
            start_y = y
            break

    m = int(min(h, w) * 0.04)
    region = np.zeros((h, w), bool)
    region[start_y:h - m, m:w - m] = True
    tile = is_tile & region
    row = tile[:, m:w - m].mean(axis=1)
    col = tile[start_y:h - m, :].mean(axis=0)

    def bands(profile, thr=0.35, min_len=10):
        on = profile > thr
        out = []
        s = None
        for i, v in enumerate(on):
            if v and s is None:
                s = i
            elif not v and s is not None:
                if i - s >= min_len:
                    out.append((s, i))
                s = None
        if s is not None and len(on) - s >= min_len:
            out.append((s, len(on)))
        return out

    colors = []
    for (y0, y1) in bands(row):
        yc = (y0 + y1) // 2
        for (x0, x1) in bands(col):
            xc = (x0 + x1) // 2
            patch = img[max(0, yc - 4):yc + 4, max(0, xc - 4):xc + 4].reshape(-1, 3)
            c = np.median(patch, 0).astype(int)
            colors.append(tuple(int(v) for v in c))
    return ["#%02x%02x%02x" % c for c in colors[:max_colors]]


@st.cache_data
def crop_faces_collage(path_str):
    """Crop a palette image to its top face collage (above the color grid)."""
    img = Image.open(path_str).convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    r, g, b = arr[..., 0].astype(int), arr[..., 1].astype(int), arr[..., 2].astype(int)
    bright = np.maximum(np.maximum(r, g), b)
    sat = bright - np.minimum(np.minimum(r, g), b)
    whiteness = ((bright > 240) & (sat < 12)).mean(axis=1)
    cut = int(h * 0.45)
    for y in range(int(h * 0.30), int(h * 0.55)):
        if whiteness[y] > 0.6:
            cut = y
            break
    return img.crop((0, 0, w, cut))


# ──────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Color season", layout="centered")

# --- theme: Ibarra Real Nova (headings) + Montserrat (body) + palette ---
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Ibarra+Real+Nova:ital,wght@0,400;0,600;0,700;1,400&family=Montserrat:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root {
    --text: #734443;
    --bg: #fdfbf7;
    --accent: #3d6e70;
}
h1, h2, h3, h4, h5, h6,
[data-testid="stHeading"], [data-testid="stHeading"] *,
[data-testid="stHeading"] h1, [data-testid="stHeading"] h2,
[data-testid="stHeading"] h3 {
    font-family: 'Ibarra Real Nova', Georgia, serif !important;
    color: var(--text) !important;
}
/* body text: Montserrat — target text containers, NOT bare span */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
.stCaption, .stCaption *,
label, .stSelectbox, .stFileUploader,
[data-testid="stFileUploaderDropzone"] * {
    font-family: 'Montserrat', sans-serif !important;
}

.stApp { background-color: var(--bg); }
.stMarkdown, p, label, .stCaption,
span:not([data-testid="stIconMaterial"]) {
    color: var(--accent);
}
a { color: var(--accent) !important; }
.stSelectbox label, .stFileUploader label { color: var(--accent) !important; }
.stButton > button,
[data-testid="stFileUploader"] button {
    background-color: var(--accent) !important;
    color: var(--bg) !important;
    border: none !important;
    white-space: nowrap !important;
}
.stButton > button p,
[data-testid="stFileUploader"] button p,
.stButton > button span:not([data-testid="stIconMaterial"]),
[data-testid="stFileUploader"] button span:not([data-testid="stIconMaterial"]) {
    color: var(--bg) !important;
    font-family: 'Montserrat', sans-serif !important;
}
[data-testid="stIconMaterial"],
button [data-testid="stIconMaterial"],
[data-testid="stFileUploader"] [data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined', 'Material Icons' !important;
    color: var(--bg) !important;
}
</style>
""", unsafe_allow_html=True)


st.title("Color season detector")
st.caption("Upload one or more photos to know your color season!\\\n Requirements: even daylight of flash, not overexposed, no harsh shadows. Better if iris is visible")

# --- model picker (dropdown) ---
available = {name: spec for name, spec in MODEL_FILES.items()
             if (MODEL_DIR / spec[0]).exists()}
if not available:
    st.error("No model files found. Expected one of: "
             + ", ".join(f[0] for f in MODEL_FILES.values()))
    st.stop()

model_name = st.selectbox("Model", list(available.keys()), index=0,
                          help="Pick which trained model to use for classification.")
model_file, feature_kind = available[model_name]
variant = "whitebalanced" if "von Kries" in model_name else "raw"

bundle, model_msg = load_model(model_file)
if bundle is None:
    st.error(model_msg)
    st.stop()
st.caption(f"Loaded: {model_name}")

SEASONS = bundle["SEASONS"]
SUBTYPE_NAMES = bundle["SUBTYPE_NAMES"]
SUB2SEA = bundle["SUB2SEA"]

# Display order for charts (does not affect classification, only drawing).
DISPLAY_ORDER = [
    "spring/bright", "spring/warm", "spring/light",
    "summer/light", "summer/cool", "summer/soft",
    "autumn/soft", "autumn/warm", "autumn/deep",
    "winter/deep", "winter/cool", "winter/bright",
]
SUBTYPE_TO_IDX = {n: i for i, n in enumerate(SUBTYPE_NAMES)}
DISPLAY_IDX = [SUBTYPE_TO_IDX[n] for n in DISPLAY_ORDER if n in SUBTYPE_TO_IDX]

# session state
if "selected_sub" not in st.session_state:
    st.session_state.selected_sub = None
if "drape_color" not in st.session_state:
    st.session_state.drape_color = None

uploaded = st.file_uploader("Face photo (multiple allowed)",
                            type=["png", "jpg", "jpeg", "webp"],
                            accept_multiple_files=True)

if uploaded:
    # Cache the heavy analysis: it depends only on the photos + the model,
    # NOT on which drape color is clicked. A color click triggers a rerun, so
    # without this the whole MediaPipe analysis would run again every click.
    import hashlib
    file_bytes = [uf.getvalue() for uf in uploaded]
    cache_key = hashlib.md5(
        (model_file + "|" + variant + "|"
         + "|".join(hashlib.md5(b).hexdigest() for b in file_bytes)
         ).encode()).hexdigest()

    if st.session_state.get("analysis_key") != cache_key:
        per_photo = []
        with st.spinner(f"Analyzing {len(uploaded)} photo(s)..."):
            for uf, b in zip(uploaded, file_bytes):
                pil = Image.open(io.BytesIO(b))
                img_rgb = np.array(pil.convert("RGB"), np.uint8)
                feat = extract_for_model(img_rgb, feature_kind, variant)
                if feat is None:
                    per_photo.append((uf.name, None, None, pil, False, None))
                    continue
                sea_p, sub_p = classify(bundle, feat)
                # chin is needed for draping; compute it ONCE here (color-independent)
                chin = find_chin(img_rgb)
                per_photo.append((uf.name, sea_p, sub_p, pil, True, chin))
        # store in session so reruns (e.g. color clicks) reuse it
        st.session_state.analysis_key = cache_key
        st.session_state.per_photo = per_photo
    else:
        per_photo = st.session_state.per_photo

    ok = [p for p in per_photo if p[4]]
    failed = [p for p in per_photo if not p[4]]

    if not ok:
        st.warning("No face found in any photo. Try clearer frontal photos.")
        st.stop()

    # aggregate = mean probability over photos with a detected face
    sub_avg = np.mean([p[2] for p in ok], axis=0)

    st.success(f"Averaged over {len(ok)} photo(s)"
               + (f" (skipped {len(failed)} without a face)" if failed else ""))

    # ── uploaded photos carousel: photo + its top-3 subtypes below ──
    st.subheader(f"Uploaded ({len(per_photo)})")

    def _carousel_with_captions(records, target_h=260):
        """Horizontal scroll of cards: image + top-3 subtypes (each on its own
        line with probability) underneath."""
        cards = ""
        for (name, sea_p, sub_p, pil, is_ok, chin) in records:
            im = _to_height(pil.convert("RGB"), target_h)
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            if is_ok:
                t3 = np.argsort(sub_p)[::-1][:3]
                lines = "".join(
                    f"<div style='white-space:nowrap;'>{SUBTYPE_NAMES[i]} "
                    f"({sub_p[i] * 100:.0f}%)</div>" for i in t3)
            else:
                lines = "<div>no face found</div>"
            fname = (f"<div style='font-weight:600;max-width:{int(target_h*0.9)}px;"
                     f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
                     f"margin:0 auto 4px auto;' title='{name}'>{name}</div>")
            cards += (
                f"<div style='flex:0 0 auto;margin-right:14px;text-align:center;"
                f"font-family:Montserrat,sans-serif;color:#3d6e70;font-size:13px;'>"
                f"<img src='data:image/png;base64,{b64}' "
                f"style='height:{target_h}px;border-radius:6px;display:block;"
                f"margin:0 auto 6px auto;'/>{fname}{lines}</div>")
        st.markdown(
            f"<div style='display:flex;overflow-x:auto;padding-bottom:10px;'>"
            f"{cards}</div>", unsafe_allow_html=True)

    _carousel_with_captions(per_photo)

    # top-3 subtypes by probability
    top3 = np.argsort(sub_avg)[::-1][:3]
    st.subheader("Top-3 subtypes")
    for rank, idx in enumerate(top3, 1):
        st.markdown(f"**{rank}. {SUBTYPE_NAMES[idx]}** "
                    f"({sub_avg[idx] * 100:.1f}%)")

    st.divider()

    # subtype chart (clickable to show palette)
    st.subheader("Subtype probability (click a bar!)")
    import plotly.graph_objects as go
    disp_names = [SUBTYPE_NAMES[i] for i in DISPLAY_IDX]
    disp_vals = [sub_avg[i] * 100 for i in DISPLAY_IDX]
    disp_cols = [SEASON_COLORS[SEASONS[SUB2SEA[i]]] for i in DISPLAY_IDX]
    fig = go.Figure(go.Bar(
        x=disp_vals, y=disp_names, orientation="h",
        marker_color=disp_cols, customdata=DISPLAY_IDX,
        text=[f"{v:.1f}%" for v in disp_vals], textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        yaxis=dict(autorange="reversed"), xaxis_title="%",
        height=420, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#734443", family="Ibarra Real Nova, serif"),
    )
    event = st.plotly_chart(fig, use_container_width=True,
                            on_select="rerun", key="subtype_bars")

    # click handling: read trained index from the selected bar's customdata
    top_idx = int(np.argmax(sub_avg))
    clicked = None
    if event and event.get("selection", {}).get("points"):
        pt = event["selection"]["points"][0]
        clicked = int(pt.get("customdata", top_idx))
    sel_idx = clicked if clicked is not None else (
        st.session_state.selected_sub
        if st.session_state.selected_sub is not None else top_idx)
    st.session_state.selected_sub = sel_idx

    # ── palette of the selected subtype (face collage only) ──
    sel_name = SUBTYPE_NAMES[sel_idx]
    st.subheader(f"Palette: {sel_name}  ({sub_avg[sel_idx] * 100:.1f}%)")
    pth = palette_path(sel_name)
    if not pth.exists():
        st.warning(f"Palette not found: {pth}")
    else:
        st.image(crop_faces_collage(str(pth)), use_container_width=True)

        # ── clickable color tiles ──
        st.markdown("<p style='text-align:center;'><b>Click a color to try it "
                    "on with draping:</b></p>", unsafe_allow_html=True)
        colors = extract_palette_colors(str(pth))
        if colors:
            from st_clickable_images import clickable_images

            def _hex_tile_uri(hx, size=48):
                im = Image.new("RGB", (size, size), _hex_to_rgb(hx))
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

            tile_uris = [_hex_tile_uri(hx) for hx in colors][:56]
            clicked_idx = clickable_images(
                tile_uris,
                titles=colors[:56],
                div_style={"display": "grid",
                           "grid-template-columns": "repeat(8, 1fr)",
                           "gap": "6px", "max-width": "480px",
                           "margin": "0 auto", "justify-content": "center"},
                img_style={"width": "100%", "aspect-ratio": "1",
                           "border-radius": "6px", "cursor": "pointer",
                           "border": "1px solid #bbb"},
                key="color_tiles",
            )
            if clicked_idx is not None and clicked_idx > -1:
                st.session_state.drape_color = colors[clicked_idx]

            # ── draping the selected color ──
            drape_hex = st.session_state.drape_color or colors[0]
            st.subheader(f"Draping: {drape_hex}")
            draped_cards = ""
            for (name, sea_p, sub_p, pil, is_ok, chin) in ok:
                dimg = drape_rect(pil, chin, drape_hex)
                im = _to_height(dimg.convert("RGB"), 300)
                buf = io.BytesIO(); im.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                draped_cards += (
                    f"<div style='flex:0 0 auto;margin-right:14px;text-align:center;"
                    f"font-family:Montserrat,sans-serif;color:#3d6e70;font-size:13px;'>"
                    f"<img src='data:image/png;base64,{b64}' "
                    f"style='height:300px;border-radius:6px;display:block;"
                    f"margin:0 auto 4px auto;'/>"
                    f"<div style='max-width:270px;overflow:hidden;"
                    f"text-overflow:ellipsis;white-space:nowrap;margin:0 auto;' "
                    f"title='{name}'>{name}</div></div>")
            st.markdown(
                f"<div style='display:flex;flex-wrap:wrap;justify-content:center;"
                f"padding-bottom:8px;width:100vw;position:relative;"
                f"left:50%;right:50%;margin-left:-50vw;margin-right:-50vw;'>"
                f"{draped_cards}</div>", unsafe_allow_html=True)
else:
    st.info("Upload one or more photos to see the result.")
