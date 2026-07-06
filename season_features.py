"""
Shared face-zone color feature extractor.

Single source of truth for:
  * the training notebook (season_classification.ipynb)
  * the Streamlit app (app_draping.py)

Keeping the extractor here guarantees the features used in training and in
production are IDENTICAL (otherwise: a silent feature mismatch and a quality
drop).

Self-contained: the MediaPipe mesh, sclera masking, and von Kries white balance
used to live in season_dataset.py; the parts needed for feature extraction are
inlined below, so this module depends only on numpy / opencv / mediapipe
(no tensorflow, no season_dataset).

Feature vector per photo:
  len(ALL_ZONES) * _ZONE_F                         — direct zone features
  + len(ALL_ZONES)*(len(ALL_ZONES)-1)//2 * 3       — contrasts (Lab median diffs)
"""
import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ═════════════════════════════════════════════════════════════════════════
# MediaPipe Face Mesh (inlined from season_dataset.py)
# ═════════════════════════════════════════════════════════════════════════
_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
              "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "face_landmarker.task")
_landmarker = None


def _ensure_model():
    """Download the FaceLandmarker model once, cache it next to this file."""
    if not os.path.exists(_MODEL_PATH):
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


def _get_mesh():
    """Lazily create and reuse a single FaceLandmarker (one per process)."""
    global _landmarker
    if _landmarker is None:
        opts = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=_ensure_model()),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
        )
        _landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
    return _landmarker


# ═════════════════════════════════════════════════════════════════════════
# Sclera-based von Kries white balance (inlined from season_dataset.py)
# ═════════════════════════════════════════════════════════════════════════
# Landmarks bounding the visible sclera of each eye (MediaPipe 468 topology).
_LEFT_EYE = [33, 133, 159, 145, 153, 154, 155, 144, 163, 7]
_RIGHT_EYE = [362, 263, 386, 374, 380, 381, 382, 373, 390, 249]

_WARM_BACK = 0.6              # 0 = full correction, 1 = none (under-correct)
_GAIN_LO, _GAIN_HI = 0.5, 2.0  # trusted per-channel gain range


class WBConfig:
    """Configuration for sclera-based von Kries white balance.

    enabled:      master switch; if False, WB is skipped entirely.
    warm_back:    under-correction strength, 0=full .. 1=none.
    gain_lo/hi:   trusted per-channel gain range; outside it WB is skipped.
    min_sclera_v: brightest eye pixel must reach this V (0-255) or WB is skipped.
    bright_frac:  keep sclera pixels within this fraction of local max brightness.
    sat_max:      max HSV saturation for a pixel to count as neutral white.
    min_pixels:   minimum sclera pixels required to estimate the illuminant.
    anti_green:   skip when both R and B gains fall below anti_green_thresh.
    """

    def __init__(self, enabled=True, warm_back=_WARM_BACK,
                 gain_lo=_GAIN_LO, gain_hi=_GAIN_HI,
                 min_sclera_v=110, bright_frac=0.78, sat_max=70,
                 min_pixels=30, anti_green=True, anti_green_thresh=0.93):
        self.enabled = enabled
        self.warm_back = warm_back
        self.gain_lo = gain_lo
        self.gain_hi = gain_hi
        self.min_sclera_v = min_sclera_v
        self.bright_frac = bright_frac
        self.sat_max = sat_max
        self.min_pixels = min_pixels
        self.anti_green = anti_green
        self.anti_green_thresh = anti_green_thresh


DEFAULT_WB = WBConfig()


def _srgb_to_linear(x):
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)


def _linear_to_srgb(x):
    a = 0.055
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * (x ** (1 / 2.4)) - a)


def _sclera_mask(img_rgb, eye_polys, h, w, cfg=DEFAULT_WB):
    """Boolean mask of sclera (eye-white) pixels inside the given eye polygons.

    The sclera is the brightest, near-neutral part of the eye region. We key off
    lightness: keep only pixels close to the local max brightness AND low
    saturation, which excludes the lid, lash line, eye-corner skin and the iris.
    If even the brightest eye pixels are dark, return an empty mask so the caller
    SKIPS white balance rather than estimating it from skin-toned pixels.
    """
    region = np.zeros((h, w), dtype=np.uint8)
    for poly in eye_polys:
        cv2.fillPoly(region, [poly], 1)
    region = region.astype(bool)
    if not region.any():
        return region

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    v_max = int(v[region].max())
    if v_max < cfg.min_sclera_v:
        return np.zeros_like(region)

    v_lo = max(cfg.min_sclera_v, int(cfg.bright_frac * v_max))
    return region & (v >= v_lo) & (v < 250) & (s < cfg.sat_max)


def _von_kries_correct(img_rgb, lm, h, w, cfg=DEFAULT_WB):
    """White-balance an RGB uint8 image using the sclera as a grey reference.

    Returns (corrected_img, applied). `applied` is True only if a trusted sclera
    estimate was found and correction was performed; otherwise the original image
    is returned with applied=False.
    """
    if not cfg.enabled:
        return img_rgb, False

    eye_polys = [np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in idxs],
                          dtype=np.int32)
                 for idxs in (_LEFT_EYE, _RIGHT_EYE)]

    sclera = _sclera_mask(img_rgb, eye_polys, h, w, cfg)
    if int(sclera.sum()) < cfg.min_pixels:
        return img_rgb, False

    # correct in LINEAR light (von Kries is only valid there)
    lin = _srgb_to_linear(img_rgb.astype(np.float32) / 255.0)
    r_w, g_w, b_w = np.median(lin[sclera], axis=0)
    if min(r_w, g_w, b_w) < 1e-5:
        return img_rgb, False

    k_r_full = g_w / r_w
    k_b_full = g_w / b_w
    if not (cfg.gain_lo < k_r_full < cfg.gain_hi
            and cfg.gain_lo < k_b_full < cfg.gain_hi):
        return img_rgb, False  # extreme cast -> don't trust it

    # anti-green safeguard: a real cast is mostly along ONE axis; pulling both R
    # and B down usually means the "white" was skin-toned, not sclera -> skip.
    if cfg.anti_green and (k_r_full < cfg.anti_green_thresh
                           and k_b_full < cfg.anti_green_thresh):
        return img_rgb, False

    wb = cfg.warm_back
    k_r = 1.0 * wb + k_r_full * (1 - wb)
    k_b = 1.0 * wb + k_b_full * (1 - wb)
    lin[..., 0] *= k_r
    lin[..., 2] *= k_b
    out = _linear_to_srgb(lin)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8), True

# ─────────────────────────────────────────────────────────────────────────
# Zone geometry (MediaPipe 478-landmark topology)
# ─────────────────────────────────────────────────────────────────────────
ZONE_LM = {
    "cheek_l":   [50, 101, 118, 117, 123, 147],
    "cheek_r":   [280, 330, 347, 346, 352, 376],
    "forehead":  [10, 67, 69, 109, 151, 338, 297, 299, 108, 337],
    "lips":      [13, 14, 17, 0, 267, 37, 39, 40, 185, 61, 291],
    "nose":      [1, 2, 98, 327, 168],
    "eyebrow_l": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "eyebrow_r": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    "chin":      [152, 148, 176, 149, 150, 136, 172, 377, 400, 378, 379, 365],
}
_LEFT_IRIS = [468, 469, 470, 471, 472]
_RIGHT_IRIS = [473, 474, 475, 476, 477]

ALL_ZONES = ["cheek_l", "cheek_r", "forehead", "lips", "nose", "iris", "sclera",
             "hair", "eyebrow_l", "eyebrow_r", "chin"]
_POLY_ZONES = ("cheek_l", "cheek_r", "forehead", "lips", "nose",
               "eyebrow_l", "eyebrow_r", "chin")

# 3 (med Lab) + 6 (Lab p25/p75) + 3 (med HSV) + 3 (hue_angle, chroma, hue_valid)
_ZONE_F = 15

# Total vector length — handy for asserts in the notebook / app
FEATURE_DIM = len(ALL_ZONES) * _ZONE_F + len(ALL_ZONES) * (len(ALL_ZONES) - 1) // 2 * 3

# Lazy shared-mesh handle (kept for backward compatibility; wraps _get_mesh)
_mesh = None


def _get_shared_mesh():
    global _mesh
    if _mesh is None:
        _mesh = _get_mesh()
    return _mesh


# ─────────────────────────────────────────────────────────────────────────
# Per-zone pixel masks
# ─────────────────────────────────────────────────────────────────────────
def _poly_px(img, lm, idxs, h, w):
    pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in idxs], np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(pts), 1)
    return img[m.astype(bool)]


def _iris_px(img, lm, iris, h, w):
    """Iris pixels WITHOUT the pupil (dark) or highlight (bright).
    Grey irises are kept as-is — that is a real color."""
    cx, cy = lm[iris[0]].x * w, lm[iris[0]].y * h
    rad = np.mean([np.hypot(lm[i].x * w - cx, lm[i].y * h - cy) for i in iris[1:]])
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (int(cx), int(cy)), max(1, int(rad * 0.7)), 1, -1)
    px = img[m.astype(bool)]
    if len(px) < 5:
        return px
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8),
                       cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    L = lab[:, 0]
    keep = (L > 30) & (L < 220)  # no chroma filter -> grey irises survive
    return px[keep] if keep.sum() >= 5 else px


def _hair_px(img, lm, h, w, skin_lab):
    """Hair = pixels above the forehead that differ from skin tone (skin_lab).
    Catches both dark and light/red hair."""
    ys = [int(lm[i].y * h) for i in ZONE_LM["forehead"]]
    xs = [int(lm[i].x * w) for i in ZONE_LM["forehead"]]
    top = max(0, min(ys) - int(0.18 * h))
    bot = min(ys)
    l = max(0, min(xs))
    r = min(w, max(xs))
    if bot <= top or r <= l or skin_lab is None:
        return np.zeros((0, 3), np.uint8)
    patch = img[top:bot, l:r].reshape(-1, 3)
    if len(patch) < 5:
        return np.zeros((0, 3), np.uint8)
    patch_lab = cv2.cvtColor(
        np.clip(patch, 0, 255).reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_RGB2LAB,
    ).reshape(-1, 3).astype(np.float32)
    dist = np.linalg.norm(patch_lab - skin_lab, axis=1)
    mask = dist > 40.0                       # clearly different from skin
    if mask.sum() < 5:                       # fallback: top-40% by distance
        mask = dist >= np.percentile(dist, 60)
    return patch[mask]


def _zone_stats(px):
    """px (N,3) RGB uint8 -> (list of _ZONE_F features, Lab median).
    Empty/too-small zone -> zeros."""
    if len(px) >= 5:
        p = px.reshape(-1, 1, 3).astype(np.uint8)
        lab = cv2.cvtColor(p, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        hsv = cv2.cvtColor(p, cv2.COLOR_RGB2HSV).reshape(-1, 3).astype(np.float32)
        med_lab = np.median(lab, 0)

        # warm/cool via hue angle + chroma (a,b centered ~128)
        a_c, b_c = med_lab[1] - 128, med_lab[2] - 128
        chroma = float(np.hypot(a_c, b_c))            # color purity
        if chroma > 6:                                # vivid color -> angle reliable
            hue_angle = float(np.degrees(np.arctan2(b_c, a_c)))
            hue_valid = 1.0
        else:                                         # grey/neutral -> angle meaningless
            hue_angle = 0.0
            hue_valid = 0.0

        feat = (list(med_lab)
                + list(np.percentile(lab, [25, 75], axis=0).ravel())
                + list(np.median(hsv, 0))
                + [hue_angle, chroma, hue_valid])
        return feat, med_lab
    return [0.0] * _ZONE_F, np.zeros(3, np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────
def extract_features(img_rgb, use_von_kries=False):
    """RGB uint8 -> color feature vector (float32), or None if no face is found.

    use_von_kries=False (default) — the 'raw' variant used by the app.
    use_von_kries=True — apply sclera von Kries white balance (the 'vk' variant).
    """
    mesh = _get_shared_mesh()
    h, w = img_rgb.shape[:2]
    res = mesh.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                               data=np.ascontiguousarray(img_rgb)))
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]

    img = img_rgb
    if use_von_kries:
        try:
            img, _ = _von_kries_correct(img_rgb, lm, h, w)
        except Exception:
            img = img_rgb

    # skin_lab from the FOREHEAD so the forehead is not counted as "hair"
    fore_px = _poly_px(img, lm, ZONE_LM["forehead"], h, w)
    if len(fore_px) >= 5:
        fl = cv2.cvtColor(fore_px.reshape(-1, 1, 3).astype(np.uint8),
                          cv2.COLOR_RGB2LAB).reshape(-1, 3)
        skin_lab = np.median(fl.astype(np.float32), 0)
    else:
        skin_lab = None

    feats = []
    meds = {}
    for name in ALL_ZONES:
        if name in _POLY_ZONES:
            px = _poly_px(img, lm, ZONE_LM[name], h, w)
        elif name == "iris":
            px = np.vstack([_iris_px(img, lm, _LEFT_IRIS, h, w),
                            _iris_px(img, lm, _RIGHT_IRIS, h, w)])
        elif name == "sclera":
            eye_polys = [np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in idxs], np.int32)
                         for idxs in (_LEFT_EYE, _RIGHT_EYE)]
            sc = _sclera_mask(img, eye_polys, h, w)
            px = img[sc] if sc.sum() > 0 else np.zeros((0, 3), np.uint8)
        elif name == "hair":
            px = _hair_px(img, lm, h, w, skin_lab)
        f, med = _zone_stats(px)
        feats += f
        meds[name] = med

    # contrasts in the FIXED ALL_ZONES order
    for i in range(len(ALL_ZONES)):
        for j in range(i + 1, len(ALL_ZONES)):
            feats += list(meds[ALL_ZONES[i]] - meds[ALL_ZONES[j]])

    return np.array(feats, np.float32)


# Feature names — useful for feature importance / EDA
def feature_names():
    stat_names = ["L_med", "a_med", "b_med", "L_p25", "a_p25", "b_p25",
                  "L_p75", "a_p75", "b_p75", "H_med", "S_med", "V_med",
                  "hue_angle", "chroma", "hue_valid"]
    names = [f"{z}__{s}" for z in ALL_ZONES for s in stat_names]
    for i in range(len(ALL_ZONES)):
        for j in range(i + 1, len(ALL_ZONES)):
            for ch in ("L", "a", "b"):
                names.append(f"contrast__{ALL_ZONES[i]}-{ALL_ZONES[j]}__{ch}")
    return names


if __name__ == "__main__":
    print("season_features ready")
    print("zones:", len(ALL_ZONES), "| vector length:", FEATURE_DIM)
    assert len(feature_names()) == FEATURE_DIM, "feature_names vs FEATURE_DIM mismatch"
    print("feature_names ok:", len(feature_names()))
