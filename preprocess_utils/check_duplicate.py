#!/usr/bin/env python3
"""
check_face_duplicates.py — local duplicate-face check between two datasets.

Question it answers: does the NEW dataset contain faces cropped from the same
frame that already exists in OLD? (same frame, even if the background/crop differs).

Method:
  1. Face detection (MediaPipe FaceMesh).
  2. Alignment along the eye line (guards against rotations).
  3. Square crop around the face -> perceptual hash (pHash) of the face itself.
  4. Compare NEW against OLD by Hamming distance.

The background (black or not) is left out of frame — we compare the bare face to
the bare face, so it works even when OLD is on a black background and NEW is not.

Run:
    pip install mediapipe opencv-python-headless pillow imagehash numpy tqdm
    python check_face_duplicates.py

Output:
    * prints NEW~OLD pairs with a small distance;
    * saves the report face_dup_report.csv;
    * (optionally) saves a collage of pairs to face_dup_pairs.png.
"""
import argparse
import csv
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import imagehash
import mediapipe as mp
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────
# CONFIG — set your paths here (or pass them via --old / --new)
# ─────────────────────────────────────────────────────────────────────────
OLD_ROOTS = [Path("./dataset/old/")]
NEW_ROOTS = [Path("./dataset/new/")]

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
FACE_THR = 8          # Hamming distance: <=8 — almost certainly the same frame
CROP_SIZE = 160       # size of the aligned face crop
CACHE_FILE = Path("face_phash_cache.pkl")
REPORT_CSV = Path("face_dup_report.csv")
PAIRS_PNG = Path("face_dup_pairs.png")

# eye corners in the MediaPipe FaceMesh topology
_L_EYE_OUT, _L_EYE_IN = 33, 133
_R_EYE_OUT, _R_EYE_IN = 263, 362


# ─────────────────────────────────────────────────────────────────────────
# MediaPipe mesh (lazy initialization)
# ─────────────────────────────────────────────────────────────────────────
_mesh = None


def get_mesh():
    global _mesh
    if _mesh is None:
        base = mp.tasks.BaseOptions
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base(model_asset_path=_ensure_model()),
            num_faces=1,
        )
        _mesh = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
    return _mesh


def _ensure_model():
    """Download face_landmarker.task next to the script if it isn't there yet."""
    model_path = Path(__file__).with_name("face_landmarker.task")
    if not model_path.exists():
        import urllib.request
        url = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
               "face_landmarker/float16/1/face_landmarker.task")
        print("downloading model face_landmarker.task ...")
        urllib.request.urlretrieve(url, model_path)
    return str(model_path)


# ─────────────────────────────────────────────────────────────────────────
# File collection
# ─────────────────────────────────────────────────────────────────────────
def iter_images(roots):
    out = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            print(f"[warn] no such directory: {root}")
            continue
        for p in root.rglob("*"):
            if p.suffix.lower() in IMG_EXT:
                out.append(p)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Aligned face crop + pHash
# ─────────────────────────────────────────────────────────────────────────
def aligned_face_crop(path, out_size=CROP_SIZE, margin=0.6):
    try:
        img = np.array(Image.open(path).convert("RGB"), np.uint8)
    except Exception:
        return None
    h, w = img.shape[:2]
    res = get_mesh().detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=np.ascontiguousarray(img)))
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]

    def px(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    l_eye = (px(_L_EYE_OUT) + px(_L_EYE_IN)) / 2
    r_eye = (px(_R_EYE_OUT) + px(_R_EYE_IN)) / 2
    eyes_center = (l_eye + r_eye) / 2
    dx, dy = (r_eye - l_eye)
    angle = np.degrees(np.arctan2(dy, dx))

    xs = np.array([lm[i].x * w for i in range(len(lm))])
    ys = np.array([lm[i].y * h for i in range(len(lm))])
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * (1 + margin)

    M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, 1.0)
    rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))

    cx, cy = eyes_center
    left, top = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    right, bottom = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    crop = rot[top:bottom, left:right]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)


def face_phash(path):
    crop = aligned_face_crop(path)
    if crop is None:
        return None
    return imagehash.phash(Image.fromarray(crop))


# ─────────────────────────────────────────────────────────────────────────
# Hashing with an on-disk cache
# ─────────────────────────────────────────────────────────────────────────
def hash_all(files, cache, desc):
    result = []
    for p in tqdm(files, desc=desc):
        key = str(p)
        if key in cache:
            h = cache[key]
        else:
            h = face_phash(p)
            cache[key] = h            # cache even None (no face found)
        if h is not None:
            result.append((p, h))
    return result


def close_mesh():
    global _mesh
    if _mesh is not None:
        try:
            _mesh.close()
        except Exception:
            pass
        _mesh = None

# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", nargs="*", default=[str(r) for r in OLD_ROOTS])
    ap.add_argument("--new", nargs="*", default=[str(r) for r in NEW_ROOTS])
    ap.add_argument("--thr", type=int, default=FACE_THR,
                    help="Hamming threshold (lower = stricter)")
    ap.add_argument("--no-viz", action="store_true", help="do not draw the collage of pairs")
    args = ap.parse_args()

    old_files = iter_images([Path(x) for x in args.old])
    new_files = iter_images([Path(x) for x in args.new])
    print(f"OLD: {len(old_files)} photos | NEW: {len(new_files)} photos")
    if not old_files or not new_files:
        print("Empty dataset — check the paths.")
        return

    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        print(f"cache: {len(cache)} entries")

    old_h = hash_all(old_files, cache, "old faces")
    new_h = hash_all(new_files, cache, "new faces")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

    print(f"faces encoded: OLD {len(old_h)}/{len(old_files)}, "
          f"NEW {len(new_h)}/{len(new_files)}")
    if len(old_h) < len(old_files) or len(new_h) < len(new_files):
        print("  (where fewer were encoded — MediaPipe found no face; those photos are not compared)")

    # for each NEW, find the closest OLD
    matches = []
    for np_, hn in tqdm(new_h, desc="matching"):
        best_op, best_d = None, None
        for op, ho in old_h:
            d = hn - ho
            if best_d is None or d < best_d:
                best_op, best_d = op, d
        if best_d is not None and best_d <= args.thr:
            matches.append((np_, best_op, best_d))

    matches.sort(key=lambda x: x[2])
    dup_new = sorted({m[0] for m in matches})

    print(f"\n=== RESULT (threshold d<={args.thr}) ===")
    print(f"NEW photos with a face matching OLD: {len(dup_new)} of {len(new_files)}")
    for np_, op, d in matches[:30]:
        print(f"  d={d:2d}  NEW {np_.name}  ~  OLD {op.name}")
    if len(matches) > 30:
        print(f"  ... {len(matches) - 30} more pairs (see {REPORT_CSV})")

    # CSV report
    with open(REPORT_CSV, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["hamming", "new_path", "old_path"])
        for np_, op, d in matches:
            wr.writerow([d, str(np_), str(op)])
    print(f"\nreport: {REPORT_CSV}")

    # collage of pairs
    if matches and not args.no_viz:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            pairs = matches
            fig, axes = plt.subplots(len(pairs), 2, figsize=(5, 2.4 * len(pairs)))
            if len(pairs) == 1:
                axes = axes.reshape(1, 2)
            for row, (np_, op, d) in enumerate(pairs):
                for col, (path, tag) in enumerate([(np_, "NEW"), (op, "OLD")]):
                    crop = aligned_face_crop(path)
                    if crop is not None:
                        axes[row, col].imshow(crop)
                    axes[row, col].set_title(f"{tag}  d={d}", fontsize=10)
                    axes[row, col].axis("off")
            fig.suptitle("Candidates: the same face from the same frame?")
            plt.tight_layout()
            plt.savefig(PAIRS_PNG, dpi=110)
            print(f"collage of pairs: {PAIRS_PNG}")
        except Exception as e:
            print("collage skipped:", e)

    if not matches:
        print("\nNo matches found — no NEW face is cropped from an OLD frame.")

    close_mesh()


if __name__ == "__main__":
    main()
