#!/usr/bin/env python3
"""
check_self_duplicates.py — find duplicate faces WITHIN a single dataset.

Question it answers: does the dataset contain faces cropped from the same frame
more than once? (same face/frame, even if background or crop differs).

Reuses the aligned-face pHash approach from check_duplicate.py, but instead of
comparing NEW against OLD, it compares every face against every other face in
the same set and groups them into clusters of duplicates.

Run:
    pip install mediapipe opencv-python-headless pillow imagehash numpy tqdm
    python check_self_duplicates.py --root /path/to/dataset
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
# CONFIG
# ─────────────────────────────────────────────────────────────────────────
ROOTS = [Path("./dataset/new/")]

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
FACE_THR = 8          # Hamming distance: <=8 — almost certainly the same frame
CROP_SIZE = 160       # size of the aligned face crop
CACHE_FILE = Path("face_phash_cache.pkl")
REPORT_CSV = Path("self_dup_report.csv")
CLUSTERS_PNG = Path("self_dup_clusters.png")

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
# Union-Find for grouping duplicates into clusters
# ─────────────────────────────────────────────────────────────────────────
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# ─────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", nargs="*", default=[str(r) for r in ROOTS])
    ap.add_argument("--thr", type=int, default=FACE_THR,
                    help="Hamming threshold (lower = stricter)")
    ap.add_argument("--no-viz", action="store_true", help="do not draw the collage of clusters")
    args = ap.parse_args()

    files = iter_images([Path(x) for x in args.root])
    print(f"dataset: {len(files)} photos")
    if not files:
        print("Empty dataset — check the paths.")
        return

    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        print(f"cache: {len(cache)} entries")

    faces = hash_all(files, cache, "faces")

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

    print(f"faces encoded: {len(faces)}/{len(files)}")
    if len(faces) < len(files):
        print("  (fewer encoded — MediaPipe found no face; those photos are not compared)")

    # compare every pair, union those within the threshold
    n = len(faces)
    uf = UnionFind(n)
    pair_dist = {}
    for i in tqdm(range(n), desc="matching"):
        pi, hi = faces[i]
        for j in range(i + 1, n):
            pj, hj = faces[j]
            d = hi - hj
            if d <= args.thr:
                uf.union(i, j)
                pair_dist[(i, j)] = d

    # collect clusters of size >= 2
    clusters = {}
    for idx in range(n):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(idx)
    dup_clusters = [c for c in clusters.values() if len(c) >= 2]
    dup_clusters.sort(key=len, reverse=True)

    dup_files = sorted({faces[i][0] for c in dup_clusters for i in c})

    print(f"\n=== RESULT (threshold d<={args.thr}) ===")
    print(f"duplicate clusters: {len(dup_clusters)}")
    print(f"photos involved in duplicates: {len(dup_files)} of {len(files)}")
    for k, c in enumerate(dup_clusters[:20], 1):
        names = ", ".join(faces[i][0].name for i in c)
        print(f"  cluster {k} (size {len(c)}): {names}")
    if len(dup_clusters) > 20:
        print(f"  ... {len(dup_clusters) - 20} more clusters (see {REPORT_CSV})")

    # CSV report: one row per cluster member, tagged with a cluster id
    with open(REPORT_CSV, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["cluster_id", "cluster_size", "path"])
        for k, c in enumerate(dup_clusters, 1):
            for i in c:
                wr.writerow([k, len(c), str(faces[i][0])])
    print(f"\nreport: {REPORT_CSV}")

    # collage: one row per cluster, one column per member
    if dup_clusters and not args.no_viz:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            max_cols = max(len(c) for c in dup_clusters)
            rows = len(dup_clusters)
            fig, axes = plt.subplots(rows, max_cols,
                                     figsize=(2.4 * max_cols, 2.4 * rows),
                                     squeeze=False)
            for r, c in enumerate(dup_clusters):
                for col in range(max_cols):
                    ax = axes[r][col]
                    ax.axis("off")
                    if col < len(c):
                        path = faces[c[col]][0]
                        crop = aligned_face_crop(path)
                        if crop is not None:
                            ax.imshow(crop)
                        ax.set_title(f"c{r + 1}  {path.name}", fontsize=8)
            fig.suptitle("Duplicate clusters (same face / same frame?)")
            plt.tight_layout()
            plt.savefig(CLUSTERS_PNG, dpi=110)
            print(f"collage of clusters: {CLUSTERS_PNG}")
        except Exception as e:
            print("collage skipped:", e)

    if not dup_clusters:
        print("\nNo duplicates found — every face is unique within the dataset.")

    close_mesh()


if __name__ == "__main__":
    main()
