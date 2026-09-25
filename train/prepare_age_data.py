"""Build one clean, consistently-cropped age/gender dataset from several sources.

Every image is re-cropped with the *same* detector + crop function used at inference
(utils.face_detector), so training crops look exactly like webcam/photo crops.

Sources (put them under data/, see README):
  UTKFace        data/utkface/UTKFace/<age>_<gender>_<race>_<date>.jpg      (all ages, mixed ethnicity)
  AFAD-Full      data/afad/AFAD-Full/<age>/<111 male|112 female>/*.jpg     (Asian, 15-72)
  MegaAge-Asian  data/megaage/megaage_asian/megaage_asian/{train,test}/     (Asian, 0-70, no gender)
  APPA-REAL      data/appa/final_files/final_files/ + data/appa/labels.csv  (in-the-wild, no gender)

Output: data/age_crops/<source>/<n>.jpg and data/age_crops/index.csv
(columns: path, age, gender (-1 = unknown), source, split).

    python train/prepare_age_data.py [--afad-max 60000] [--workers 16]
"""
import argparse
import csv
import glob
import os
import random
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config  # noqa: E402
from utils.data import parse_utkface_filename, utkface_splits  # noqa: E402
from utils.face_detector import FaceDetector, crop_face  # noqa: E402

OUT_DIR = os.path.join(config.DATA_DIR, "age_crops")
SAVE_SIZE = 192  # a bit above the model input so augmentation can zoom without upsampling

_detector = None


def _crop_one(job):
    """Detect the main face and save the standard crop. Returns the saved path or None."""
    global _detector
    if _detector is None:
        cv2.setNumThreads(1)
        _detector = FaceDetector()
    src, dst, tight = job
    img = cv2.imread(src)
    if img is None:
        return None
    # Detection needs some context and resolution: pad tight crops and upscale small images.
    pad = int(0.25 * max(img.shape[:2])) if tight else 0
    work = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else img
    up = max(1.0, 256.0 / min(work.shape[:2]))
    if up > 1:
        work = cv2.resize(work, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    faces = _detector.detect(work)
    if not faces:
        if not tight:
            return None  # loose image with no detectable face: skip it
        face = img  # tight crop: already a face
    else:
        h, w = work.shape[:2]

        # main face = largest, penalised by distance from the image centre
        def score(f):
            cx, cy = f[0] + f[2] / 2, f[1] + f[3] / 2
            return f[2] * f[3] * (1 - 0.5 * np.hypot((cx - w / 2) / w, (cy - h / 2) / h))
        face = crop_face(work, max(faces, key=score), config.AGE_GENDER_CROP_SCALE)
    face = cv2.resize(face, (SAVE_SIZE, SAVE_SIZE), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, face, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return dst


def list_sources(afad_max, seed):
    """Yield (src_path, age, gender, source, split_or_None, tight)."""
    items = []
    utk_split = {p: name for name, (paths, _, _) in utkface_splits().items() for p in paths}
    for p in sorted(glob.glob(os.path.join(config.UTKFACE_DIR, "*.jpg"))):
        parsed = parse_utkface_filename(p)
        if parsed:
            items.append((p, parsed[0], parsed[1], "utkface", utk_split[p], True))

    afad = []
    for p in sorted(glob.glob(os.path.join(config.DATA_DIR, "afad", "AFAD-Full", "*", "*", "*.jpg"))):
        age_dir, g_dir = p.split(os.sep)[-3:-1]
        afad.append((p, int(age_dir), 0 if g_dir == "111" else 1, "afad", None, True))
    random.Random(seed).shuffle(afad)
    items += afad[:afad_max]

    mega = os.path.join(config.DATA_DIR, "megaage", "megaage_asian", "megaage_asian")
    for split in ("train", "test"):
        names = open(os.path.join(mega, "list", f"{split}_name.txt")).read().split()
        ages = open(os.path.join(mega, "list", f"{split}_age.txt")).read().split()
        for n, a in zip(names, ages):
            items.append((os.path.join(mega, split, n), int(a), -1, "megaage", "test" if split == "test" else None,
                          False))

    appa = os.path.join(config.DATA_DIR, "appa")
    with open(os.path.join(appa, "labels.csv")) as f:
        for row in csv.DictReader(f):
            items.append((os.path.join(appa, "final_files", "final_files", row["file_name"]),
                          int(float(row["real_age"])), -1, "appa", None, False))
    return items


def assign_splits(rows, seed):
    """80/10/10 per source, stratified by age decade. MegaAge keeps its official test split and
    UTKFace keeps the split from utils.data.utkface_splits (so old and new models compare fairly)."""
    rng = random.Random(seed)
    by_key = {}
    for r in rows:
        if r["split"] is None:
            by_key.setdefault((r["source"], min(r["age"] // 10, 8)), []).append(r)
    for (source, _), group in by_key.items():
        rng.shuffle(group)
        n = len(group)
        n_test = 0 if source == "megaage" else round(0.1 * n)
        for i, r in enumerate(group):
            r["split"] = "test" if i < n_test else "val" if i < n_test + round(0.1 * n) else "train"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--afad-max", type=int, default=60000, help="cap AFAD so it doesn't dominate")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    items = list_sources(args.afad_max, config.SEED)
    jobs, rows = [], []
    for i, (src, age, gender, source, split, tight) in enumerate(items):
        os.makedirs(os.path.join(OUT_DIR, source), exist_ok=True)
        dst = os.path.join(OUT_DIR, source, f"{i:06d}.jpg")
        jobs.append((src, dst, tight))
        rows.append({"path": os.path.relpath(dst, config.DATA_DIR), "age": min(max(age, 0), config.MAX_AGE),
                     "gender": gender, "source": source, "split": split})

    print(f"Cropping {len(jobs)} images with {args.workers} workers...")
    with Pool(args.workers) as pool:
        saved = pool.map(_crop_one, jobs, chunksize=256)
    rows = [r for r, s in zip(rows, saved) if s]
    assign_splits(rows, config.SEED)

    with open(os.path.join(OUT_DIR, "index.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "age", "gender", "source", "split"])
        w.writeheader()
        w.writerows(rows)
    for source in ("utkface", "afad", "megaage", "appa"):
        n = {s: sum(r["source"] == source and r["split"] == s for r in rows) for s in ("train", "val", "test")}
        total = sum(src == source for _, _, _, src, _, _ in items)
        print(f"{source:8s} kept {sum(n.values())}/{total}  {n}")


if __name__ == "__main__":
    main()
