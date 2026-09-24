"""Measure how large the detected face is relative to the training crops.

Inference crops a detected box enlarged by `scale`; to look like the training data
that scale should be image_side / detected_box_side. Prints the median per dataset,
which is what config.AGE_GENDER_CROP_SCALE / EMOTION_CROP_SCALE are set from.
"""
import glob
import os
import random
import sys

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import config  # noqa: E402
from utils.face_detector import FaceDetector  # noqa: E402


def measure(paths, detector, upscale=1):
    ratios = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        if upscale != 1:
            img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        faces = detector.detect(img)
        if not faces:
            continue
        x, y, w, h, _ = max(faces, key=lambda f: f[2] * f[3])
        ratios.append(img.shape[1] / max(w, h))
    return np.array(ratios)


def main():
    random.seed(config.SEED)
    detector = FaceDetector()
    utk = random.sample(glob.glob(os.path.join(config.UTKFACE_DIR, "*.jpg")), 1000)
    fer = random.sample(glob.glob(os.path.join(config.FER_DIR, "train", "*", "*.jpg")), 1000)
    for name, paths, up in [("UTKFace", utk, 1), ("FER2013", fer, 4)]:
        r = measure(paths, detector, up)
        print(f"{name}: detected {len(r)}/{len(paths)}  scale median={np.median(r):.3f} "
              f"p25={np.percentile(r, 25):.3f} p75={np.percentile(r, 75):.3f}")


if __name__ == "__main__":
    main()
