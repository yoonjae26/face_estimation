"""Face detection (YuNet, with Haar cascade fallback) and square face cropping."""
import os

import cv2
import numpy as np

from utils import config


class FaceDetector:
    def __init__(self, model_path=config.YUNET_MODEL, score_threshold=0.7):
        self.use_yunet = os.path.exists(model_path) and hasattr(cv2, "FaceDetectorYN")
        if self.use_yunet:
            self.detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score_threshold, 0.3, 5000)
        else:
            print("YuNet model not found, falling back to Haar cascade (less accurate).")
            cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.detector = cv2.CascadeClassifier(cascade)

    def detect(self, image_bgr):
        """Return a list of (x, y, w, h, score) boxes for faces in a BGR image."""
        h, w = image_bgr.shape[:2]
        if self.use_yunet:
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(image_bgr)
            if faces is None:
                return []
            return [(int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[14])) for f in faces]

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [(int(x), int(y), int(fw), int(fh), 1.0) for (x, y, fw, fh) in faces]


def crop_face(image, box, scale=1.0):
    """Crop a square region centred on `box`, enlarged by `scale`.

    Out-of-image areas are filled by edge replication so the face stays centred,
    which matches how the training crops look.
    """
    x, y, w, h = box[:4]
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * scale
    x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x2, y2 = int(round(cx + side / 2)), int(round(cy + side / 2))

    ih, iw = image.shape[:2]
    pad = max(0, -x1, -y1, x2 - iw, y2 - ih)
    if pad:
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad
    return image[y1:y2, x1:x2]


def to_model_input(face_bgr, size, grayscale=False):
    """BGR crop -> float32 RGB array (size, size, 3) in [0, 255], as used in training."""
    face = cv2.resize(face_bgr, (size, size), interpolation=cv2.INTER_AREA)
    if grayscale:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
    else:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    return face.astype(np.float32)
