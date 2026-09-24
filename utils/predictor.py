"""End-to-end face analysis: detect faces, then predict age, gender and emotion for each."""
import os

import cv2
import numpy as np

from utils import config
from utils.face_detector import FaceDetector, crop_face, to_model_input


class FaceAnalyzer:
    def __init__(self, age_gender_path=config.AGE_GENDER_MODEL, emotion_path=config.EMOTION_MODEL, tta=True):
        import keras

        import utils.models  # noqa: F401  registers the custom AgeExpectation layer

        for path in (age_gender_path, emotion_path):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model not found: {path}. Train it with the scripts in train/.")
        self.age_gender_model = keras.models.load_model(age_gender_path, compile=False)
        self.emotion_model = keras.models.load_model(emotion_path, compile=False)
        self.detector = FaceDetector()
        self.tta = tta

    def _predict(self, model, batch):
        """Run the model, averaging with the horizontally flipped batch if TTA is on."""
        if not self.tta:
            return model.predict_on_batch(batch)
        out = model.predict_on_batch(np.concatenate([batch, batch[:, :, ::-1]]))
        n = len(batch)
        if isinstance(out, dict):
            return {k: (np.asarray(v[:n]) + np.asarray(v[n:])) / 2 for k, v in out.items()}
        return (np.asarray(out[:n]) + np.asarray(out[n:])) / 2

    def predict_crops(self, face_crops_ag, face_crops_em):
        """Predict from already-cropped BGR faces. Returns a list of result dicts."""
        if not face_crops_ag:
            return []
        ag = np.stack([to_model_input(f, config.AGE_GENDER_IMG_SIZE) for f in face_crops_ag])
        em = np.stack([to_model_input(f, config.EMOTION_IMG_SIZE, grayscale=True) for f in face_crops_em])
        ag_out = self._predict(self.age_gender_model, ag)
        em_out = self._predict(self.emotion_model, em)

        results = []
        for i in range(len(ag)):
            p_female = float(ag_out["gender"][i][0])
            probs = em_out[i]
            k = int(np.argmax(probs))
            results.append({
                "age": float(ag_out["age"][i][0]),
                "gender": config.GENDER_LABELS[int(p_female >= 0.5)],
                "gender_confidence": max(p_female, 1 - p_female),
                "emotion": config.EMOTION_LABELS[k],
                "emotion_confidence": float(probs[k]),
                "emotion_probs": {name: float(p) for name, p in zip(config.EMOTION_LABELS, probs)},
            })
        return results

    def analyze(self, image_bgr, min_face=24):
        """Detect and analyse every face in a BGR image. Each result also has a 'box' (x, y, w, h)."""
        boxes = [b for b in self.detector.detect(image_bgr) if min(b[2], b[3]) >= min_face]
        crops_ag = [crop_face(image_bgr, b, config.AGE_GENDER_CROP_SCALE) for b in boxes]
        crops_em = [crop_face(image_bgr, b, config.EMOTION_CROP_SCALE) for b in boxes]
        results = self.predict_crops(crops_ag, crops_em)
        for r, b in zip(results, boxes):
            r["box"] = tuple(int(v) for v in b[:4])
        return results


EMOTION_COLORS = {  # BGR
    "Angry": (40, 40, 220), "Disgust": (40, 140, 60), "Fear": (160, 60, 160), "Happy": (40, 200, 240),
    "Neutral": (180, 180, 180), "Sad": (200, 120, 40), "Surprise": (30, 150, 255),
}


def draw_results(image_bgr, results):
    out = image_bgr.copy()
    scale = max(0.45, min(out.shape[:2]) / 900)
    thick = max(1, int(round(scale * 2)))
    for r in results:
        x, y, w, h = r["box"]
        color = EMOTION_COLORS.get(r["emotion"], (0, 255, 0))
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thick + 1)
        lines = [f"{r['gender']} {r['gender_confidence']:.0%}, {r['age']:.0f} yrs",
                 f"{r['emotion']} {r['emotion_confidence']:.0%}"]
        ty = y - 6
        for text in reversed(lines):
            (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            ty_top = max(0, ty - th - base)
            cv2.rectangle(out, (x, ty_top), (x + tw + 6, ty_top + th + base + 4), color, cv2.FILLED)
            cv2.putText(out, text, (x + 3, ty_top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick,
                        cv2.LINE_AA)
            ty = ty_top - 2
    return out
