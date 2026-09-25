"""Build docs/demo.gif from random held-out test images (no webcam needed).

Each face goes through the real pipeline (detect -> crop -> predict), and the frame
shows the prediction next to the ground truth. Faces are sampled at random from the
test splits (fixed seed), so the GIF is not cherry-picked. UTKFace is shown from its
original files; AFAD is left out because its 89 px images look blurry at GIF size.

    python docs/make_demo_gif.py [--n 8] [--seed 3]
"""
import argparse
import os
import random
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import matplotlib  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from utils import config  # noqa: E402
from utils.data import age_crops_splits, utkface_splits  # noqa: E402
from utils.predictor import EMOTION_COLORS, FaceAnalyzer  # noqa: E402

W, H, IMG = 760, 380, 340
BG, PANEL, TEXT, MUTED, TRACK = (15, 17, 21), (24, 27, 34), (232, 234, 240), (138, 144, 160), (38, 42, 51)
GOOD, BAD = (46, 204, 113), (255, 107, 107)
SOURCE_NAMES = {"utkface": "UTKFace", "megaage": "MegaAge-Asian", "appa": "APPA-REAL"}
FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")


def font(size, bold=False):
    return ImageFont.truetype(os.path.join(FONT_DIR, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), size)


def rgb(bgr):
    return tuple(int(c) for c in bgr[::-1])


def pick_faces(analyzer, n, seed):
    """Random test faces (balanced over sources) that the detector finds, with their results."""
    paths, ages, genders, sources = age_crops_splits()["test"]
    candidates = {s: [(p, a, g) for p, a, g, src in zip(paths, ages, genders, sources) if src == s]
                  for s in ("megaage", "appa")}
    candidates["utkface"] = list(zip(*utkface_splits()["test"]))  # original files, not re-cropped
    rng = random.Random(seed)
    order = {s: rng.sample(candidates[s], 200) for s in SOURCE_NAMES}
    picked, k = [], 0
    while len(picked) < n:
        src = list(SOURCE_NAMES)[k % len(SOURCE_NAMES)]
        path, age, gender = order[src].pop()
        k += 1
        img = cv2.resize(cv2.imread(path), (IMG - 80, IMG - 80), interpolation=cv2.INTER_AREA)
        img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=BG[::-1])
        results = analyzer.analyze(img)
        if results:
            picked.append((img, max(results, key=lambda r: r["box"][2] * r["box"][3]), age, gender, src))
    return picked


def render(img, r, true_age, true_gender, src, show_result):
    canvas = Image.new("RGB", (W, H), BG)
    canvas.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (20, 20))
    d = ImageDraw.Draw(canvas)
    px = IMG + 44
    d.rounded_rectangle((px - 12, 20, W - 20, H - 20), radius=12, fill=PANEL)
    d.text((px + 4, H - 44), f"Held-out test image · {SOURCE_NAMES[src]}", font=font(12), fill=MUTED)

    x, y, w, h = r["box"]
    color = rgb(EMOTION_COLORS[r["emotion"]])
    if not show_result:
        d.rectangle((x + 20, y + 20, x + w + 20, y + h + 20), outline=(79, 140, 255), width=3)
        d.text((px + 4, 40), "Detecting face...", font=font(22, True), fill=TEXT)
        return canvas
    d.rectangle((x + 20, y + 20, x + w + 20, y + h + 20), outline=color, width=3)

    d.text((px + 4, 34), "AGE", font=font(12), fill=MUTED)
    d.text((px + 4, 50), f"{r['age']:.0f} yrs", font=font(30, True), fill=TEXT)
    err = abs(r["age"] - true_age)
    d.text((px + 150, 64), f"true {true_age:.0f}  (±{err:.0f})", font=font(14), fill=GOOD if err <= 5 else BAD)

    d.text((px + 4, 98), "GENDER", font=font(12), fill=MUTED)
    d.text((px + 4, 114), f"{r['gender']} {r['gender_confidence']:.0%}", font=font(24, True), fill=TEXT)
    if true_gender >= 0:
        truth = config.GENDER_LABELS[int(true_gender)]
        d.text((px + 200, 122), f"true {truth}", font=font(14), fill=GOOD if truth == r["gender"] else BAD)

    d.text((px + 4, 154), "EMOTION", font=font(12), fill=MUTED)
    d.text((px + 4, 170), r["emotion"], font=font(24, True), fill=TEXT)
    for j, e in enumerate(config.EMOTION_LABELS):
        yy, p = 208 + j * 18, r["emotion_probs"][e]
        d.text((px + 4, yy - 3), e, font=font(12), fill=TEXT)
        d.rounded_rectangle((px + 72, yy, px + 262, yy + 9), radius=4, fill=TRACK)
        if p > 0.01:
            d.rounded_rectangle((px + 72, yy, px + 72 + max(9, int(190 * p)), yy + 9), radius=4,
                                fill=rgb(EMOTION_COLORS[e]))
        d.text((px + 270, yy - 3), f"{p:.0%}", font=font(12), fill=MUTED)
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="number of faces")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--output", default=os.path.join(config.ROOT_DIR, "docs", "demo.gif"))
    args = parser.parse_args()

    analyzer = FaceAnalyzer()
    frames, durations = [], []
    for img, r, age, gender, src in pick_faces(analyzer, args.n, args.seed):
        print(f"{SOURCE_NAMES[src]:14s} true {age:3.0f}  pred {r['age']:5.1f}  {r['gender']:6s}  {r['emotion']}")
        frames += [render(img, r, age, gender, src, False), render(img, r, age, gender, src, True)]
        durations += [350, 1800]

    # Each face gets its own palette (a shared one leaves speckles on the other photos).
    frames = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
    print(f"Saved {args.output} ({os.path.getsize(args.output) / 1e6:.2f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
