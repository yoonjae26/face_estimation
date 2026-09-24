"""Try the fine-tuned models directly and save visual results to demo_results/.

    # Random held-out test images, predictions vs. ground truth
    python demo.py                          # both datasets, 12 images each
    python demo.py --dataset utkface -n 24  # age + gender only
    python demo.py --dataset fer --seed 7   # emotion only, different sample

    # Your own photos (full pipeline: detect faces -> predict)
    python demo.py --images photo.jpg
    python demo.py --images my_photos/
"""
import argparse
import glob
import os
import random

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils import config  # noqa: E402
from utils.data import fer_splits, utkface_splits  # noqa: E402
from utils.predictor import FaceAnalyzer, draw_results  # noqa: E402

OUT_DIR = os.path.join(config.ROOT_DIR, "demo_results")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def save_grid(images, titles, colors, path, cols=6):
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 3.1))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, img, title, color in zip(np.ravel(axes), images, titles, colors):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=8.5, color=color)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"-> saved {path}\n")


def demo_utkface(analyzer, n, seed):
    paths, ages, genders = utkface_splits()["test"]
    idx = random.Random(seed).sample(range(len(paths)), n)
    crops = [cv2.imread(paths[i]) for i in idx]
    results = analyzer.predict_crops(crops, crops)  # test images are already face crops

    print(f"UTKFace test set: {n} random images")
    print(f"{'#':>3}  {'true age':>8}  {'pred age':>8}  {'error':>5}  {'true gender':<11}  {'pred gender':<16}")
    titles, colors, errs, correct = [], [], [], []
    for k, (i, r) in enumerate(zip(idx, results), 1):
        true_g = config.GENDER_LABELS[int(genders[i])]
        err = abs(r["age"] - ages[i])
        ok = r["gender"] == true_g
        errs.append(err)
        correct.append(ok)
        print(f"{k:>3}  {ages[i]:>8.0f}  {r['age']:>8.1f}  {err:>5.1f}  {true_g:<11}  "
              f"{r['gender']} ({r['gender_confidence']:.0%}){'' if ok else '  <- wrong'}")
        titles.append(f"true: {ages[i]:.0f}, {true_g}\npred: {r['age']:.0f}, {r['gender']}")
        colors.append("green" if ok and err <= 5 else "darkorange" if ok else "red")
    print(f"Sample age MAE {np.mean(errs):.2f} years, gender accuracy {np.mean(correct):.0%}")
    save_grid(crops, titles, colors, os.path.join(OUT_DIR, "utkface_test.jpg"))


def demo_fer(analyzer, n, seed):
    paths, labels = fer_splits()["test"]
    idx = random.Random(seed).sample(range(len(paths)), n)
    crops = [cv2.imread(paths[i]) for i in idx]
    results = analyzer.predict_crops(crops, crops)

    print(f"FER2013 test set: {n} random images")
    print(f"{'#':>3}  {'true':<9}  {'predicted':<16}  top-3")
    titles, colors, correct = [], [], []
    for k, (i, r) in enumerate(zip(idx, results), 1):
        true_e = config.EMOTION_LABELS[labels[i]]
        ok = r["emotion"] == true_e
        correct.append(ok)
        top3 = sorted(r["emotion_probs"].items(), key=lambda kv: -kv[1])[:3]
        top3 = ", ".join(f"{e} {p:.0%}" for e, p in top3)
        print(f"{k:>3}  {true_e:<9}  {r['emotion']} ({r['emotion_confidence']:.0%})".ljust(34) + f"  {top3}"
              + ("" if ok else "  <- wrong"))
        titles.append(f"true: {true_e}\npred: {r['emotion']} {r['emotion_confidence']:.0%}")
        colors.append("green" if ok else "red")
    print(f"Sample accuracy {np.mean(correct):.0%}")
    save_grid([cv2.resize(c, (144, 144), interpolation=cv2.INTER_CUBIC) for c in crops],
              titles, colors, os.path.join(OUT_DIR, "fer_test.jpg"))


def demo_images(analyzer, target):
    if os.path.isdir(target):
        files = sorted(f for f in glob.glob(os.path.join(target, "*")) if f.lower().endswith(IMAGE_EXTS))
    else:
        files = [target]
    if not files:
        raise SystemExit(f"No images found at {target}")

    for path in files:
        image = cv2.imread(path)
        if image is None:
            print(f"{path}: cannot read, skipped")
            continue
        results = analyzer.analyze(image)
        print(f"{path}: {len(results)} face(s)")
        for k, r in enumerate(results, 1):
            top3 = sorted(r["emotion_probs"].items(), key=lambda kv: -kv[1])[:3]
            print(f"  face {k}: age {r['age']:.0f}, {r['gender']} ({r['gender_confidence']:.0%}), "
                  f"{r['emotion']} ({r['emotion_confidence']:.0%})  |  "
                  + ", ".join(f"{e} {p:.0%}" for e, p in top3))
        out = os.path.join(OUT_DIR, "result_" + os.path.splitext(os.path.basename(path))[0] + ".jpg")
        cv2.imwrite(out, draw_results(image, results))
        print(f"  -> saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", help="an image file or a folder of images to analyse")
    parser.add_argument("--dataset", choices=["utkface", "fer", "both"], default="both",
                        help="which test set to sample from (ignored with --images)")
    parser.add_argument("-n", type=int, default=12, help="number of random test images per dataset")
    parser.add_argument("--seed", type=int, default=0, help="change to see different random images")
    parser.add_argument("--no-tta", action="store_true", help="disable flip test-time augmentation")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    analyzer = FaceAnalyzer(tta=not args.no_tta)
    if args.images:
        demo_images(analyzer, args.images)
        return
    if args.dataset in ("utkface", "both"):
        demo_utkface(analyzer, args.n, args.seed)
    if args.dataset in ("fer", "both"):
        demo_fer(analyzer, args.n, args.seed)


if __name__ == "__main__":
    main()
