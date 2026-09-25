"""Evaluate the trained models on the held-out test splits and write history/test_metrics.json.

    python evaluate.py [--no-tta]
"""
import argparse
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import utils.models  # noqa: E402,F401  registers custom layers
from utils import config  # noqa: E402
from utils.data import age_crops_splits, fer_splits, make_dataset, utkface_splits  # noqa: E402

OUT_DIR = os.path.join(config.ROOT_DIR, "history")


def predict(model, ds, tta):
    out = model.predict(ds, verbose=0)
    if tta:
        flipped = model.predict(ds.map(lambda x, y: (x[:, :, ::-1], y)), verbose=0)
        out = {k: (out[k] + flipped[k]) / 2 for k in out} if isinstance(out, dict) else (out + flipped) / 2
    return out


def _age_stats(pred, ages):
    err = pred - ages
    return {"n": int(len(ages)), "mae": round(float(np.abs(err).mean()), 3), "bias": round(float(err.mean()), 2),
            "within_5_years": round(float((np.abs(err) <= 5).mean()), 4)}


def eval_age_gender(model_path, tta):
    """Test on every source of the combined dataset (UTKFace only if it hasn't been built)."""
    try:
        paths, ages, genders, sources = age_crops_splits()["test"]
    except FileNotFoundError:
        paths, ages, genders = utkface_splits()["test"]
        sources = np.array(["utkface"] * len(paths))
    ds = make_dataset(paths, ages, config.AGE_GENDER_IMG_SIZE, 128, training=False)
    model = keras.models.load_model(model_path, compile=False)
    out = predict(model, ds, tta)
    pred_age, p_female = out["age"][:, 0], out["gender"][:, 0]

    groups = {}
    for lo, hi in [(0, 12), (13, 19), (20, 29), (30, 39), (40, 49), (50, 59), (60, 69), (70, 101)]:
        m = (ages >= lo) & (ages <= hi)
        groups[f"{lo}-{min(hi, 100)}"] = {"n": int(m.sum()), "mae": round(float(np.abs(pred_age - ages)[m].mean()), 2)}
    per_source = {}
    for src in np.unique(sources):
        m = sources == src
        per_source[str(src)] = _age_stats(pred_age[m], ages[m])
        labelled = m & (genders >= 0)
        if labelled.any():
            acc = ((p_female[labelled] >= 0.5) == (genders[labelled] == 1)).mean()
            per_source[str(src)]["gender_accuracy"] = round(float(acc), 4)
    labelled = genders >= 0
    return {
        "all_sources": _age_stats(pred_age, ages),
        "per_source": per_source,
        "age_mae_by_group": groups,
        "gender_accuracy": round(float(((p_female[labelled] >= 0.5) == (genders[labelled] == 1)).mean()), 4),
    }, (ages, pred_age)


def eval_emotion(model_path, tta):
    paths, labels = fer_splits()["test"]
    ds = make_dataset(paths, labels, config.EMOTION_IMG_SIZE, 256, training=False, grayscale=True)
    model = keras.models.load_model(model_path, compile=False)
    pred = predict(model, ds, tta).argmax(axis=1)
    report = classification_report(labels, pred, target_names=config.EMOTION_LABELS, output_dict=True, digits=4)
    return {
        "test_images": int(len(labels)),
        "accuracy": round(float((pred == labels).mean()), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "per_class_f1": {c: round(float(report[c]["f1-score"]), 4) for c in config.EMOTION_LABELS},
    }, confusion_matrix(labels, pred, normalize="true")


def plot(age_data, cm):
    ages, pred = age_data
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    ax.scatter(ages, pred, s=3, alpha=0.15, color="#2a6fdb")
    ax.plot([0, 100], [0, 100], color="#e8710a", linewidth=1.5, label="perfect")
    ax.set(xlabel="true age", ylabel="predicted age", title="Age: test sets (all sources)", xlim=(0, 100), ylim=(0, 100))
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    n = len(config.EMOTION_LABELS)
    ax.set_xticks(range(n), config.EMOTION_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(n), config.EMOTION_LABELS)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set(xlabel="predicted", ylabel="true", title="Emotion: FER2013 test (row-normalised)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "test_results.png"), dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument("--age-gender-model", default=config.AGE_GENDER_MODEL)
    parser.add_argument("--emotion-model", default=config.EMOTION_MODEL)
    parser.add_argument("--no-save", action="store_true", help="only print, don't overwrite history/test_metrics.json")
    args = parser.parse_args()
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    tta = not args.no_tta
    ag, age_data = eval_age_gender(args.age_gender_model, tta)
    em, cm = eval_emotion(args.emotion_model, tta)
    metrics = {"tta": tta, "age_gender": ag, "emotion": em}
    print(json.dumps(metrics, indent=2))
    if not args.no_save:
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(os.path.join(OUT_DIR, "test_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        plot(age_data, cm)


if __name__ == "__main__":
    main()
