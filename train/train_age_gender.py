"""Train the multi-task age + gender model.

By default it uses the combined dataset (UTKFace + AFAD + MegaAge-Asian + APPA-REAL, built by
train/prepare_age_data.py). Training on UTKFace alone generalised poorly to other sources
(e.g. MAE 7-8.5 years on AFAD / APPA-REAL test images) and to real webcams.

    python train/prepare_age_data.py                 # once
    python train/train_age_gender.py [--epochs 25]
    python train/train_age_gender.py --data utkface  # the previous UTKFace-only setup
"""
import argparse

import common  # sets CUDA_VISIBLE_DEVICES=0 before TensorFlow loads
import keras
import numpy as np

from utils import config
from utils.data import age_crops_splits, make_dataset, utkface_splits
from utils.models import build_age_gender_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["combined", "utkface"], default="combined")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--aug-strength", type=float, default=1.5)
    parser.add_argument("--no-erasing", action="store_true")
    parser.add_argument("--degrade", type=float, default=0.3,
                        help="probability of each simulated low-quality-camera corruption")
    parser.add_argument("--output", default=config.AGE_GENDER_MODEL)
    parser.add_argument("--history-name", default="age_gender")
    args = parser.parse_args()

    common.setup()
    if args.data == "combined":
        splits = {k: v[:3] for k, v in age_crops_splits().items()}
        for name, (p, a, g, src) in age_crops_splits().items():
            counts = {s: int((src == s).sum()) for s in np.unique(src)}
            print(f"{name}: {len(p)} images, mean age {a.mean():.1f}, {counts}")
    else:
        splits = utkface_splits()

    size = config.AGE_GENDER_IMG_SIZE

    def ds(split, training):
        p, a, g = splits[split]
        # Sources without gender labels (gender = -1) get zero weight in the gender loss/metric.
        weights = {"age": np.ones_like(a), "gender": (g >= 0).astype(np.float32)}
        return make_dataset(p, {"age": a, "gender": np.maximum(g, 0)}, size, args.batch_size, training,
                            aug_strength=args.aug_strength, erasing=not args.no_erasing,
                            sample_weights=weights, degrade=args.degrade)

    model, backbone = build_age_gender_model(dropout=args.dropout)
    compile_kwargs = dict(
        # L1 on the expected age, BCE (lightly smoothed) for gender; age loss is in years so
        # it is down-weighted to keep both tasks on a similar scale.
        loss={"age": "mae", "gender": keras.losses.BinaryCrossentropy(label_smoothing=0.05)},
        loss_weights={"age": 0.1, "gender": 1.0},
        metrics={"age": ["mae"], "gender": ["accuracy"]},
    )
    history = common.fit_two_phase(
        model, backbone, ds("train", True), ds("val", False), compile_kwargs,
        args.output, monitor="val_age_mae", mode="min", epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, patience=6)
    common.save_history(history, args.history_name, [
        ("loss", "Total loss"), ("age_mae", "Age MAE (years)"), ("gender_accuracy", "Gender accuracy")])
    print(f"Saved best model to {args.output}")


if __name__ == "__main__":
    main()
