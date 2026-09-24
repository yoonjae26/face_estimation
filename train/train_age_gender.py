"""Train the multi-task age + gender model on UTKFace.

    python train/train_age_gender.py [--epochs 40] [--batch-size 64]
"""
import argparse

import common  # sets CUDA_VISIBLE_DEVICES=0 before TensorFlow loads
import keras

from utils import config
from utils.data import make_dataset, utkface_splits
from utils.models import build_age_gender_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--aug-strength", type=float, default=1.5)
    parser.add_argument("--no-erasing", action="store_true")
    parser.add_argument("--output", default=config.AGE_GENDER_MODEL)
    parser.add_argument("--history-name", default="age_gender")
    args = parser.parse_args()

    common.setup()
    splits = utkface_splits()
    for name, (p, a, g) in splits.items():
        print(f"{name}: {len(p)} images, mean age {a.mean():.1f}, female {g.mean():.2%}")

    size = config.AGE_GENDER_IMG_SIZE

    def ds(split, training):
        p, a, g = splits[split]
        return make_dataset(p, {"age": a, "gender": g}, size, args.batch_size, training,
                            aug_strength=args.aug_strength, erasing=not args.no_erasing)

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
        weight_decay=args.weight_decay)
    common.save_history(history, args.history_name, [
        ("loss", "Total loss"), ("age_mae", "Age MAE (years)"), ("gender_accuracy", "Gender accuracy")])
    print(f"Saved best model to {args.output}")


if __name__ == "__main__":
    main()
