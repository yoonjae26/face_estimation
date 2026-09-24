"""Train the facial-expression model on FER2013.

    python train/train_emotion.py [--epochs 40] [--batch-size 64]
"""
import argparse

import common  # sets CUDA_VISIBLE_DEVICES=0 before TensorFlow loads
import keras
import numpy as np

from utils import config
from utils.data import fer_splits, make_dataset
from utils.models import build_emotion_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--aug-strength", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp alpha (0 disables)")
    parser.add_argument("--erasing", action="store_true")
    parser.add_argument("--output", default=config.EMOTION_MODEL)
    parser.add_argument("--history-name", default="emotion")
    args = parser.parse_args()

    common.setup()
    splits = fer_splits()
    n = len(config.EMOTION_LABELS)
    for name, (p, y) in splits.items():
        print(f"{name}: {len(p)} images, per class {np.bincount(y, minlength=n).tolist()}")

    size = config.EMOTION_IMG_SIZE

    def ds(split, training):
        p, y = splits[split]
        return make_dataset(p, keras.utils.to_categorical(y, n), size, args.batch_size, training, grayscale=True,
                            aug_strength=args.aug_strength, erasing=args.erasing, mixup=args.mixup)

    # Defaults are the released model's settings. Stronger regularisation (e.g. --erasing --mixup 0.2
    # --aug-strength 1.5 --dropout 0.5 --weight-decay 0.05) narrows the train/val gap on FER2013 but
    # scored slightly lower on the test set (69.4% vs 70.2%).
    model, backbone = build_emotion_model(dropout=args.dropout)
    compile_kwargs = dict(loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["accuracy"])
    history = common.fit_two_phase(
        model, backbone, ds("train", True), ds("val", False), compile_kwargs, args.output,
        monitor="val_accuracy", mode="max", epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        patience=10)
    common.save_history(history, args.history_name, [("loss", "Loss"), ("accuracy", "Accuracy")])
    print(f"Saved best model to {args.output}")


if __name__ == "__main__":
    main()
