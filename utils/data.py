"""Dataset parsing, reproducible splits and tf.data pipelines."""
import glob
import os

import numpy as np
from sklearn.model_selection import train_test_split

from utils import config


def parse_utkface_filename(filename):
    """'<age>_<gender>_<race>_<date>.jpg.chip.jpg' -> (age, gender) or None if malformed."""
    parts = os.path.basename(filename).split("_")
    if len(parts) < 4:
        return None
    try:
        age, gender = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if gender not in (0, 1) or not 0 < age <= 116:
        return None
    return min(age, config.MAX_AGE), gender


def utkface_splits(data_dir=config.UTKFACE_DIR, val_frac=0.1, test_frac=0.1, seed=config.SEED):
    """Split UTKFace into train/val/test (80/10/10), stratified by age decade and gender.

    Returns {split: (paths, ages, genders)} as numpy arrays.
    """
    paths, ages, genders = [], [], []
    for p in sorted(glob.glob(os.path.join(data_dir, "*.jpg"))):
        parsed = parse_utkface_filename(p)
        if parsed is None:
            continue
        paths.append(p)
        ages.append(parsed[0])
        genders.append(parsed[1])
    paths, ages, genders = np.array(paths), np.array(ages, np.float32), np.array(genders, np.float32)
    if len(paths) == 0:
        raise FileNotFoundError(f"No UTKFace images found in {data_dir}")

    strata = np.minimum(ages // 10, 8).astype(int) * 2 + genders.astype(int)
    idx = np.arange(len(paths))
    train_idx, hold_idx = train_test_split(idx, test_size=val_frac + test_frac, stratify=strata, random_state=seed)
    val_idx, test_idx = train_test_split(hold_idx, test_size=test_frac / (val_frac + test_frac),
                                         stratify=strata[hold_idx], random_state=seed)
    return {name: (paths[i], ages[i], genders[i]) for name, i in
            [("train", train_idx), ("val", val_idx), ("test", test_idx)]}


def _list_fer(folder):
    paths, labels = [], []
    for label, name in enumerate(config.EMOTION_LABELS):
        files = sorted(glob.glob(os.path.join(folder, name.lower(), "*.jpg")))
        paths += files
        labels += [label] * len(files)
    return np.array(paths), np.array(labels, np.int32)


def fer_splits(data_dir=config.FER_DIR, val_frac=0.1, seed=config.SEED):
    """FER2013: official train split -> train/val (90/10 stratified); official test split -> test."""
    paths, labels = _list_fer(os.path.join(data_dir, "train"))
    if len(paths) == 0:
        raise FileNotFoundError(f"No FER2013 images found in {data_dir}/train")
    tr_p, va_p, tr_l, va_l = train_test_split(paths, labels, test_size=val_frac, stratify=labels, random_state=seed)
    te_p, te_l = _list_fer(os.path.join(data_dir, "test"))
    return {"train": (tr_p, tr_l), "val": (va_p, va_l), "test": (te_p, te_l)}


def _augmenter(strength, erasing):
    import keras
    f32 = dict(dtype="float32")  # augmentation runs in the input pipeline, never in mixed precision
    augs = [
        keras.layers.RandomFlip("horizontal", **f32),
        keras.layers.RandomRotation(0.04 * strength, fill_mode="reflect", **f32),
        keras.layers.RandomZoom((-0.1 * strength, 0.1 * strength), fill_mode="reflect", **f32),
        keras.layers.RandomTranslation(0.06 * strength, 0.06 * strength, fill_mode="reflect", **f32),
        keras.layers.RandomBrightness(0.2 * strength, value_range=(0, 255), **f32),
        keras.layers.RandomContrast(0.2 * strength, value_range=(0, 255), **f32),
    ]
    if erasing:
        augs.append(keras.layers.RandomErasing(factor=0.5, scale=(0.02, 0.2), value_range=(0, 255), **f32))
    return keras.Sequential(augs, name="augment")


def make_dataset(paths, labels, img_size, batch_size, training, grayscale=False, aug_strength=1.0,
                 erasing=False, mixup=0.0):
    """Build a tf.data pipeline yielding (float32 RGB images in [0, 255], labels).

    `labels` may be an array or a dict of arrays (multi-output models).
    Decoded images are cached in memory, so decoding happens once.
    `erasing` adds random erasing; `mixup` > 0 applies MixUp (one-hot labels only) with that alpha.
    """
    import tensorflow as tf

    def load(path):
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=1 if grayscale else 3)
        if grayscale:
            img = tf.image.grayscale_to_rgb(img)
        img = tf.image.resize(img, (img_size, img_size), method="area", antialias=True)
        return tf.clip_by_value(img, 0.0, 255.0)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (load(p), y), num_parallel_calls=16).cache()
    if training:
        ds = ds.shuffle(min(len(paths), 20000), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=training)
    if training:
        aug = _augmenter(aug_strength, erasing)
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0.0, 255.0), y), num_parallel_calls=16)
        if mixup:
            import keras
            mix = keras.layers.MixUp(alpha=mixup, dtype="float32")

            def apply_mixup(x, y):
                out = mix({"images": x, "labels": y}, training=True)
                return out["images"], out["labels"]
            ds = ds.map(apply_mixup, num_parallel_calls=16)
    return ds.prefetch(tf.data.AUTOTUNE)
