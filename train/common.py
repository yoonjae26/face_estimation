"""Training helpers shared by the train_* scripts."""
import json
import os
import sys

# Shared server: only ever use GPU 0 unless the caller explicitly chose a device.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras  # noqa: E402
import matplotlib  # noqa: E402
import tensorflow as tf  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils import config  # noqa: E402

HISTORY_DIR = os.path.join(config.ROOT_DIR, "history")


def setup(mixed_precision=True, threads=16):
    """Grow GPU memory on demand (don't grab the whole card) and cap CPU threads."""
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.threading.set_inter_op_parallelism_threads(threads)
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    if mixed_precision and tf.config.list_physical_devices("GPU"):
        keras.mixed_precision.set_global_policy("mixed_float16")
    keras.utils.set_random_seed(config.SEED)
    print("GPUs visible:", tf.config.list_physical_devices("GPU"))


def fit_two_phase(model, backbone, train_ds, val_ds, compile_kwargs, model_path, monitor, mode,
                  head_epochs=3, epochs=30, lr=3e-4, weight_decay=1e-4, patience=8):
    """Phase 1: train only the new head with the pretrained backbone frozen.
    Phase 2: fine-tune everything with AdamW + warmup/cosine decay, keeping the best checkpoint.
    Returns the merged history dict.
    """
    history = {}

    def merge(h):
        for k, v in h.history.items():
            history.setdefault(k, []).extend(float(x) for x in v)

    backbone.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-3), **compile_kwargs)
    merge(model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, verbose=2))

    backbone.trainable = True
    steps = len(train_ds) * epochs
    schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr / 10, decay_steps=steps - len(train_ds), alpha=0.01,
        warmup_target=lr, warmup_steps=len(train_ds))
    model.compile(optimizer=keras.optimizers.AdamW(schedule, weight_decay=weight_decay), **compile_kwargs)
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, monitor=monitor, mode=mode, save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor=monitor, mode=mode, patience=patience, restore_best_weights=True,
                                      verbose=1),
    ]
    merge(model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2))
    history["phase2_start_epoch"] = head_epochs
    return history


def save_history(history, name, plots):
    """Write history/<name>/history.json and a PNG with one subplot per (train, val) metric pair."""
    out_dir = os.path.join(HISTORY_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    start = history.get("phase2_start_epoch")
    fig, axes = plt.subplots(1, len(plots), figsize=(6 * len(plots), 4.5))
    for ax, (key, title) in zip(axes if len(plots) > 1 else [axes], plots):
        ax.plot(history[key], label="train", color="#2a6fdb")
        ax.plot(history["val_" + key], label="validation", color="#e8710a")
        if start:
            ax.axvline(start - 0.5, color="gray", linestyle=":", label="unfreeze backbone")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "training_curves.png"), dpi=110)
    plt.close(fig)
