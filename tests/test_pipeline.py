"""Fast checks that need neither the datasets nor the trained models."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import config  # noqa: E402
from utils.data import parse_utkface_filename  # noqa: E402
from utils.face_detector import crop_face, to_model_input  # noqa: E402


@pytest.mark.parametrize("name, expected", [
    ("26_1_2_20170116174525125.jpg.chip.jpg", (26, 1)),
    ("1_0_0_20161219140623097.jpg.chip.jpg", (1, 0)),
    ("116_1_0_20170120134921760.jpg.chip.jpg", (config.MAX_AGE, 1)),  # clipped
    ("39_1_20170116174525125.jpg.chip.jpg", None),  # known malformed UTKFace file (missing race)
    ("abc_0_0_x.jpg", None),
    ("30_3_0_x.jpg", None),
])
def test_parse_utkface_filename(name, expected):
    assert parse_utkface_filename(name) == expected


def test_emotion_labels_match_fer_folder_order():
    # Keras/sklearn loaders and our own lister use alphabetical folder order; a mismatch
    # here silently swaps predicted emotions (the bug in the first version of this project).
    assert [e.lower() for e in config.EMOTION_LABELS] == sorted(e.lower() for e in config.EMOTION_LABELS)


def test_crop_face_is_square_and_centred_even_at_border():
    img = np.zeros((100, 200, 3), np.uint8)
    img[40:60, 0:20] = 255  # a "face" touching the left edge
    crop = crop_face(img, (0, 40, 20, 20), scale=2.0)
    assert crop.shape == (40, 40, 3)
    assert crop[20, 20].tolist() == [255, 255, 255]


def test_to_model_input_rgb_order_and_range():
    face = np.zeros((50, 50, 3), np.uint8)
    face[..., 0] = 255  # pure blue in BGR
    x = to_model_input(face, 32)
    assert x.shape == (32, 32, 3) and x.dtype == np.float32
    assert x[..., 2].min() == 255 and x[..., 0].max() == 0  # blue ends up in the last (B of RGB) channel
    g = to_model_input(face, 32, grayscale=True)
    assert np.allclose(g[..., 0], g[..., 1]) and np.allclose(g[..., 1], g[..., 2])


def test_age_expectation_layer():
    keras = pytest.importorskip("keras")
    from utils.models import AgeExpectation

    probs = np.zeros((2, config.MAX_AGE + 1), np.float32)
    probs[0, 30] = 1.0
    probs[1, 20] = probs[1, 40] = 0.5
    out = keras.ops.convert_to_numpy(AgeExpectation()(probs))
    assert np.allclose(out[:, 0], [30, 30])
