"""Model definitions. Both models take RGB float images in [0, 255] (normalisation is built in)."""
import keras
from keras import layers, ops

from utils import config


@keras.saving.register_keras_serializable(package="face_estimation")
class AgeExpectation(layers.Layer):
    """Turns a softmax over ages 0..MAX_AGE into its expected value (in years).

    Classifying over age bins and taking the expectation (DEX-style) is much more
    stable than plain regression and never predicts ages outside [0, MAX_AGE].
    """

    def call(self, probs):
        ages = ops.arange(ops.shape(probs)[-1], dtype=probs.dtype)
        return ops.sum(probs * ages, axis=-1, keepdims=True)


def _backbone(img_size):
    return keras.applications.EfficientNetV2B0(
        include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3),
        pooling="avg", include_preprocessing=True)


def build_age_gender_model(img_size=config.AGE_GENDER_IMG_SIZE, dropout=0.3):
    """Multi-task model: outputs {'age': years (N, 1), 'gender': P(female) (N, 1)}."""
    inputs = keras.Input((img_size, img_size, 3), name="image")
    backbone = _backbone(img_size)
    feats = layers.Dropout(dropout)(backbone(inputs))

    age_probs = layers.Dense(config.MAX_AGE + 1, activation="softmax", dtype="float32", name="age_probs")(feats)
    age = AgeExpectation(dtype="float32", name="age")(age_probs)
    gender = layers.Dense(1, activation="sigmoid", dtype="float32", name="gender")(feats)
    return keras.Model(inputs, {"age": age, "gender": gender}, name="age_gender"), backbone


def build_emotion_model(img_size=config.EMOTION_IMG_SIZE, num_classes=len(config.EMOTION_LABELS), dropout=0.4):
    inputs = keras.Input((img_size, img_size, 3), name="image")
    backbone = _backbone(img_size)
    feats = layers.Dropout(dropout)(backbone(inputs))
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="emotion")(feats)
    return keras.Model(inputs, outputs, name="emotion"), backbone
