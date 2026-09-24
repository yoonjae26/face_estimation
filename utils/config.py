"""Shared constants so training and inference always agree on preprocessing and labels."""
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(ROOT_DIR, "models"))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(ROOT_DIR, "data"))

UTKFACE_DIR = os.path.join(DATA_DIR, "utkface", "UTKFace")
FER_DIR = os.path.join(DATA_DIR, "fer2013")

AGE_GENDER_MODEL = os.path.join(MODEL_DIR, "age_gender_model.keras")
EMOTION_MODEL = os.path.join(MODEL_DIR, "emotion_model.keras")
YUNET_MODEL = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")

# Input sizes (square). Images are fed as RGB uint8-range floats [0, 255];
# the EfficientNetV2 backbones do their own normalisation internally.
AGE_GENDER_IMG_SIZE = 160
EMOTION_IMG_SIZE = 112

MAX_AGE = 100  # ages are clipped to [0, MAX_AGE]; the age head is a softmax over MAX_AGE + 1 bins

# UTKFace: gender 0 = male, 1 = female. The model predicts P(female).
GENDER_LABELS = ["Male", "Female"]

# Must match the alphabetical folder order of FER2013 (train/<emotion>/...).
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# How much to enlarge a detected face box before cropping, so crops look like the
# training images. Both datasets are tightly cropped (median ratio ~1.02-1.03), calibrated by
# running the detector over the training sets (see train/calibrate_crops.py).
AGE_GENDER_CROP_SCALE = 1.03
EMOTION_CROP_SCALE = 1.02

SEED = 42
