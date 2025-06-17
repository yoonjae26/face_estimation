from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import cv2
import numpy as np
import os
import h5py

# Constants
IMG_SIZE = 64
MODEL_PATH = os.getenv("MODEL_PATH", "models")

def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma)
        loss = weight * cross_entropy
        return K.sum(loss, axis=1)
    return focal_loss_fixed

class AgeGenderPredictor:
    def __init__(self, age_model_path=None, gender_model_path=None):
        """
        Initialize the Age and Gender predictor
        
        Args:
            age_model_path (str): Path to the age prediction model
            gender_model_path (str): Path to the gender prediction model
        """
        self.age_model_path = age_model_path or os.path.join(MODEL_PATH, "age_model.h5")
        self.gender_model_path = gender_model_path or os.path.join(MODEL_PATH, "gender_model.h5")
        
        try:
            print(f"Loading age model from: {self.age_model_path}")
            self.age_model = load_model(self.age_model_path, custom_objects={'focal_loss': focal_loss()}, compile=False)
            print("Age model loaded successfully.")

            print(f"Loading gender model from: {self.gender_model_path}")
            self.gender_model = load_model(self.gender_model_path, custom_objects={'focal_loss': focal_loss()}, compile=False)
            print("Gender model loaded successfully.")
        except Exception as e:
            print(f"Error details: {str(e)}")
            raise Exception(f"Error loading models: {str(e)}")

    def preprocess_image(self, img):
        """
        Preprocess image for model input
        
        Args:
            img (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            if img is None:
                raise ValueError("Invalid input image")

            # Ensure the image has 3 channels (convert grayscale to RGB if necessary)
            if len(img.shape) == 2 or img.shape[2] != 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            # Resize image
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            
            # Normalize pixel values
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            return img_batch
            
        except Exception as e:
            raise Exception(f"Error in image preprocessing: {str(e)}")

    def predict_age(self, img):
        """
        Predict age from face image
        
        Args:
            img (numpy.ndarray): Input face image
            
        Returns:
            int: Predicted age
        """
        try:
            img_processed = self.preprocess_image(img)
            age_pred = self.age_model.predict(img_processed, verbose=0)[0][0]
            return int(age_pred)
            
        except Exception as e:
            raise Exception(f"Error in age prediction: {str(e)}")

    def predict_gender(self, img):
        """
        Predict gender from face image
        
        Args:
            img (numpy.ndarray): Input face image
            
        Returns:
            tuple: (predicted_gender, confidence_score)
        """
        try:
            img_processed = self.preprocess_image(img)
            gender_prob = self.gender_model.predict(img_processed, verbose=0)[0]
            confidence = float(max(gender_prob[0], 1 - gender_prob[0]))
            gender = "Male" if gender_prob[0] > 0.5 else "Female"
            
            return gender, confidence
            
        except Exception as e:
            raise Exception(f"Error in gender prediction: {str(e)}")

    def predict_age_gender(self, face_img):
        """
        Predict both age and gender from face image
        
        Args:
            face_img (numpy.ndarray): Input face image
            
        Returns:
            tuple: (predicted_age, gender, gender_confidence)
        """
        try:
            age = self.predict_age(face_img)
            gender, confidence = self.predict_gender(face_img)
            
            return age, gender, confidence
            
        except Exception as e:
            raise Exception(f"Error in age and gender prediction: {str(e)}")