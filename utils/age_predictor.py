from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

# Constants
IMG_SIZE = 64
MODEL_PATH = "models"

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
            self.age_model = load_model(self.age_model_path)
            self.gender_model = load_model(self.gender_model_path)
        except Exception as e:
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
                
            if len(img.shape) != 3:
                raise ValueError("Expected 3 channel image")
                
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