import os
from tensorflow.keras.models import load_model
import numpy as np
import cv2

class EmotionPredictor:
    """Class for emotion prediction from face images"""
    
    # Map emotion indices to labels
    EMOTION_LABELS = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }
    
    def __init__(self, model_path=None):
        """
        Initialize EmotionPredictor
        
        Args:
            model_path (str): Path to emotion model file. If None, uses default path
        """
        if model_path is None:
            model_path = os.path.join("models", "emotion_model.h5")
            
        try:
            self.model = load_model(model_path)
            self.input_shape = self.model.input_shape[1:3]  # Get expected input size
        except Exception as e:
            raise Exception(f"Error loading emotion model: {str(e)}")
    
    def preprocess_image(self, face_img):
        """
        Preprocess face image for emotion prediction
        
        Args:
            face_img (numpy.ndarray): Input face image (BGR format)
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            if face_img is None:
                raise ValueError("Invalid input image")
                
            # Convert to grayscale if image is BGR
            if len(face_img.shape) == 3:
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                
            # Resize to expected input size
            face_img = cv2.resize(face_img, self.input_shape)
            
            # Normalize pixel values
            face_img = face_img.astype('float32') / 255.0
            
            # Add channel and batch dimensions
            face_img = np.expand_dims(face_img, axis=-1)
            face_img = np.expand_dims(face_img, axis=0)
            
            return face_img
            
        except Exception as e:
            raise Exception(f"Error in image preprocessing: {str(e)}")
    
    def predict_emotion(self, face_img, return_confidence=False):
        """
        Predict emotion from face image
        
        Args:
            face_img (numpy.ndarray): Input face image (BGR format)
            return_confidence (bool): Whether to return confidence scores
            
        Returns:
            str or tuple: Predicted emotion label, or (emotion, confidence) if return_confidence=True
        """
        try:
            # Preprocess image
            processed_img = self.preprocess_image(face_img)
            
            # Get prediction
            prediction = self.model.predict(processed_img, verbose=0)[0]
            emotion_idx = np.argmax(prediction)
            confidence = float(prediction[emotion_idx])
            
            # Get emotion label
            emotion = self.EMOTION_LABELS[emotion_idx]
            
            if return_confidence:
                return emotion, confidence
            return emotion
            
        except Exception as e:
            raise Exception(f"Error in emotion prediction: {str(e)}")
    
    def predict_emotion_probabilities(self, face_img):
        """
        Get probabilities for all emotions
        
        Args:
            face_img (numpy.ndarray): Input face image (BGR format)
            
        Returns:
            dict: Dictionary mapping emotion labels to their probabilities
        """
        try:
            # Preprocess image
            processed_img = self.preprocess_image(face_img)
            
            # Get predictions
            predictions = self.model.predict(processed_img, verbose=0)[0]
            
            # Create dictionary of emotion probabilities
            emotion_probs = {
                label: float(prob) 
                for label, prob in zip(self.EMOTION_LABELS.values(), predictions)
            }
            
            return emotion_probs
            
        except Exception as e:
            raise Exception(f"Error in probability prediction: {str(e)}")

    def get_emotion_color(self, emotion):
        """
        Get color for visualization based on emotion
        
        Args:
            emotion (str): Emotion label
            
        Returns:
            tuple: BGR color values
        """
        # Define colors for each emotion (in BGR format)
        emotion_colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 140, 255),  # Orange
            'Fear': (0, 255, 255),     # Yellow
            'Happy': (0, 255, 0),      # Green
            'Sad': (255, 0, 0),        # Blue
            'Surprise': (255, 0, 255), # Purple
            'Neutral': (128, 128, 128)  # Gray
        }
        
        return emotion_colors.get(emotion, (255, 255, 255))  # White as default
