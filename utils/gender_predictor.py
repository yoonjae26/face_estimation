import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

class GenderPredictor:
    """Class for gender prediction from face images"""
    
    GENDER_LABELS = {
        0: 'Male',
        1: 'Female'
    }
    
    def __init__(self, model_path=None, use_mtcnn=False):
        """
        Initialize GenderPredictor
        
        Args:
            model_path (str): Path to gender model file. If None, uses default path
            use_mtcnn (bool): Whether to use MTCNN for face detection (more accurate but slower)
        """
        # Load gender prediction model
        if model_path is None:
            model_path = os.path.join("models", "gender_model.h5")
            
        try:
            print(f"Loading gender model from: {model_path}")
            self.model = load_model(model_path, compile=False)
            print("Gender model loaded successfully.")
            self.input_shape = self.model.input_shape[1:3]  # Get expected input size
        except Exception as e:
            print(f"Error details: {str(e)}")
            print("Inspecting model file structure...")
            import h5py
            with h5py.File(model_path, 'r') as f:
                print("Model file keys:", list(f.keys()))
                if 'model_config' in f:
                    print("Model config:", f['model_config'][:])
            raise Exception(f"Error loading gender model: {str(e)}")
            
        # Initialize face detector
        if use_mtcnn:
            try:
                from mtcnn import MTCNN
                self.detector = MTCNN()
                self.use_mtcnn = True
            except ImportError:
                print("MTCNN not available. Falling back to Haar Cascade")
                self.use_mtcnn = False
        else:
            self.use_mtcnn = False
            
        if not self.use_mtcnn:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
    
    def detect_faces(self, frame):
        """
        Detect faces in frame
        
        Args:
            frame (numpy.ndarray): Input frame/image
            
        Returns:
            list: List of (x, y, w, h) face rectangles
        """
        if self.use_mtcnn:
            try:
                results = self.detector.detect_faces(frame)
                return [res['box'] for res in results]
            except Exception as e:
                raise Exception(f"Error in MTCNN face detection: {str(e)}")
        else:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                return faces
            except Exception as e:
                raise Exception(f"Error in Haar Cascade face detection: {str(e)}")
    
    def preprocess_image(self, face_img):
        """
        Preprocess face image for gender prediction
        
        Args:
            face_img (numpy.ndarray): Input face image (BGR format)
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        try:
            if face_img is None:
                raise ValueError("Invalid input image")
                
            # Resize to expected input size
            face_img = cv2.resize(face_img, self.input_shape)
            
            # Normalize pixel values
            face_img = face_img.astype('float32') / 255.0
            
            # Add batch dimension
            face_img = np.expand_dims(face_img, axis=0)
            
            return face_img
            
        except Exception as e:
            raise Exception(f"Error in image preprocessing: {str(e)}")
    
    def predict_gender(self, face_img, return_confidence=False):
        """
        Predict gender from face image
        
        Args:
            face_img (numpy.ndarray): Input face image (BGR format)
            return_confidence (bool): Whether to return confidence score
            
        Returns:
            str or tuple: Predicted gender label, or (gender, confidence) if return_confidence=True
        """
        try:
            # Preprocess image
            processed_img = self.preprocess_image(face_img)
            
            # Get prediction
            prediction = self.model.predict(processed_img, verbose=0)[0]
            confidence = float(prediction[0])
            gender_idx = 0 if confidence > 0.5 else 1
            
            # Convert confidence to be relative to predicted class
            confidence = confidence if gender_idx == 0 else 1 - confidence
            
            # Get gender label
            gender = self.GENDER_LABELS[gender_idx]
            
            if return_confidence:
                return gender, confidence
            return gender
            
        except Exception as e:
            raise Exception(f"Error in gender prediction: {str(e)}")
    
    def get_gender_color(self, gender):
        """
        Get color for visualization based on gender
        
        Args:
            gender (str): Gender label
            
        Returns:
            tuple: BGR color values
        """
        gender_colors = {
            'Male': (255, 0, 0),    # Blue
            'Female': (147, 20, 255)  # Pink
        }
        
        return gender_colors.get(gender, (255, 255, 255))  # White as default
