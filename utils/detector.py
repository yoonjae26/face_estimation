import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class UTKFaceDataset(Sequence):
    def __init__(self, folder, batch_size=32, img_size=(64, 64), mode='age', **kwargs):
        """
        Initialize UTKFace dataset
        Args:
            folder: Path to dataset folder
            batch_size: Batch size for training
            img_size: Input image size
            mode: 'age', 'gender', or 'emotion' for different tasks
        """
        super().__init__(**kwargs)  # Ensure compatibility with TensorFlow
        self.folder = folder
        self.batch_size = batch_size
        self.img_size = img_size
        self.mode = mode

        # Recursively find all .jpg files in subdirectories
        self.img_paths = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.jpg'):
                    self.img_paths.append(os.path.join(root, file))

        # Log the number of images found
        logging.info(f"Found {len(self.img_paths)} images in directory: {folder}")

        # Emotion mapping
        self.emotion_map = {
            'Angry': 0,
            'Happy': 1,
            'Sad': 2,
            'Disgust': 3,
            'Fear': 4,
            'Surprise': 5,
            'Neutral': 6
        }

        # Check if any valid images are found
        if not self.img_paths:
            logging.error(f"No valid images found in directory: {folder}")
            raise ValueError(f"No valid images found in directory: {folder}")

        self.on_epoch_end()

    def __len__(self):
        # Ensure length is at least 1 if there are images
        return max(1, len(self.img_paths) // self.batch_size)

    def __getitem__(self, idx):
        batch_paths = self.img_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_imgs = []
        batch_labels = []

        for path in batch_paths:
            try:
                # Load and preprocess image
                img = cv2.imread(path)
                if img is None:
                    logging.warning(f"Image at path {path} could not be loaded.")
                    continue

                img = cv2.resize(img, self.img_size)
                img = img.astype('float32') / 255.0

                # Get labels from filename
                filename = os.path.basename(path)
                logging.info(f"Processing file: {filename}")

                if self.mode == 'age':
                    try:
                        label = int(filename.split('_')[0])
                    except ValueError:
                        logging.error(f"Failed to extract age from filename: {filename}")
                        continue

                elif self.mode == 'gender':
                    subfolder = os.path.basename(os.path.dirname(path)).lower()
                    label = 0 if subfolder == 'male' else 1 if subfolder == 'female' else None
                    if label is None:
                        logging.error(f"Invalid gender label for file: {path}")
                        continue

                elif self.mode == 'emotion':
                    emotion = os.path.basename(os.path.dirname(path)).lower()
                    if emotion in self.emotion_map:
                        label = self.emotion_map[emotion]
                    else:
                        logging.error(f"Invalid emotion label in filename: {filename}")
                        continue

                batch_imgs.append(img)
                batch_labels.append(label)

            except Exception as e:
                logging.error(f"Error processing {path}: {str(e)}")
                continue

        batch_imgs = np.array(batch_imgs)
        batch_labels = np.array(batch_labels)

        # Convert labels to appropriate format
        if self.mode == 'emotion':
            batch_labels = to_categorical(batch_labels, num_classes=7)
        elif self.mode == 'gender':
            batch_labels = to_categorical(batch_labels, num_classes=2)

        return batch_imgs, batch_labels

    def on_epoch_end(self):
        """Shuffle dataset at the end of every epoch"""
        np.random.shuffle(self.img_paths)

class FaceDetector:
    def __init__(self):
        """Initialize face detector using OpenCV's face detection"""
        # Load face detection cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
    def detect_faces(self, image):
        """
        Detect faces in image
        Args:
            image: Input image (BGR format)
        Returns:
            List of (x, y, w, h) face rectangles
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces
    
    def extract_face(self, image, face_rect, target_size=(64, 64)):
        """
        Extract and preprocess face region from image
        Args:
            image: Input image
            face_rect: (x, y, w, h) face rectangle
            target_size: Size to resize face to
        Returns:
            Preprocessed face image
        """
        x, y, w, h = face_rect
        
        # Extract face ROI
        face = image[y:y+h, x:x+w]
        
        # Resize to target size
        face = cv2.resize(face, target_size)
        
        # Normalize pixel values
        face = face.astype('float32') / 255.0
        
        return face

def create_data_generators(data_dir, img_size=(64, 64), batch_size=32, mode='age'):
    """
    Create train and validation data generators
    Args:
        data_dir: Directory containing the dataset
        img_size: Input image size
        batch_size: Batch size
        mode: 'age', 'gender', or 'emotion'
    Returns:
        train_generator, val_generator
    """
    dataset = UTKFaceDataset(data_dir, batch_size=batch_size, img_size=img_size, mode=mode)
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    
    train_dataset = UTKFaceDataset(data_dir, batch_size=batch_size, img_size=img_size, mode=mode)
    train_dataset.img_paths = dataset.img_paths[:train_size]
    
    val_dataset = UTKFaceDataset(data_dir, batch_size=batch_size, img_size=img_size, mode=mode)
    val_dataset.img_paths = dataset.img_paths[train_size:]
    
    return train_dataset, val_dataset
