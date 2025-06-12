import os
import cv2
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical

class UTKFaceDataset(Sequence):
    def __init__(self, folder, batch_size=32, img_size=(64, 64), mode='age'):
        """
        Initialize UTKFace dataset
        Args:
            folder: Path to dataset folder
            batch_size: Batch size for training
            img_size: Input image size
            mode: 'age', 'gender', or 'emotion' for different tasks
        """
        self.folder = folder
        self.batch_size = batch_size
        self.img_size = img_size
        self.mode = mode
        self.img_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
        
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

        self.on_epoch_end()

    def __len__(self):
        return len(self.img_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_paths = self.img_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_imgs = []
        batch_labels = []

        for path in batch_paths:
            try:
                # Load and preprocess image
                img = cv2.imread(path)
                if img is None:
                    continue
                    
                img = cv2.resize(img, self.img_size)
                img = img.astype('float32') / 255.0
                
                # Get labels from filename
                filename = os.path.basename(path)
                if self.mode == 'age':
                    # Age is the first component in filename: [age]_[gender]_[race]_[date].jpg
                    label = int(filename.split('_')[0])
                
                elif self.mode == 'gender':
                    # Gender is the second component: 0 for male, 1 for female
                    label = int(filename.split('_')[1])
                    
                elif self.mode == 'emotion':
                    # For emotion, use directory name as label
                    emotion = os.path.basename(os.path.dirname(path)).lower()
                    if emotion in self.emotion_map:
                        label = self.emotion_map[emotion]
                    else:
                        continue
                
                batch_imgs.append(img)
                batch_labels.append(label)
                
            except Exception as e:
                print(f"Error processing {path}: {str(e)}")
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
