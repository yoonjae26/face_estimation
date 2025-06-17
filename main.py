import cv2
import numpy as np
from utils.age_gender_predictor import AgeGenderPredictor
from utils.emotion_predictor import EmotionPredictor
from utils.gender_predictor import GenderPredictor

class FaceAnalyzer:
    def __init__(self):
        """Initialize all predictors and parameters"""
        self.age_gender_predictor = AgeGenderPredictor()
        self.emotion_predictor = EmotionPredictor()
        self.gender_predictor = GenderPredictor(use_mtcnn=False)  # Use Haar Cascade for speed

        # Initialize video capture parameters
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 0

        # Font settings
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = 0.6
        self.thickness = 2

    def preprocess_frame(self, frame):
        """
        Preprocess the input frame for all models
        Args:
            frame (numpy.ndarray): Input frame (BGR or grayscale format)
        Returns:
            numpy.ndarray: Preprocessed frame (RGB format)
        """
        try:
            # Ensure the frame has 3 channels
            if len(frame.shape) == 2 or frame.shape[-1] != 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            # Resize to model input size
            frame = cv2.resize(frame, (64, 64))

            # Normalize pixel values
            frame = frame.astype('float32') / 255.0

            # Add batch dimension
            frame = np.expand_dims(frame, axis=0)

            return frame
        except Exception as e:
            raise Exception(f"Error in frame preprocessing: {str(e)}")

    def process_frame(self, frame):
        """Process a single frame"""
        display = frame.copy()

        try:            # Detect faces
            faces = self.gender_predictor.detect_faces(frame)
              # Debug information
            print(f"Detected {len(faces)} faces")
            
            for (x, y, w, h) in faces:
                # Ensure bounding box is within frame bounds
                h_frame, w_frame = frame.shape[:2]
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(x + w, w_frame)
                y2 = min(y + h, h_frame)
                
                # Skip invalid crops
                if x2 <= x1 or y2 <= y1 or (x2 - x1) < 20 or (y2 - y1) < 20:
                    print(f"Skipping small face: {x2-x1}x{y2-y1}")
                    continue
                    
                # Extract face region
                face_img = frame[y1:y2, x1:x2]
                if face_img is None or face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
                    print(f"Invalid face crop shape: {face_img.shape if face_img is not None else 'None'}")
                    continue
                
                print(f"Face crop shape: {face_img.shape}")

                # Convert BGR to RGB
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                
                # Resize to common size for all models
                face_resized = cv2.resize(face_rgb, (64, 64))
                
                # Normalize pixel values
                face_normalized = face_resized.astype('float32') / 255.0
                
                # Add batch dimension for model input
                preprocessed_face = np.expand_dims(face_normalized, axis=0)

                # Get predictions
                age, gender, gender_conf = self.age_gender_predictor.predict_age_gender(preprocessed_face)
                emotion, emotion_conf = self.emotion_predictor.predict_emotion(preprocessed_face, return_confidence=True)

                # Draw face rectangle
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Create labels
                labels = [
                    f"Age: {age}",
                    f"{gender} ({gender_conf:.2f})",
                    f"{emotion} ({emotion_conf:.2f})"
                ]

                y_offset = y - 10
                for label in labels:
                    (text_width, text_height), baseline = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
                    cv2.rectangle(display, (x, y_offset - text_height - baseline), (x + text_width, y_offset + baseline), (0, 0, 0), cv2.FILLED)
                    cv2.putText(display, label, (x, y_offset), self.font, self.font_scale, (255, 255, 255), self.thickness)
                    y_offset -= text_height + 2 * baseline

            return display
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return frame

    def calculate_fps(self, prev_time):
        """Calculate FPS"""
        current_time = cv2.getTickCount()
        self.fps = cv2.getTickFrequency() / (current_time - prev_time)
        return current_time

def process_image_file(image_path):
    analyzer = FaceAnalyzer()
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot read image from path: {image_path}")
        return
    display = img.copy()
    faces = analyzer.gender_predictor.detect_faces(img)
    print(f"Detected {len(faces)} faces in the image.")
    for idx, (x, y, w, h) in enumerate(faces):
        h_frame, w_frame = img.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(x + w, w_frame)
        y2 = min(y + h, h_frame)
        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 20 or (y2 - y1) < 20:
            continue
        face_img = img[y1:y2, x1:x2]
        if face_img is None or face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
            continue
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (64, 64))
        face_normalized = face_resized.astype('float32') / 255.0
        preprocessed_face = np.expand_dims(face_normalized, axis=0)
        try:
            # Preprocess cho từng model
            try:
                # Age/Gender: dùng đúng hàm preprocess_image của AgeGenderPredictor (ảnh BGR)
                age_gender_input = analyzer.age_gender_predictor.preprocess_image(face_img)
                # Emotion: dùng đúng hàm preprocess_image của EmotionPredictor (ảnh RGB)
                emotion_input = analyzer.emotion_predictor.preprocess_image(face_img)

                # Dự đoán gender với ảnh BGR gốc
                gender_prob = analyzer.age_gender_predictor.gender_model.predict(age_gender_input, verbose=0)[0]
                gender = "Female" if gender_prob[0] < 0.5 else "Male"
                gender_conf = float(max(gender_prob[0], 1 - gender_prob[0]))
                print(f"[DEBUG] Gender raw output: {gender_prob}, Gender: {gender}, Confidence: {gender_conf}")

                # Dự đoán age
                age = analyzer.age_gender_predictor.age_model.predict(age_gender_input, verbose=0)[0][0]
                # Scale lại giá trị tuổi nếu cần
                if age < 1.5:  # Nếu giá trị nhỏ, có thể cần nhân với 100
                    age *= 100
                age = int(round(age))  # Làm tròn giá trị tuổi
                print(f"[DEBUG] Age raw output: {age}")

                # Dự đoán gender với ngưỡng confidence
                gender_prob = analyzer.age_gender_predictor.gender_model.predict(age_gender_input, verbose=0)[0]
                gender = "Male" if gender_prob[0] > 0.5 else "Female"
                gender_conf = float(max(gender_prob[0], 1 - gender_prob[0]))

                # Thêm debug chi tiết
                print(f"[DEBUG] Gender raw probabilities: {gender_prob}")
                print(f"[DEBUG] Predicted Gender: {gender}, Confidence: {gender_conf:.2f}")

                # Xử lý trường hợp không chắc chắn
                if gender_conf < 0.7:  # Tăng ngưỡng confidence lên 0.7
                    gender = "Uncertain"
                    print(f"[DEBUG] Gender confidence too low, setting to 'Uncertain'")

                # Dự đoán emotion
                emotion, emotion_conf = analyzer.emotion_predictor.predict_emotion(face_img, return_confidence=True)

                print(f"[DEBUG] Age/Gender input shape: {age_gender_input.shape}, Emotion input shape: {emotion_input.shape}")
                print(f"Face {idx+1}: Age: {age}, Gender: {gender} ({gender_conf:.2f}), Emotion: {emotion} ({emotion_conf:.2f})")
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{gender}({gender_conf:.2f}), {age}, {emotion}({emotion_conf:.2f})"
                cv2.putText(display, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            except Exception as e:
                print(f"Error predicting face {idx+1}: {str(e)}")
        except Exception as e:
            print(f"Error predicting face {idx+1}: {str(e)}")
    cv2.imshow("Image Analysis Result", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    analyzer = FaceAnalyzer()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, analyzer.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, analyzer.frame_height)

    prev_time = cv2.getTickCount()

    print("Press 'q' to quit")
    print("Press 's' to save screenshot")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera")
            break

        # Preprocess the frame to ensure it is in RGB format
        try:
            if len(frame.shape) == 2 or frame.shape[-1] != 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        except Exception as e:
            print(f"Error converting frame to RGB: {str(e)}")
            break

        prev_time = analyzer.calculate_fps(prev_time)
        display = analyzer.process_frame(frame)

        cv2.imshow("Real-time Face Analysis", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = cv2.getTickCount()
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, display)
            print(f"Screenshot saved as {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Sample image path
    image_path = "\data\1.jpg"
    process_image_file(image_path)
    # To run real-time analysis, uncomment the line below
    # main()