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
        self.fps = 30
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
    def process_frame(self, frame):
        """Process a single frame"""
        # Resize frame for consistent processing
        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
        
        # Create a copy for drawing
        display = frame.copy()
        
        try:
            # Detect faces
            faces = self.gender_predictor.detect_faces(frame)
            
            # Process each face
            for (x, y, w, h) in faces:
                # Extract face region
                face_img = frame[y:y+h, x:x+w]
                
                # Get predictions with confidence scores
                age, gender, gender_conf = self.age_gender_predictor.predict_age_gender(face_img)
                emotion, emotion_conf = self.emotion_predictor.predict_emotion(face_img, return_confidence=True)
                
                # Get colors for visualization
                emotion_color = self.emotion_predictor.get_emotion_color(emotion)
                gender_color = self.gender_predictor.get_gender_color(gender)
                
                # Draw face rectangle
                cv2.rectangle(display, (x, y), (x+w, y+h), emotion_color, 2)
                
                # Create labels with confidence scores
                labels = [
                    f"Age: {age}",
                    f"{gender} ({gender_conf:.2f})",
                    f"{emotion} ({emotion_conf:.2f})"
                ]
                
                # Draw labels with background
                y_offset = y - 10
                for i, label in enumerate(labels):
                    # Get color based on label type
                    color = gender_color if "Male" in label or "Female" in label else emotion_color
                    
                    # Get text size
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, self.font, self.font_scale, self.thickness
                    )
                    
                    # Draw background rectangle
                    cv2.rectangle(
                        display,
                        (x, y_offset - text_height - baseline),
                        (x + text_width, y_offset + baseline),
                        (0, 0, 0),
                        cv2.FILLED
                    )
                    
                    # Draw text
                    cv2.putText(
                        display,
                        label,
                        (x, y_offset),
                        self.font,
                        self.font_scale,
                        color,
                        self.thickness
                    )
                    
                    y_offset -= text_height + 2*baseline
                
            # Add FPS counter
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(
                display,
                fps_text,
                (10, 30),
                self.font,
                1,
                (0, 255, 0),
                2
            )
            
            return display
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return frame

def main():
    # Initialize FaceAnalyzer
    analyzer = FaceAnalyzer()
    
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, analyzer.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, analyzer.frame_height)
    
    # Variables for FPS calculation
    prev_time = cv2.getTickCount()
    
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera")
            break
            
        # Calculate FPS
        current_time = cv2.getTickCount()
        analyzer.fps = cv2.getTickFrequency() / (current_time - prev_time)
        prev_time = current_time
        
        # Process frame
        display = analyzer.process_frame(frame)
        
        # Show result
        cv2.imshow("Real-time Face Analysis", display)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = cv2.getTickCount()
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, display)
            print(f"Screenshot saved as {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()