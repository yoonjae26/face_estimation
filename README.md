# Face Estimation Project

This project is a real-time face analysis system that predicts age, gender, and emotion from camera input or static images. It includes a robust pipeline for preprocessing, inference, and visualization.

## Features
- Real-time face detection and analysis.
- Age, gender, and emotion prediction using pre-trained models.
- Unified preprocessing pipeline for consistent input.
- Debugging tools for model outputs and confidence scores.
- Training scripts for custom models with data augmentation and learning rate scheduling.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd face_estimation
   ```

2. Create a Python virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained models and place them in the `models/` directory:
   - `age_model.h5`
   - `gender_model.h5`
   - `emotion_model.h5`

## Usage

### Real-time Analysis
Run the following command to start real-time face analysis:
```bash
python main.py
```

### Static Image Analysis
To analyze a static image, modify the `image_path` variable in `main.py` and run:
```bash
python main.py
```

### Training
To train models, use the scripts in the `train/` directory:
- `train_age.py`
- `train_gender.py`
- `train_emotion.py`

Example:
```bash
python train/train_gender.py
```

## Models

### Age Model
- Predicts the age of a person from a face image.
- Output: Integer value representing the predicted age.

### Gender Model
- Predicts the gender of a person from a face image.
- Output: "Male" or "Female" with a confidence score.

### Emotion Model
- Predicts the emotion of a person from a face image.
- Output: Emotion label (e.g., "Happy", "Sad") with a confidence score.

## Notes
- Ensure the `data/` directory is structured correctly for training and inference.
- Use the `utils/` directory for helper functions and preprocessing.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
