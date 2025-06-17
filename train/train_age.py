import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import Huber
import matplotlib.pyplot as plt
import sys
import cv2
from tensorflow.keras.utils import Sequence
import json

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.detector import UTKFaceDataset

# Constants
DATA_DIR = os.getenv("DATA_DIR", "data/output/all")
MODEL_PATH = os.getenv("MODEL_PATH", "models/age_model.h5")

def create_age_model(input_shape=(64, 64, 3)):
    """
    Create CNN model for age prediction
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01), input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.25),
        Dense(1, activation='linear')  # Linear activation for age regression
    ])
    return model

def plot_training_history(history):
    """
    Plot training and validation loss and MAE as line plots
    """
    plt.figure(figsize=(12, 6))

    # Plot for loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', color='blue', linestyle='--')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot for MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE', color='green', linestyle='--')
    plt.plot(history.history['val_mae'], label='Validation MAE', color='red')
    plt.title('Model MAE Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history_age_plots.png')
    plt.close()

def save_training_history(history, config, history_path='training_history.json', config_path='training_config.json'):
    """
    Save training history and configuration to JSON files.
    """
    # Save training history
    with open(history_path, 'w') as f:
        json.dump({
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'mae': history.history['mae'],
            'val_mae': history.history['val_mae']
        }, f, indent=4)

    # Save training configuration
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def main():
    # Parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    IMG_SIZE = (64, 64)

    # Training configuration
    config = {
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'img_size': IMG_SIZE,
        'data_dir': DATA_DIR,
        'model_path': MODEL_PATH,
        'optimizer': 'RMSprop',
        'learning_rate': 0.00005,
        'loss': 'Huber',
        'metrics': ['mae']
    }

    # Create data generators using UTKFaceDataset
    from utils.detector import UTKFaceDataset
    dataset = UTKFaceDataset(DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE, mode='age')
    train_size = int(0.8 * len(dataset))
    train_dataset = UTKFaceDataset(DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE, mode='age')
    train_dataset.img_paths = dataset.img_paths[:train_size]
    val_dataset = UTKFaceDataset(DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE, mode='age')
    val_dataset.img_paths = dataset.img_paths[train_size:]

    # Create and compile model
    model = create_age_model()
    model.compile(
        optimizer=RMSprop(learning_rate=config['learning_rate']),  # Reduced learning rate
        loss=Huber(delta=1.0),  # Huber loss with delta=1.0
        metrics=config['metrics']
    )
    model.summary()

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            verbose=1,
            min_lr=1e-6
        )
    ]

    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history and configuration
    save_training_history(history, config)

    # Plot training history
    plot_training_history(history)
    print(f"\nTraining completed. Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()

# Normalize age labels to [0, 1] in UTKFaceDataset
class UTKFaceDataset(Sequence):
    def __getitem__(self, idx):
        batch_paths = self.img_paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_imgs = []
        batch_labels = []

        for path in batch_paths:
            try:
                img = cv2.imread(path)
                if img is None:
                    continue

                img = cv2.resize(img, self.img_size)
                img = img.astype('float32')
                img = (img - np.mean(img)) / np.std(img)  # Zero mean and unit variance

                filename = os.path.basename(path)
                if self.mode == 'age':
                    label = int(filename.split('_')[0]) / 100.0  # Normalize age to [0, 1]

                batch_imgs.append(img)
                batch_labels.append(label)

            except Exception as e:
                continue

        batch_imgs = np.array(batch_imgs)
        batch_labels = np.array(batch_labels)

        return batch_imgs, batch_labels