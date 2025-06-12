import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import sys

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.detector import UTKFaceDataset

def create_age_model(input_shape=(64, 64, 3)):
    """
    Create CNN model for age prediction
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.25),
        Dense(1, activation='linear')  # Linear activation for age regression
    ])
    return model

def plot_training_history(history):
    """
    Plot training and validation loss
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history_age.png')
    plt.close()

def main():
    # Parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    IMG_SIZE = (64, 64)
    DATA_DIR = "../data/output"
    MODEL_PATH = "../models/age_model.h5"
    
    # Create data generators
    dataset = UTKFaceDataset(DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    train_dataset = UTKFaceDataset(DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    train_dataset.img_paths = dataset.img_paths[:train_size]
    
    val_dataset = UTKFaceDataset(DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    val_dataset.img_paths = dataset.img_paths[train_size:]
    
    # Create and compile model
    model = create_age_model()
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='mse',
        metrics=['mae']
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
    
    # Plot training history
    plot_training_history(history)
    print(f"\nTraining completed. Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()