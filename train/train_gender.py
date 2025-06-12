import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import sys

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.detector import UTKFaceDataset

def create_gender_model(input_shape=(64, 64, 3)):
    """
    Create CNN model for gender classification
    Args:
        input_shape: Input image dimensions
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # First Conv Block
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Second Conv Block
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Third Conv Block
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Dense Layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification (Male/Female)
    ])
    
    return model

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    """
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history_gender.png')
    plt.close()

def main():
    # Parameters
    BATCH_SIZE = 32
    EPOCHS = 30
    IMG_SIZE = (64, 64)
    DATA_DIR = "../data/output"
    MODEL_PATH = "../models/gender_model.h5"
    
    # Create dataset
    dataset = UTKFaceDataset(DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    train_dataset = UTKFaceDataset(DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    train_dataset.img_paths = dataset.img_paths[:train_size]
    
    val_dataset = UTKFaceDataset(DATA_DIR, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    val_dataset.img_paths = dataset.img_paths[train_size:]
    
    # Create and compile model
    model = create_gender_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
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
            factor=0.2,
            patience=5,
            min_lr=1e-6,
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