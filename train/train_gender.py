import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import sys
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.regularizers import l2
import json
from tensorflow.keras.optimizers.schedules import CosineDecay
import tensorflow.keras.backend as K
import matplotlib.cm as cm
import tensorflow as tf  # Import TensorFlow
import cv2  # Import OpenCV

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.detector import UTKFaceDataset

# Constants
DATA_DIR = os.getenv("DATA_DIR", "data")

def create_gender_model(input_shape=(64, 64, 3)):
    """
    Create CNN model for gender classification with EfficientNetB0 backbone and fine-tuning
    """
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=input_shape)

    # Fine-tune các lớp cuối cùng
    for layer in base_model.layers[-50:]:  # Mở rộng fine-tune thêm 50 lớp cuối
        layer.trainable = True

    # Add Dropout to base model layers
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            layer = tf.keras.layers.Dropout(0.2)(layer.output)  # Add Dropout to Conv2D layers

    # Increase L2 regularization
    # Add additional Dense layer to increase model complexity
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.03)),
        Dropout(0.6),
        Dense(128, activation='relu', kernel_regularizer=l2(0.03)),  # New Dense layer
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    return model

def save_training_history(history, history_path='training_history_gender.json'):
    """
    Save training history to a JSON file.
    """
    with open(history_path, 'w') as f:
        json.dump({
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy']
        }, f, indent=4)

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss as line plots
    """
    plt.figure(figsize=(12, 6))

    # Plot for accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue', linestyle='--')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Plot for loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', color='green', linestyle='--')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_history_gender_plots.png')
    plt.close()

def compute_class_weights(train_dataset):
    """
    Compute class weights for imbalanced datasets
    """
    class_totals = np.bincount(train_dataset.classes)
    total_samples = sum(class_totals)
    class_weights = {i: total_samples / (len(class_totals) * class_totals[i]) for i in range(len(class_totals))}
    print(f"[DEBUG] Class weights: {class_weights}")
    return class_weights

def grad_cam(model, img_array, layer_name):
    """
    Generate Grad-CAM heatmap for a given image and model
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.dot(conv_outputs[0], weights)

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam

def plot_grad_cam(img, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on the original image
    """
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    overlay = jet_heatmap * alpha + img
    return np.uint8(overlay)

def main():
    # Parameters
    BATCH_SIZE = 32
    EPOCHS = 50
    IMG_SIZE = (64, 64)
    DATA_DIR = r"C:\Users\nguye\Desktop\ComputerVison\face_estimation\data\ref"
    MODEL_PATH = r"C:\Users\nguye\Desktop\ComputerVison\face_estimation\models\gender_model.h5"

    # Ensure base_model is defined consistently at the start of main
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

    # Enhance data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=50,  # Increase rotation range
        width_shift_range=0.5,  # Increase width shift
        height_shift_range=0.5,  # Increase height shift
        shear_range=0.5,  # Increase shear range
        zoom_range=0.5,  # Increase zoom range
        brightness_range=[0.6, 1.4],  # Expand brightness range
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=lambda x: x + np.random.normal(0, 0.05, x.shape)  # Add Gaussian noise
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    # Create dataset
    train_dataset = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_dataset = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    # Compute class weights
    class_weights = compute_class_weights(train_dataset)

    # Create and compile model
    model = create_gender_model()
    lr_schedule = CosineDecay(initial_learning_rate=0.001, decay_steps=EPOCHS * len(train_dataset), alpha=0.1)  # Increase learning rate
    optimizer = Adam(learning_rate=lr_schedule)

    # Replace binary_crossentropy with focal loss
    def focal_loss(gamma=2., alpha=0.25):
        def focal_loss_fixed(y_true, y_pred):
            epsilon = K.epsilon()
            y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
            pt = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
            return -K.mean(alpha * K.pow(1. - pt, gamma) * K.log(pt))
        return focal_loss_fixed

    model.compile(
        optimizer=optimizer,
        loss=focal_loss(),  # Use focal loss
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
            monitor='val_accuracy',
            patience=15,  # Increase patience
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
        class_weight=class_weights,
        verbose=1
    )

    # Save training history
    save_training_history(history)

    # Plot training history
    plot_training_history(history)
    print(f"\nTraining completed. Model saved to {MODEL_PATH}")

    # Experiment with different learning rates
    learning_rates = [ 0.0005]
    best_lr = None
    best_val_accuracy = 0

    for lr in learning_rates:
        print(f"\n[INFO] Training with learning rate: {lr}")
        lr_schedule = CosineDecay(initial_learning_rate=lr, decay_steps=EPOCHS * len(train_dataset), alpha=0.1)
        optimizer = Adam(learning_rate=lr_schedule)

        model.compile(
            optimizer=optimizer,
            loss=focal_loss(),
            metrics=['accuracy']
        )

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        val_accuracy = max(history.history['val_accuracy'])
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_lr = lr

    print(f"\n[INFO] Best learning rate: {best_lr} with validation accuracy: {best_val_accuracy:.2f}")

    # Increase epochs
    EPOCHS = 100  # Increase maximum epochs

    # Add additional Dropout layers
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu', kernel_regularizer=l2(0.03)),
        Dropout(0.6),
        Dense(128, activation='relu', kernel_regularizer=l2(0.03)),
        Dropout(0.5),
        Dense(64, activation='relu', kernel_regularizer=l2(0.03)),  # New Dense layer with Dropout
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])

    # Dự đoán gender với ngưỡng confidence mới
    # Fix mapping logic for gender prediction
    # Define preprocessed_face immediately before prediction
    preprocessed_face = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'val'),
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode=None
    )

    # Predict gender probabilities
    gender_prob = model.predict(preprocessed_face, verbose=0)[0]  # Predict probabilities
    gender = "Female" if gender_prob[0] >= 0.5 else "Male"  # Ensure correct mapping
    gender_conf = float(max(gender_prob[0], 1 - gender_prob[0]))

    # Add detailed debug for probabilities
    print(f"[DEBUG] Gender raw probabilities: {gender_prob}")
    print(f"[DEBUG] Predicted Gender: {gender}, Confidence: {gender_conf:.2f}")

    # Handle uncertain cases
    if gender_conf < 0.5:  # Adjust confidence threshold
        gender = "Uncertain"
        print(f"[DEBUG] Gender confidence too low, setting to 'Uncertain'")

    # Set learning rate to the best value
    lr_schedule = CosineDecay(initial_learning_rate=0.0005, decay_steps=EPOCHS * len(train_dataset), alpha=0.1)
    optimizer = Adam(learning_rate=lr_schedule)

if __name__ == "__main__":
    main()