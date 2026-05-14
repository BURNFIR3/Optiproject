"""
Brain Tumour Detection - Optimized for CPU Training
- Skull stripping preprocessing
- Small dataset for quick iteration
- Export ready for web deployment
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import random

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Evaluation
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_curve, roc_auc_score, accuracy_score

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_PATH = "./Brain_Tumor_Dataset"
TARGET_SIZE = (64, 64)  # Small for fast CPU training
BATCH_SIZE = 32
EPOCHS = 20
TRAIN_PER_CLASS = 1500
VAL_PER_CLASS = 200
TEST_PER_CLASS = 200

# =============================================================================
# SKULL STRIPPING - Remove non-brain tissue from MRI
# =============================================================================
def skull_strip(image):
    """
    Remove skull and non-brain tissue from MRI using morphological operations.
    Returns stripped image or original if processing fails.
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold to get brain region (MRI typically has distinct intensity)
        _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean up
        kernel = np.ones((8, 8), np.uint8)
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)

        # Find largest contour (brain region)
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image

        # Get the largest contour (brain)
        brain_contour = max(contours, key=cv2.contourArea)

        # Create mask
        mask = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(mask, [brain_contour], -1, 255, -1)

        # Apply mask to original image
        result = cv2.bitwise_and(image, image, mask=mask)

        # Check if result is mostly black (processing failed)
        if np.mean(result) < 10:
            return image

        return result
    except Exception:
        return image


def preprocess_image(image, target_size=TARGET_SIZE):
    """Apply skull stripping and resize image."""
    stripped = skull_strip(image)
    resized = cv2.resize(stripped, target_size)
    normalized = resized / 255.0
    return normalized


# =============================================================================
# LOAD AND SPLIT DATA
# =============================================================================
def load_and_split_data(dataset_path, train_n, val_n, test_n):
    """Load images and split into train/val/test sets."""
    images = {'positive': [], 'negative': []}

    # Load positive samples
    positive_path = os.path.join(dataset_path, 'Positive')
    for filename in os.listdir(positive_path)[:train_n + val_n + test_n]:
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(positive_path, filename)
            img = cv2.imread(path)
            if img is not None:
                images['positive'].append(img)

    # Load negative samples
    negative_path = os.path.join(dataset_path, 'Negative')
    for filename in os.listdir(negative_path)[:train_n + val_n + test_n]:
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(negative_path, filename)
            img = cv2.imread(path)
            if img is not None:
                images['negative'].append(img)

    print(f"Loaded {len(images['positive'])} positive, {len(images['negative'])} negative images")

    # Shuffle and split each class
    random.seed(42)
    random.shuffle(images['positive'])
    random.shuffle(images['negative'])

    # Split
    train_pos = images['positive'][:train_n]
    val_pos = images['positive'][train_n:train_n + val_n]
    test_pos = images['positive'][train_n + val_n:train_n + val_n + test_n]

    train_neg = images['negative'][:train_n]
    val_neg = images['negative'][train_n:train_n + val_n]
    test_neg = images['negative'][train_n + val_n:train_n + val_n + test_n]

    # Combine
    train_images = train_pos + train_neg
    train_labels = [1] * len(train_pos) + [0] * len(train_neg)

    val_images = val_pos + val_neg
    val_labels = [1] * len(val_pos) + [0] * len(val_neg)

    test_images = test_pos + test_neg
    test_labels = [1] * len(test_pos) + [0] * len(test_neg)

    # Preprocess
    print("Preprocessing (skull stripping + resizing)...")
    X_train = np.array([preprocess_image(img) for img in train_images])
    X_val = np.array([preprocess_image(img) for img in val_images])
    X_test = np.array([preprocess_image(img) for img in test_images])

    y_train = np.array(train_labels)
    y_val = np.array(val_labels)
    y_test = np.array(test_labels)

    print(f"Training: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================================================
# BUILD MODEL - Lightweight for CPU
# =============================================================================
def build_model(input_shape=TARGET_SIZE + (3,)):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),

        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', AUC(), Precision(), Recall()]
    )

    return model


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_training_history(history):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    metrics = [('accuracy', 'val_accuracy'), ('loss', 'val_loss'),
               ('auc', 'val_auc'), ('recall', 'val_recall')]

    for ax, (train_metric, val_metric) in zip(axes.flat, metrics):
        ax.plot(history.history[train_metric], label=f'Train')
        ax.plot(history.history[val_metric], label=f'Validation')
        ax.set_title(train_metric.capitalize())
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    plt.show()


def plot_evaluation(y_true, y_pred, y_pred_prob):
    """Plot confusion matrix and ROC curve."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    axes[0].imshow(cm, cmap='Blues')
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(['No Tumour', 'Tumour'])
    axes[0].set_yticklabels(['No Tumour', 'Tumour'])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, cm[i, j], ha='center', va='center', fontsize=14)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc_val = roc_auc_score(y_true, y_pred_prob)
    axes[1].plot(fpr, tpr, label=f'Model (AUC = {auc_val:.2f})')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1].set_title('ROC Curve')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150)
    plt.show()


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================
def main():
    print("=" * 70)
    print("BRAIN TUMOUR DETECTION - FULL DATASET TRAINING")
    print("=" * 70)
    print(f"Config: {TRAIN_PER_CLASS} train, {VAL_PER_CLASS} val, {TEST_PER_CLASS} test per class")
    print(f"Image size: {TARGET_SIZE}, Batch: {BATCH_SIZE}, Epochs: {EPOCHS}")
    print(f"Total training samples: {TRAIN_PER_CLASS * 2}")
    print(f"Total val samples: {VAL_PER_CLASS * 2}")
    print(f"Total test samples: {TEST_PER_CLASS * 2}")
    print("=" * 70)

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data(
        DATASET_PATH, TRAIN_PER_CLASS, VAL_PER_CLASS, TEST_PER_CLASS
    )

    # Build model
    model = build_model()
    model.summary()

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )

    # Train
    print("\nStarting training...")
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f} seconds ({train_time/60:.1f} minutes)")

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION ON TEST SET")
    print("=" * 60)

    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Tumour', 'Tumour']))
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Visualizations
    plot_training_history(history)
    plot_evaluation(y_test, y_pred, y_pred_prob)

    # Save model for web deployment
    model.save('brain_tumor_model.keras')
    print("\n[OK] Model saved to 'brain_tumor_model.keras'")

    # Save preprocessing info
    with open('model_config.txt', 'w') as f:
        f.write(f"TARGET_SIZE={TARGET_SIZE}\n")
        f.write(f"CLASS_LABELS=['no_tumor', 'tumor']\n")
        f.write(f"THRESHOLD=0.5\n")

    print("[OK] Config saved to 'model_config.txt'")

    return model, history


if __name__ == '__main__':
    # Check for GPU (shouldn't use it anyway)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Warning: {len(gpus)} GPU(s) found. For CPU training, this is fine.")
    else:
        print("Running on CPU (as intended for fast local training)")

    model, history = main()
