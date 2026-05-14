"""
Fine-tune model on new images + Data Augmentation
Option 1: Fine-tune existing model on new images
Option 3: Data augmentation to improve generalization
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

# =============================================================================
# CONFIGURATION
# =============================================================================
MODEL_PATH = "./brain_tumor_model.keras"
NEW_DATA_PATH = "./new test images"
TARGET_SIZE = (64, 64)
BATCH_SIZE = 16
EPOCHS = 30

# =============================================================================
# PREPROCESSING
# =============================================================================
def preprocess(image, target_size=TARGET_SIZE):
    """Preprocess image for model."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((8, 8), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    brain_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [brain_contour], -1, 255, -1)
    result = cv2.bitwise_and(image, image, mask=mask)
    if np.mean(result) < 10:
        return image
    resized = cv2.resize(result, target_size)
    return resized / 255.0


def load_new_data(path):
    """Load and preprocess new images."""
    images = []
    filenames = []

    for filename in sorted(os.listdir(path)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img = cv2.imread(os.path.join(path, filename))
            if img is not None:
                processed = preprocess(img)
                images.append(processed)
                filenames.append(filename)

    return np.array(images), filenames


def augment_data(X, y, augment_factor=5):
    """
    Data augmentation to increase effective dataset size.
    Creates variations of each image.
    """
    augmented_X = [X]
    augmented_y = [y]

    for _ in range(augment_factor):
        X_aug = []
        for img in X:
            # Random rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            img_rot = np.rot90(img, k)

            # Random flip
            if np.random.rand() > 0.5:
                img_rot = np.flip(img_rot, axis=1)

            # Random brightness
            if np.random.rand() > 0.5:
                brightness = np.random.uniform(0.8, 1.2)
                img_rot = np.clip(img_rot * brightness, 0, 1)

            X_aug.append(img_rot)

        augmented_X.append(np.array(X_aug))
        augmented_y.append(y)

    return np.concatenate(augmented_X), np.concatenate(augmented_y)


# =============================================================================
# SPLIT DATA
# =============================================================================
def train_test_split_data(X, y, filenames, test_ratio=0.2):
    """Split data keeping some for testing."""
    n = len(X)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)

    split_idx = int(n * (1 - test_ratio))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    test_files = [filenames[i] for i in test_idx]

    return X_train, y_train, X_test, y_test, test_files


# =============================================================================
# MAIN
# =============================================================================
print("=" * 70)
print("FINE-TUNING MODEL ON NEW IMAGES")
print("=" * 70)

# Load pre-trained model
print("\nLoading pre-trained model...")
model = load_model(MODEL_PATH)
print("Model loaded!")

# Load new data
print("\nLoading new images...")
X, filenames = load_new_data(NEW_DATA_PATH)
print(f"Loaded {len(X)} images")

# All new images are positive (tumors)
y = np.ones(len(X))

print(f"\nOriginal dataset size: {len(X)}")

# Split into train/test
X_train, y_train, X_test, y_test, test_files = train_test_split_data(X, y, filenames, test_ratio=0.2)
print(f"Training set: {len(X_train)} images")
print(f"Test set: {len(X_test)} images")

# Apply data augmentation
print("\nApplying data augmentation (5x)...")
X_train_aug, y_train_aug = augment_data(X_train, y_train, augment_factor=5)
print(f"Augmented training set: {len(X_train_aug)} images")

# Create data generators for additional augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

# Fine-tune the model
print("\n" + "-" * 50)
print("FINE-TUNING MODEL")
print("-" * 50)

# Freeze early layers and fine-tune
for layer in model.layers[:-3]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.0001),  # Lower LR for fine-tuning
    loss='binary_crossentropy',
    metrics=['accuracy', AUC(), Precision(), Recall()]
)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

# Train
print("\nStarting fine-tuning...")
start_time = time.time()

history = model.fit(
    datagen.flow(X_train_aug, y_train_aug, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    callbacks=[early_stop],
    verbose=1
)

train_time = time.time() - start_time

# Save fine-tuned model
model.save('brain_tumor_model_finetuned.keras')
print(f"\n[OK] Fine-tuned model saved: brain_tumor_model_finetuned.keras")
print(f"Training time: {train_time:.1f} seconds")

# =============================================================================
# EVALUATE ON HELD-OUT TEST SET
# =============================================================================
print("\n" + "=" * 70)
print("EVALUATION ON HELD-OUT TEST SET")
print("=" * 70)

y_pred_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

correct = np.sum(y_pred == y_test)
incorrect = len(y_test) - correct
accuracy = correct / len(y_test)

print(f"\nTest Set Size: {len(y_test)}")
print(f"Correct: {correct} | Incorrect: {incorrect}")
print(f"Accuracy: {accuracy:.1%}")

print("\nPer-image results:")
for i, (prob, pred, actual, fname) in enumerate(zip(y_pred_prob, y_pred, y_test, test_files)):
    status = "TUMOR" if pred == 1 else "NO TUMOR"
    correct_str = "[OK]" if pred == actual else "[WRONG]"
    print(f"  {correct_str} {fname}: {status} (Conf: {prob:.1%})")

# =============================================================================
# TEST ON FULL NEW DATASET
# =============================================================================
print("\n" + "=" * 70)
print("EVALUATION ON FULL NEW DATASET")
print("=" * 70)

y_full_prob = model.predict(X, verbose=0).flatten()
y_full_pred = (y_full_prob > 0.5).astype(int)

correct_full = np.sum(y_full_pred == y)
accuracy_full = correct_full / len(y)

print(f"\nFull Dataset: {len(y)} images (all with tumors)")
print(f"Correctly Detected: {correct_full} ({accuracy_full:.1%})")
print(f"Misclassified: {len(y) - correct_full} ({100-accuracy_full:.1%})")

# Show misclassified
print("\nMisclassified images:")
for i, (prob, pred, fname) in enumerate(zip(y_full_prob, y_full_pred, filenames)):
    if pred == 0:
        print(f"  - {fname}: Confidence = {prob:.1%}")

# =============================================================================
# PLOT TRAINING HISTORY
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

metrics = [('accuracy', 'val_accuracy'), ('loss', 'val_loss'),
           ('auc', 'val_auc'), ('recall', 'val_recall')]

for ax, (train_metric, val_metric) in zip(axes.flat, metrics):
    ax.plot(history.history[train_metric], label='Train')
    ax.plot(history.history[val_metric], label='Validation')
    ax.set_title(train_metric.capitalize())
    ax.set_xlabel('Epoch')
    ax.legend()
    ax.grid(True)

plt.suptitle('Fine-Tuning Training History', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('finetune_history.png', dpi=150)
plt.close()
print("\n[OK] Training history saved: finetune_history.png")

print("\n" + "=" * 70)
print("FINE-TUNING COMPLETE")
print("=" * 70)
